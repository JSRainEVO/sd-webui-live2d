from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
import zipfile
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SCRIPTS = ("auto_live2d.py", "auto_setup.py", "layer_quality.py", "verify_model.cjs")


class BridgeError(RuntimeError):
    pass


class Cancelled(BridgeError):
    pass


def normalize_resolution(value) -> int:
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise ValueError
        size = int(number)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        raise BridgeError("拆层分辨率必须是正整数。") from None
    return ((size + 63) // 64) * 64


def default_engine() -> Path:
    # The development checkout can reuse its existing environment and weights.
    candidate = EXTENSION_ROOT.parent
    if all((candidate / name).is_file() for name in REQUIRED_SCRIPTS):
        return candidate
    return EXTENSION_ROOT / "engine"


def engine_root(value: str) -> Path:
    root = Path(value.strip().strip('"')).expanduser().resolve() if value.strip() else default_engine()
    if not all((root / name).is_file() for name in REQUIRED_SCRIPTS):
        raise BridgeError("转换环境目录无效，请选择包含 auto_live2d.py 的目录。")
    return root


def engine_command(root: Path, script: str, *args: str) -> list[str]:
    python = root / ".auto_env" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    return [str(python), "-u", str(root / script), *args]


def selected_infotext(info: str, index: int) -> str:
    if not info:
        return ""
    try:
        data = json.loads(info)
        index = max(0, int(index))
        if index < int(data.get("index_of_first_image", 0)):
            raise BridgeError("当前选中的是拼图，请先在结果中选择一张角色图片。")
        texts = data.get("infotexts", [])
        return str(texts[index]) if index < len(texts) else ""
    except (ValueError, TypeError, AttributeError):
        return ""


def save_input(image: Image.Image, directory: Path, infotext: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "character.png"
    metadata = PngInfo()
    parameters = infotext or image.info.get("parameters", "")
    if parameters:
        metadata.add_text("parameters", str(parameters))
    ImageOps.exif_transpose(image).convert("RGBA").save(source, pnginfo=metadata)
    (directory / "source_info.json").write_text(json.dumps(
        {"source": "Stable Diffusion WebUI", "generationParameters": str(parameters)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return source


def new_job_directory(root: Path) -> Path:
    return root / "outputs" / "webui" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def command_lines(command: list[str], cwd: Path,
                  cancel: threading.Event) -> Iterator[str | None]:
    """Stream an isolated engine; cancellation stops its inference children too."""
    if cancel.is_set():
        raise Cancelled("任务已停止")
    env = os.environ.copy()
    # Python package paths inherited from SD must not contaminate the engine.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONIOENCODING"] = "utf-8"
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", creationflags=flags, start_new_session=os.name != "nt")
    lines: queue.Queue[str | None] = queue.Queue()
    assert process.stdout is not None

    def reader() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while True:
            if cancel.is_set():
                raise Cancelled("任务已停止")
            try:
                line = lines.get(timeout=.3)
            except queue.Empty:
                yield None
                continue
            if line is None:
                break
            clean = line.strip()
            if clean:
                yield clean
        if process.wait():
            raise BridgeError("转换步骤失败，请展开运行记录查看原因。中间文件已保留。")
    finally:
        _terminate_tree(process)
        thread.join(timeout=5)
        process.stdout.close()


def pack_result(directory: Path) -> tuple[Path, Path | None, list[str]]:
    report = json.loads((directory / "conversion_report.json").read_text(encoding="utf-8"))
    model = directory / "model"
    if len(list(model.rglob("*.model3.json"))) != 1:
        raise BridgeError("未找到唯一的 Live2D 模型入口。")
    archive = directory / "live2d-model.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in model.rglob("*"):
            if file.is_file():
                bundle.write(file, Path("model") / file.relative_to(model))
        for name in ("conversion_report.json", "source_info.json"):
            file = directory / name
            if file.is_file():
                bundle.write(file, name)
        psd = Path(report["layeredPsd"])
        if psd.is_relative_to(directory.resolve()) and psd.is_file():
            bundle.write(psd, Path("source") / psd.name)
        bundle.writestr("使用说明.txt", "解压后保留 model 文件夹内的全部文件。模型入口为 .model3.json；.cmo3 为可编辑工程。请在目标软件里预览眨眼、张嘴、转头和遮挡。")
    audit = report.get("layerAudit") or {}
    preview = Path(audit["preview"]) if audit.get("preview") else None
    if preview and not preview.is_relative_to(directory.resolve()):
        preview = None
    return archive, preview if preview and preview.is_file() else None, report.get("warnings", [])


class Jobs:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Installation and conversion must not edit/use the same environment together.
        self.work_lock = threading.Lock()
        self.events: dict[str, threading.Event] = {}

    def create(self, previous: str | None = None) -> str:
        with self.lock:
            if previous in self.events:
                raise BridgeError("这个页面已有任务，请等待完成或先停止。")
            key = uuid.uuid4().hex
            self.events[key] = threading.Event()
            return key

    def event(self, key: str) -> threading.Event:
        with self.lock:
            if key not in self.events:
                raise BridgeError("任务已结束，请重新点击开始。")
            return self.events[key]

    def cancel(self, key: str | None) -> bool:
        with self.lock:
            event = self.events.get(key)
            if event is None:
                return False
            event.set()
            return True

    def finish(self, key: str) -> None:
        with self.lock:
            self.events.pop(key, None)
