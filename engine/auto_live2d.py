"""Single image or layered PSD -> portable Cubism model bundle.

The image path invokes See-through for semantic layers and occlusion fill.
Both paths invoke PSD2Live for rigging and model export. The two upstream
projects remain separate; this file only orchestrates their public CLIs.
"""

from __future__ import annotations

import argparse
import json
import locale
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
SEE_REPO = ROOT / "third_party" / "see-through"
PSD2LIVE_EXE = ROOT / "third_party" / "psd2live-portable" / "PSD2Live.exe"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_RESOLUTION = 512
MODEL_REPOS = ("24yearsold/seethroughv0.0.2_layerdiff3d_nf4",
               "24yearsold/seethroughv0.0.1_marigold_nf4")


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(PipelineError):
    pass


def normalize_resolution(value: int) -> int:
    """Align the square inference canvas without imposing a preset upper limit."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError("拆层分辨率必须是正整数")
    return ((value + 63) // 64) * 64


def prepare_canvas(source: Path, destination: Path, size: tuple[int, int]) -> Path:
    """Fit a character to the selected canvas without stretching or cropping it."""
    from PIL import Image, ImageOps

    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size)
    canvas.alpha_composite(image, ((size[0] - image.width) // 2,
                                   (size[1] - image.height) // 2))
    canvas.save(destination)
    return destination


def cached_model(repo: str, cache: Path | None = None) -> Path | None:
    """Use a complete local snapshot without making Hub metadata requests."""
    folder = (cache or ROOT / ".model-cache" / "hub") / ("models--" + repo.replace("/", "--"))
    try:
        revision = (folder / "refs" / "main").read_text(encoding="utf-8").strip()
        snapshot = folder / "snapshots" / revision
        index = json.loads((snapshot / "model_index.json").read_text(encoding="utf-8"))
        components = {name for name, spec in index.items()
                      if isinstance(spec, list) and len(spec) == 2 and spec[0]}
        components.add("unet")
        if repo == MODEL_REPOS[0]:
            components.add("trans_vae")
        for name in components:
            part = snapshot / name
            if name.startswith("tokenizer"):
                required = ["vocab.json", "merges.txt", "tokenizer_config.json"]
            elif name == "scheduler":
                required = ["scheduler_config.json"]
            else:
                required = ["config.json"]
                indexes = list(part.glob("*.index.json"))
                if indexes:
                    weight_map = json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"]
                    required.extend(set(weight_map.values()))
                elif not any(p.is_file() and p.stat().st_size for p in part.glob("*.safetensors")):
                    return None
            if any(not (part / p).is_file() or not (part / p).stat().st_size for p in required):
                return None
        return snapshot
    except (OSError, ValueError, KeyError, TypeError):
        return None


def see_python() -> Path | None:
    override = os.environ.get("SEE_THROUGH_PYTHON")
    candidates = [Path(override)] if override else []
    candidates += [ROOT / ".auto_env" / "Scripts" / "python.exe"]
    return next((path for path in candidates if path.is_file()), None)


def psd2live_exe() -> Path | None:
    override = os.environ.get("PSD2LIVE_EXE")
    candidates = [Path(override)] if override else []
    candidates.append(PSD2LIVE_EXE)
    found = shutil.which("PSD2Live")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def check_image_engine(python: Path) -> tuple[bool, str]:
    command = [str(python), "-c", "import sys, torch, bitsandbytes, diffusers, "
               "transformers, psd_tools, cv2, PIL, safetensors; "
               "print(sys.version.split()[0]); print(torch.__version__); "
               "print(torch.cuda.is_available())"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    lines = result.stdout.strip().splitlines()
    if result.returncode or len(lines) < 3:
        return False, (result.stderr or result.stdout).strip()[-500:]
    if not lines[0].startswith("3.12."):
        return False, f"See-through 建议 Python 3.12；当前为 {lines[0]}"
    if lines[-1] != "True":
        return False, "PyTorch 未检测到可用的 NVIDIA CUDA GPU"
    return True, f"Python {lines[0]} / PyTorch {lines[1]} / CUDA 可用"


def readiness(image_input: bool) -> list[str]:
    issues: list[str] = []
    if not psd2live_exe():
        issues.append("缺少 PSD2Live Windows 便携版；运行 python auto_setup.py --fix")
    if image_input:
        if not (SEE_REPO / "inference" / "scripts" / "inference_psd_quantized.py").is_file():
            issues.append("缺少 See-through 源码；运行 python auto_setup.py --fix")
        python = see_python()
        if not python:
            issues.append("缺少 See-through 的 Python 3.12 环境；运行 python auto_setup.py --fix")
        else:
            ok, detail = check_image_engine(python)
            if not ok:
                issues.append("自动拆层环境不可用：" + detail)
    return issues


def run_command(command: list[str], cwd: Path, log: Callable[[str], None],
                env: dict[str, str] | None = None,
                log_file: Path | None = None,
                cancel_event: threading.Event | None = None) -> None:
    if cancel_event and cancel_event.is_set():
        raise PipelineCancelled("转换已停止")
    encoding = "utf-8" if env and env.get("PYTHONIOENCODING", "").startswith("utf-8") \
        else locale.getpreferredencoding(False)
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding=encoding,
                                   errors="replace", bufsize=1,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as exc:
        raise PipelineError(f"启动失败：{exc}") from exc
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()
    def read_output() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        with log_file.open("a", encoding="utf-8") if log_file else open(os.devnull, "w") as stream:
            while True:
                if cancel_event and cancel_event.is_set():
                    raise PipelineCancelled("转换已停止")
                try:
                    line = lines.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                stream.write(line)
                stream.flush()
                clean = line.strip()
                if not clean or "Materializing param=" in clean or "it/s" in clean or "s/it" in clean \
                        or clean.startswith("Loading weights:") or clean == "\x1b[A":
                    continue
                log(clean)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        reader.join(timeout=10)
        process.stdout.close()
    code = process.wait()
    if code:
        raise PipelineError(f"上一步失败，退出代码 {code}")


def model_bundle(model_dir: Path) -> tuple[Path, list[Path]]:
    manifests = list(model_dir.rglob("*.model3.json"))
    if len(manifests) != 1:
        raise PipelineError(f"应生成一个 .model3.json，实际找到 {len(manifests)} 个")
    manifest = manifests[0]
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        refs = data["FileReferences"]
        moc = refs["Moc"]
        textures = refs["Textures"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PipelineError("model3.json 内容无效") from exc
    if not isinstance(moc, str) or not isinstance(textures, list) or not textures:
        raise PipelineError("模型缺少 MOC3 或贴图引用")

    names: list[str] = []
    def collect(value: object) -> None:
        if isinstance(value, str):
            if value.lower().endswith((".moc3", ".png", ".json")):
                names.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
    collect(refs)
    base = manifest.parent.resolve()
    files: list[Path] = []
    for name in names:
        path = (base / name.replace("/", os.sep)).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size == 0:
            raise PipelineError(f"模型引用的文件缺失或无效：{name}")
        files.append(path)
    if not (base / moc).read_bytes().startswith(b"MOC3"):
        raise PipelineError("导出的 .moc3 文件头无效")
    return manifest, files


def write_quality_report(source: Path, psd: Path, manifest: Path,
                         files: list[Path], output: Path,
                         layer_audit: dict | None = None,
                         core_validation: dict | None = None) -> list[str]:
    """Record objective export checks and likely issues for a human preview."""
    warnings: list[str] = []
    if layer_audit and layer_audit.get("foregroundMethod") == "complex_background_checks_skipped":
        warnings.append("背景较复杂，已跳过原图轮廓和多余部件过滤；请检查背景残留。")
    diagnostic = manifest.with_name(manifest.name.replace(".model3.json", ".psd2live.json"))
    summary: dict = {}
    tags: set[str] = set()
    if diagnostic.is_file():
        try:
            data = json.loads(diagnostic.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            tags = {str(layer.get("tag", "")).lower().replace("_", " ")
                    for layer in data.get("layers", []) if isinstance(layer, dict)}
        except (OSError, ValueError, TypeError):
            warnings.append("无法读取 PSD2Live 诊断文件，请在目标软件中检查模型。")
    else:
        warnings.append("没有 PSD2Live 诊断文件，请在目标软件中检查模型。")
    if tags:
        for label, candidates in (
            ("脸部", {"face"}),
            ("眼睛", {"irides", "eyes", "eyewhite"}),
            ("嘴巴", {"mouth", "mouth open", "mouth close"}),
        ):
            if not tags.intersection(candidates):
                warnings.append(f"未识别到{label}图层；相关参数效果可能较弱。")
        if "mouth open" not in tags:
            warnings.append("未发现独立张嘴素材；请特别检查张嘴效果。")
        if "eyewhite" not in tags or "eyelash" not in tags:
            warnings.append("眼白或睫毛未独立成层；请特别检查眨眼效果。")
    report = {
        "source": str(source), "layeredPsd": str(psd),
        "modelManifest": str(manifest),
        "referencedFilesVerified": len(files),
        "summary": summary, "warnings": warnings,
        "layerAudit": layer_audit,
        "validation": {"bundleFiles": "passed", "mocHeader": "passed",
                       "officialCore": core_validation or {"status": "not_tested"},
                       "targetSoftwarePlayback": "not_tested"},
        "nextStep": "在目标软件里预览眨眼、张嘴、转头、遮挡和动作。",
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return warnings


def verify_official_core(manifest: Path, output: Path,
                         log: Callable[[str], None]) -> dict:
    override = os.environ.get("LIVE2D_CORE_JS")
    core = Path(override) if override else ROOT / ".tools" / "live2dcubismcore.min.js"
    node = shutil.which("node")
    if not node or not core.is_file():
        return {"status": "not_tested", "reason": "official_core_or_node_missing"}
    result = output / "core-validation.json"
    run_command([node, str(ROOT / "verify_model.cjs"), str(manifest), str(core),
                 str(result)], ROOT, log, log_file=output / "conversion.log")
    return json.loads(result.read_text(encoding="utf-8"))


def convert(source: Path, output: Path, resolution: int = DEFAULT_RESOLUTION,
            log: Callable[[str], None] = print,
            no_group_offload: bool = True,
            cancel_event: threading.Event | None = None,
            canvas_size: tuple[int, int] | None = None) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"找不到输入文件：{source}")
    if source.suffix.lower() not in IMAGE_EXTS | {".psd"}:
        raise PipelineError("支持 PNG/JPG/WebP/BMP 图片或分层 PSD")
    if canvas_size is not None:
        if source.suffix.lower() == ".psd":
            raise PipelineError("自定义宽高仅用于图片转换；分层 PSD 使用已有画布。")
        canvas_size = tuple(normalize_resolution(value) for value in canvas_size)
        resolution = max(canvas_size)
        log(f"转换画布：{canvas_size[0]} × {canvas_size[1]}，保持角色比例并居中。")
    aligned_resolution = normalize_resolution(resolution)
    if aligned_resolution != resolution:
        log(f"拆层分辨率已从 {resolution} 对齐为 {aligned_resolution}（64 的倍数）。")
    resolution = aligned_resolution
    issues = readiness(source.suffix.lower() != ".psd")
    if issues:
        raise PipelineError("环境尚未就绪：\n- " + "\n- ".join(issues))
    output.mkdir(parents=True, exist_ok=True)
    log_file = output / "conversion.log"
    layer_audit = None
    model_dir = output / "model"
    if model_dir.exists() and any(model_dir.iterdir()):
        raise PipelineError(f"模型输出目录非空，请换一个输出目录：{model_dir}")
    if source.suffix.lower() == ".psd":
        psd = source
    else:
        stage = output / "stages"
        stage.mkdir(exist_ok=True)
        save_dir = stage / "layers"
        inference_source = source
        if canvas_size is not None:
            inference_source = prepare_canvas(source, stage / f"{source.stem}.canvas.png", canvas_size)
        psd = save_dir / f"{inference_source.stem}.psd"
        if psd.exists():
            raise PipelineError(f"拆层输出已存在，请换一个输出目录：{psd}")
        log(f"[1/3] 自动拆层与遮挡补全（{resolution} 分辨率）。")
        env = os.environ.copy()
        env["HF_HOME"] = str(ROOT / ".model-cache")
        env["TORCH_HOME"] = str(ROOT / ".model-cache" / "torch")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        command = [str(see_python()), "inference/scripts/inference_psd_quantized.py",
                   "--srcp", str(inference_source), "--save_dir", str(save_dir),
                   "--resolution", str(resolution), "--save_to_psd", "--tblr_split"]
        snapshots = [cached_model(repo) for repo in MODEL_REPOS]
        if all(snapshots):
            command += ["--repo_id_layerdiff", str(snapshots[0]),
                        "--repo_id_depth", str(snapshots[1])]
            env["HF_HUB_OFFLINE"] = "1"
            log("使用已下载的本地权重，无需联网。")
        else:
            log("首次运行会下载缺少的模型权重。")
        if no_group_offload:
            command.append("--no_group_offload")
        run_command(command, SEE_REPO, log, env, log_file, cancel_event)
        if not psd.is_file() or psd.stat().st_size == 0:
            raise PipelineError(f"拆层步骤没有生成 PSD：{psd}")
        log("[2/3] 检查多余部件，并尝试从原图恢复遗漏的嘴部。")
        cleaned = psd.with_name(source.stem + ".cleaned.psd")
        cleanup_command = [str(see_python()), str(ROOT / "layer_quality.py"),
                           "--source", str(inference_source), "--raw-psd", str(psd),
                           "--output", str(cleaned)]
        if canvas_size is not None:
            cleanup_command += ["--width", str(canvas_size[0]), "--height", str(canvas_size[1])]
        run_command(cleanup_command, ROOT, log, env, log_file, cancel_event)
        psd = cleaned
        layer_audit = json.loads(psd.with_suffix(".audit.json").read_text(encoding="utf-8"))
    log("[3/3] 自动绑模并导出 Live2D 模型。")
    run_command([str(psd2live_exe()), "--input", str(psd), "--output", str(model_dir),
                 "--lang", "en"], ROOT, log, log_file=log_file, cancel_event=cancel_event)
    manifest, files = model_bundle(model_dir)
    core_validation = verify_official_core(manifest, output, log)
    for warning in write_quality_report(source, psd, manifest, files, output, layer_audit, core_validation):
        log("检查提示：" + warning)
    log(f"完成：{manifest}（已检查 {len(files)} 个关联文件）")
    if source.suffix.lower() != ".psd":
        log(f"自动拆层 PSD：{psd}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="一张图或分层 PSD 自动导出可移植 Live2D 模型")
    parser.add_argument("input", nargs="?", type=Path, help="正面二次元角色图片或分层 PSD")
    parser.add_argument("--output", type=Path, help="输出文件夹")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="自定义拆层分辨率，默认 512，自动向上对齐到 64 的倍数")
    parser.add_argument("--width", type=int, help="图片转换画布宽度，与 --height 一起使用")
    parser.add_argument("--height", type=int, help="图片转换画布高度；设置宽高时拆层分辨率使用较长的一边")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument("--no-group-offload", action="store_true", dest="no_group_offload",
                        help="禁用分组卸载（默认，当前 8 GB GPU 已验证）")
    offload.add_argument("--group-offload", action="store_false", dest="no_group_offload",
                        help="启用上游分组卸载，部分量化环境存在兼容问题")
    parser.set_defaults(no_group_offload=True)
    parser.add_argument("--check", action="store_true", help="检查自动转换环境")
    args = parser.parse_args()
    if args.check:
        issues = readiness(True)
        print("自动转换环境已就绪" if not issues else "自动转换环境还需补齐：\n- " + "\n- ".join(issues))
        return 0 if not issues else 1
    if not args.input:
        parser.error("请提供角色图片或 PSD 路径")
    if (args.width is None) != (args.height is None):
        parser.error("请同时填写 --width 和 --height")
    canvas_size = (args.width, args.height) if args.width is not None else None
    output = args.output or ROOT / "outputs" / args.input.stem
    try:
        convert(args.input, output, args.resolution,
                no_group_offload=args.no_group_offload, canvas_size=canvas_size)
        return 0
    except PipelineError as exc:
        print("转换失败：" + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
