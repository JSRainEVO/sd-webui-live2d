"""Inspect or provision the automatic image-to-Live2D pipeline on Windows.

All downloads and installations are kept inside this project. First image
conversion additionally downloads upstream model weights into the local cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from auto_live2d import ROOT, SEE_REPO, PSD2LIVE_EXE, readiness


TOOLS = ROOT / ".tools"
AUTO_PYTHON = ROOT / ".auto_python"
AUTO_ENV = ROOT / ".auto_env"
UV_CACHE = ROOT / ".uv-cache"
RELEASE_API = "https://api.github.com/repos/tsunehimatoi/psd2live/releases/latest"
RELEASE_ASSET_SUFFIXES = (
    "windows-x64-portable.zip",
    "windows-x86_64-portable.zip",
    "windows-amd64-portable.zip",
    "windows-x64-portable-ffmpeg.zip",
    "windows-x86_64-portable-ffmpeg.zip",
    "windows-amd64-portable-ffmpeg.zip",
)
SEE_URL = "https://github.com/shitagaki-lab/see-through.git"


def run(command: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("运行：" + " ".join(command[:3]), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_uv() -> list[str]:
    found = shutil.which("uv")
    if found:
        return [found]
    TOOLS.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--target", str(TOOLS), "uv>=0.9,<1"])
    for candidate in (TOOLS / "Scripts" / "uv.exe", TOOLS / "uv.exe",
                      TOOLS / "bin" / "uv.exe"):
        if candidate.is_file():
            return [str(candidate)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TOOLS) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-m", "uv", "--version"], env=env,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("uv 安装后无法启动")
    return [sys.executable, "-m", "uv"]


def uv_env() -> dict[str, str]:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(UV_CACHE)
    env["UV_PYTHON_INSTALL_DIR"] = str(AUTO_PYTHON)
    env["UV_NO_PROGRESS"] = "1"
    if TOOLS.is_dir():
        env["PYTHONPATH"] = str(TOOLS) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def install_see_through() -> None:
    script = SEE_REPO / "inference" / "scripts" / "inference_psd_quantized.py"
    if not script.is_file():
        SEE_REPO.parent.mkdir(parents=True, exist_ok=True)
        if SEE_REPO.exists():
            raise RuntimeError("See-through 文件夹不完整，请先移走该文件夹")
        run(["git", "clone", "--depth", "1", SEE_URL, str(SEE_REPO)])
    # In the current upstream revision the quantized CPU-offload path uses the
    # text encoder's nominal device (CPU) for input IDs after Accelerate has
    # moved the quantized embedding to CUDA. Use Diffusers' execution device.
    source = SEE_REPO / "common" / "modules" / "layerdiffuse" / "diffusers_kdiffusion_sdxl.py"
    original = source.read_text(encoding="utf-8")
    old = "device = self.text_encoder.device"
    new = "device = self._execution_device"
    if old in original:
        original = original.replace(old, new, 1)
    elif new not in original:
        raise RuntimeError("See-through 版本发生变化，无法应用 8 GB GPU 修复")
    old_batch = """            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(body_tag_list)
            for ii in range(len(body_tag_list)):
                self._cached_prompt_embeds[body_tag_list[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(head_tag_list)
            for ii in range(len(head_tag_list)):
                self._cached_prompt_embeds[head_tag_list[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]"""
    small_batch = """            for tag in body_tag_list + head_tag_list:
                prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens([tag])
                self._cached_prompt_embeds[tag] = [prompt_embeds.cpu(), pooled_prompt_embeds.cpu()]
                del prompt_embeds, pooled_prompt_embeds
                torch.cuda.empty_cache()"""
    if old_batch in original:
        original = original.replace(old_batch, small_batch, 1)
    elif small_batch not in original:
        raise RuntimeError("See-through 版本发生变化，无法应用小显存批处理修复")
    source.write_text(original, encoding="utf-8")
    inference = SEE_REPO / "inference" / "scripts" / "inference_psd_quantized.py"
    inference_text = inference.read_text(encoding="utf-8")
    anchor = """        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            repo, trans_vae=trans_vae, unet=unet, scheduler=None)

        if args.cpu_offload:"""
    early_cache = """        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            repo, trans_vae=trans_vae, unet=unet, scheduler=None)

        # Cache tags before moving VAEs onto GPU to reduce peak memory.
        pipeline.cache_tag_embeds()

        if args.cpu_offload:"""
    tail = """        # Cache tag embeddings and unload text encoders to save VRAM
        pipeline.cache_tag_embeds()

    return pipeline"""
    if anchor in inference_text and tail in inference_text:
        inference_text = inference_text.replace(anchor, early_cache, 1)
        inference_text = inference_text.replace(tail, "    return pipeline", 1)
        inference.write_text(inference_text, encoding="utf-8")
    elif "pipeline.cache_tag_embeds()\n\n        if args.cpu_offload:" not in inference_text:
        raise RuntimeError("See-through 版本发生变化，无法应用显存优化修复")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("没有检测到 NVIDIA GPU；自动拆层需要 CUDA GPU")
    uv = ensure_uv()
    env = uv_env()
    python = AUTO_ENV / "Scripts" / "python.exe"
    if not python.is_file():
        run(uv + ["python", "install", "3.12"], env=env)
        run(uv + ["venv", "--python", "3.12", str(AUTO_ENV)], env=env)
    probe = subprocess.run([str(python), "-c",
                            "import torch,bitsandbytes,psd_tools,diffusers; "
                            "assert torch.cuda.is_available()"],
                           capture_output=True, env=env, check=False)
    if probe.returncode == 0:
        print("自动拆层依赖已就绪")
        return
    print("安装 CUDA PyTorch 与 See-through 依赖；这一步需要下载数 GB 文件。")
    run(uv + ["pip", "install", "--python", str(python),
              "torch==2.8.0+cu128", "torchvision==0.23.0+cu128",
              "torchaudio==2.8.0+cu128", "--index-url",
              "https://download.pytorch.org/whl/cu128"], env=env)
    run(uv + ["pip", "install", "--python", str(python), "-r",
              str(SEE_REPO / "requirements.txt"), "-r",
              str(SEE_REPO / "requirements-inference-bnb.txt")],
        cwd=SEE_REPO, env=env)
    run([str(python), "-c", "import torch,bitsandbytes,psd_tools,diffusers; "
         "assert torch.cuda.is_available()"], env=env)


def select_psd2live_asset(release: dict) -> dict:
    """Accept both upstream x64 naming schemes and prefer the smaller build."""
    assets = release.get("assets", [])
    for suffix in RELEASE_ASSET_SUFFIXES:
        matches = [asset for asset in assets
                   if str(asset.get("name", "")).casefold().endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = "、".join(str(asset.get("name", "")) for asset in matches)
            raise RuntimeError(f"PSD2Live 便携版存在多个同名架构候选：{names}")
    version = release.get("tag_name", "未知版本")
    names = "、".join(str(asset.get("name", "")) for asset in assets) or "暂无附件"
    raise RuntimeError(
        f"无法定位 PSD2Live Windows x64 便携版（{version}）。发布附件：{names}。"
        "需要 Windows x64 / x86_64 / amd64 portable ZIP；若版本刚发布，请稍后重试。")


def install_psd2live() -> None:
    if PSD2LIVE_EXE.is_file():
        print("PSD2Live 便携版已就绪")
        return
    request = urllib.request.Request(RELEASE_API, headers={"User-Agent": "Live2D-Auto-Pipeline"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = select_psd2live_asset(release)
    archive = ROOT / "third_party" / "psd2live-portable.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 PSD2Live {release['tag_name']}：{asset['name']}（约 {asset['size'] // 1_000_000} MB）")
    request = urllib.request.Request(asset["browser_download_url"],
                                     headers={"User-Agent": "Live2D-Auto-Pipeline"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as dest:
        shutil.copyfileobj(response, dest)
    if archive.stat().st_size != asset["size"]:
        raise RuntimeError("PSD2Live 下载大小不符")
    digest = asset.get("digest", "")
    if digest.startswith("sha256:"):
        with archive.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
            actual = checksum.hexdigest()
        if actual.lower() != digest.partition(":")[2].lower():
            raise RuntimeError("PSD2Live 下载校验失败")
    target = PSD2LIVE_EXE.parent
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise RuntimeError("压缩包内含越界路径")
        bundle.extractall(target)
    if not PSD2LIVE_EXE.is_file():
        raise RuntimeError("PSD2Live 解压后未找到主程序")
    print("PSD2Live 便携版已就绪")


def main() -> int:
    parser = argparse.ArgumentParser(description="检测并补齐图片自动转 Live2D 所需环境")
    parser.add_argument("--fix", action="store_true", help="下载缺失程序和依赖")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("当前自动安装脚本支持 Windows")
        return 1
    if args.fix:
        try:
            install_psd2live()
            install_see_through()
        except (OSError, RuntimeError, subprocess.CalledProcessError,
                urllib.error.URLError, zipfile.BadZipFile) as exc:
            print("安装中断：" + str(exc), file=sys.stderr)
            return 1
    issues = readiness(True)
    if issues:
        print("尚需补齐：\n- " + "\n- ".join(issues))
        return 1
    print("自动转换环境已就绪。首次转换图片时会继续下载 AI 模型权重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
