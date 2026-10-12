"""Download both complete model repositories without loading inference models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from auto_live2d import MODEL_REPOS, cached_model


def main() -> int:
    parser = argparse.ArgumentParser(description="整库下载 Live2D 转换模型，不运行图片转换")
    parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parent,
                        help="插件页面中的转换环境目录")
    parser.add_argument("--proxy", help="可选 HTTP 代理地址")
    parser.add_argument("--prompt", action="store_true", help="下载前询问代理设置")
    args = parser.parse_args()
    root = args.engine.expanduser().resolve()
    if not (root / "auto_live2d.py").is_file():
        print("请把下载工具放到插件的 engine 文件夹，或用 --engine 指定转换环境目录。")
        return 1

    proxy = args.proxy
    if args.prompt and proxy is None:
        print("如果浏览器通过代理访问 Hugging Face，请填写相同代理的 HTTP 地址。")
        print("格式：http://127.0.0.1:端口；无需代理时直接按回车。")
        proxy = input("代理地址：").strip()
    if proxy:
        if not proxy.startswith(("http://", "https://")):
            print("请填写完整的 HTTP 代理地址，例如 http://127.0.0.1:端口。")
            return 1
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        # A stale lowercase setting can otherwise override the selected proxy.
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # Hugging Face reads environment settings during import.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少下载依赖，请先在 Live2D 页面完成“检测并补齐环境”。")
        return 1

    cache = root / ".model-cache" / "hub"
    print(f"保存位置：{cache}", flush=True)
    print("两套模型合计约 5.7 GB。此工具只下载文件，不加载模型、不占用显存。", flush=True)
    for index, repo in enumerate(MODEL_REPOS, 1):
        if cached_model(repo, cache) is not None:
            print(f"[{index}/2] 已有完整模型，跳过下载：{repo}", flush=True)
            continue
        folder = cache / ("models--" + repo.replace("/", "--"))
        snapshot = folder / "snapshots" / "manual"
        print(f"[{index}/2] 下载完整模型：{repo}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(snapshot), max_workers=1)
        # Activate the local snapshot only after the entire download succeeds.
        refs = folder / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        temporary = refs / "main.download.tmp"
        temporary.write_text("manual\n", encoding="utf-8")
        temporary.replace(refs / "main")
        if cached_model(repo, cache) is None:
            raise RuntimeError(f"文件已下载，但模型目录尚不符合插件识别要求：{snapshot}")
        print(f"[{index}/2] 下载完成：{repo}", flush=True)
    print("两套模型已就绪。返回 Forge 的 Live2D 页面即可转换，无需手动建目录或 main 文件。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("下载已停止，已完成的文件保留；再次运行可继续下载。")
        raise SystemExit(130)
    except Exception as error:
        print(f"下载中断：{error}", file=sys.stderr)
        print("已完成的文件保留。请检查网络和代理设置后再次运行；浏览器可访问不代表下载程序能直接访问。",
              file=sys.stderr)
        raise SystemExit(1)
