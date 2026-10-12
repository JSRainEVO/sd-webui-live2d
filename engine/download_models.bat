@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".auto_env\Scripts\python.exe" (
    echo 请把下载工具解压到已配置环境的插件 engine 文件夹，再双击运行。
    echo 若环境尚未配置，请先在 Live2D 页面点击“检测并补齐环境”。
    pause
    exit /b 1
)
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONIOENCODING=utf-8"
".auto_env\Scripts\python.exe" -u download_models.py --prompt
set "LIVE2D_DOWNLOAD_EXIT=%ERRORLEVEL%"
pause
exit /b %LIVE2D_DOWNLOAD_EXIT%
