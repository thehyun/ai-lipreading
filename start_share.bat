@echo off
chcp 65001 >nul
REM Double-click to start the website with a public link (https://xxxx.gradio.live, 72 hours)
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Double-click setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -X utf8 web_app.py --share %*
pause
