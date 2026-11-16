@echo off
REM Double-click to start the lip-reading website (uses .venv in this folder)
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Double-click setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -X utf8 web_app.py %*
pause
