@echo off
chcp 65001 >nul
cd /d %~dp0
echo.
echo ===== AI Lip Reading - first-time setup =====
echo.
echo [1/4] Checking Python 3.10...
py -3.10 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.10 not found. Installing it with winget...
  winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
  echo.
  echo Python 3.10 installed. Close this window and double-click setup.bat again.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo [2/4] Creating virtual environment .venv ...
  py -3.10 -m venv .venv || goto :fail
) else (
  echo [2/4] Virtual environment already exists.
)
echo [3/4] Installing packages - takes 5-10 minutes...
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
echo [4/4] Downloading pretrained models - about 1.1GB...
.venv\Scripts\python.exe -X utf8 setup_models.py || goto :fail
echo.
echo ===== Done! =====
echo Double-click start_web.bat (only you) or start_share.bat (public link).
pause
exit /b 0
:fail
echo.
echo Setup stopped. Read the message above, fix it, and run setup.bat again.
pause
exit /b 1
