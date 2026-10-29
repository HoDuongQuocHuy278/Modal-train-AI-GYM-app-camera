@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo     VisionFit AI Studio - Khoi dong Web App
echo ===================================================

set PYTHON_EXE=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
)

echo Dang khoi dong Web Server bang %PYTHON_EXE%...
%PYTHON_EXE% web_server.py --port 8000

endlocal
