@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)
if errorlevel 1 (echo Cannot create Python environment. & exit /b 1)
.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
if errorlevel 1 (echo Dependency install failed. Check Python and PyTorch compatibility. & exit /b 1)
echo Setup completed. See README.md for REAL data and training.
endlocal
