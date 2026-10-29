@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup_windows.bat first. & exit /b 1)
.venv\Scripts\python.exe visionfit.py predict --model models/action_demo.pt --input data/example.npy --threshold 0
if errorlevel 1 exit /b 1
echo.
echo DEMO ONLY: synthetic action checkpoint, NOT human recognition.
endlocal
