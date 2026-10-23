@echo off
rem PTX Downgrader GUI launcher (Windows, double-click to start)
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [!] Python not found. Install Python 3.8+ from https://python.org
  echo     Remember to check "Add python.exe to PATH" during install.
  pause
  exit /b 1
)
python -m downgrader --gui
if errorlevel 1 pause
