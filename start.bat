@echo off
rem Windows: double-click to update (git pull) and start the app
cd /d "%~dp0"
where py > nul 2>&1
if %errorlevel%==0 (
  py -3 launcher.py
) else (
  python launcher.py
)
pause
