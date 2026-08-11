@echo off
chcp 65001 >nul
title Ready Player One - Auto Hunt
echo ============================================
echo   Ready Player One - Auto Hunt Launcher
echo   Requesting administrator privileges...
echo ============================================
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Need admin rights, UAC prompt will appear.
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d %~dp0 && python main.py --process Maplestory_Classic.exe' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"
python main.py --process Maplestory_Classic.exe

echo.
echo ============================================
echo   Process exited.
echo ============================================
pause
