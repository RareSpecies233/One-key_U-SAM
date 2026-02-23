@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_windows.ps1"
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo 训练脚本异常退出，退出码: %EXIT_CODE%
)

echo.
pause
exit /b %EXIT_CODE%
