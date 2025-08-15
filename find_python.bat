@echo off
setlocal enabledelayedexpansion

echo Searching for Python installations...
echo ================================
echo.

set "found=0"

REM Check common Python installation directories
for %%d in (
    "C:\Python*"
    "%LOCALAPPDATA%\Programs\Python\Python*"
    "%ProgramFiles%\Python*"
    "%ProgramFiles(x86)%\Python*"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python*"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
) do (
    if exist "%%~d" (
        set /a found+=1
        echo [!] Found Python installation in: %%~d
        dir /b "%%~d"
        echo.
    )
)

if "!found!"=="0" (
    echo No Python installations found in common locations.
) else (
    echo Total Python installations found: !found!
)

echo.
echo Checking PATH environment variable for Python...
echo =========================================
echo.

echo %PATH% | find /i "python" >nul
if errorlevel 1 (
    echo No Python found in PATH.
) else (
    echo Python found in PATH.
    where python
)

echo.
pause
