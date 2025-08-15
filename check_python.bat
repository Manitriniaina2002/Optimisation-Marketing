@echo off
echo === PYTHON CHECK ===
echo.

echo 1. Checking for Python in PATH...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Python is in PATH
    python --version
) else (
    echo   Python is NOT in PATH
)

echo.
echo 2. Checking common Python installation directories...

echo   - Checking C:\Python*...
if exist "C:\Python*" (
    dir /b C:\Python*
) else (
    echo     No Python found in C:\
)

echo.
echo   - Checking %APPDATA%\Python...
if exist "%APPDATA%\Python" (
    dir /b "%APPDATA%\Python"
) else (
    echo     No Python found in %%APPDATA%%\Python
)

echo.
echo   - Checking %LOCALAPPDATA%\Programs\Python...
if exist "%LOCALAPPDATA%\Programs\Python" (
    dir /b "%LOCALAPPDATA%\Programs\Python"
) else (
    echo     No Python found in %%LOCALAPPDATA%%\Programs\Python
)

echo.
echo 3. Environment Variables:
echo   PYTHONPATH=%PYTHONPATH%
echo   PATH=%PATH%

echo.
echo 4. Trying to run a Python command...
python -c "import sys; print(f'Python {sys.version}')"

echo.
echo === CHECK COMPLETE ===
pause
