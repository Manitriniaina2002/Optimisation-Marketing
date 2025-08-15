@echo off
echo === ENVIRONMENT TEST ===
echo.

echo 1. Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Python is in PATH
    python --version
) else (
    echo   Python is NOT in PATH
)

echo.
echo 2. Running a simple Python command...
python -c "print('Hello from Python!')"

echo.
echo 3. Running the minimal test script...
python minimal_test.py

echo.
echo 4. Checking for Python in common locations...
if exist "C:\Python*" (
    echo   Found Python installation in C:\
    dir /b C:\Python*
) else (
    echo   No Python found in C:\
)

if exist "%LOCALAPPDATA%\Programs\Python" (
    echo   Found Python installation in %%LOCALAPPDATA%%\Programs\Python
    dir /b "%LOCALAPPDATA%\Programs\Python"
) else (
    echo   No Python found in %%LOCALAPPDATA%%\Programs\Python
)

echo.
echo 5. Current directory contents:
dir /b

echo.
echo === TEST COMPLETE ===
pause
