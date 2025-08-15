@echo off
setlocal enabledelayedexpansion

echo === RUNNING M3 CUSTOMER SEGMENTATION SCRIPT ===
echo.

echo 1. Verifying Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not found in PATH.
    echo Please install Python and add it to your system's PATH.
    pause
    exit /b 1
)

python --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not working correctly.
    pause
    exit /b 1
)

echo.
echo 2. Checking for required Python packages...
python -c "import sys, pkg_resources; required = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'}; installed = {pkg.key for pkg in pkg_resources.working_set}; missing = required - installed; print(f'Missing packages: {missing}' if missing else 'All required packages are installed')"

if %ERRORLEVEL% NEQ 0 (
    echo Error checking Python packages.
    pause
    exit /b 1
)

echo.
echo 3. Running M3 Customer Segmentation script...
echo =========================================
echo.

set PYTHONUNBUFFERED=1
python -u m3_customer_segmentation.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: The M3 script failed with error code %ERRORLEVEL%
) else (
    echo.
    echo M3 script completed successfully.
)

echo.
echo 4. Checking for output files...
echo ===========================
echo.

if exist output\ (
    echo Output directory exists. Contents:
    dir /b output
) else (
    echo Output directory does not exist. No files were generated.
)

echo.
pause
