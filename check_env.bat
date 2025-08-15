@echo off
setlocal enabledelayedexpansion

echo === ENVIRONMENT CHECK === > env_check.txt
echo Timestamp: %DATE% %TIME% >> env_check.txt
echo.

echo 1. System Information: >> env_check.txt
ver >> env_check.txt
echo.

echo 2. Current Directory: >> env_check.txt
cd >> env_check.txt
dir /b >> env_check.txt
echo.

echo 3. Environment Variables: >> env_check.txt
set >> env_check.txt
echo.

echo 4. Python Check: >> env_check.txt
where python >> env_check.txt 2>&1
python --version >> env_check.txt 2>&1
echo.

echo 5. File System Test: >> env_check.txt
echo This is a test file. > test_file.txt
echo Test file created: >> env_check.txt
type test_file.txt >> env_check.txt
del test_file.txt
echo.

echo 6. Directory Listing: >> env_check.txt
dir /b >> env_check.txt
echo.

echo Environment check completed. Results saved to env_check.txt

:: Display the first few lines of the output
type env_check.txt | more
