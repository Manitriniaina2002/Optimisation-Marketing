@echo off
echo === BASIC SYSTEM TEST ===
echo.

echo 1. System Information:
systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type"
echo.

echo 2. Current Directory:
cd
echo.

echo 3. Directory Contents:
dir /b
echo.

echo 4. Environment Variables:
echo PATH=%PATH%
echo.

echo 5. Checking for Python:
where python
echo.

echo 6. Simple Command Test:
echo Hello, World! > test_output.txt
type test_output.txt
del test_output.txt
echo.

echo === TEST COMPLETE ===
pause
