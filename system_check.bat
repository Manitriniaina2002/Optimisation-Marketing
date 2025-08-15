@echo off
echo === SYSTEM CHECK ===
echo.

echo 1. Basic Command Test:
echo Hello, World! > test_output.txt
type test_output.txt
del test_output.txt
echo.

echo 2. Directory Listing:
dir /b
echo.

echo 3. Environment Variables:
echo PATH=%PATH%
echo.

echo 4. Python Check:
where python
echo.

if exist "C:\Python*" (
    echo Python found in C:\
    dir /b C:\Python*
) else (
    echo No Python found in C:\
)
echo.

echo 5. Current Directory:
cd
echo.

echo === CHECK COMPLETE ===
pause
