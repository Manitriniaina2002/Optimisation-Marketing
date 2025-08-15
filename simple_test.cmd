@echo off
echo Simple Test Script
echo =================
echo.
echo 1. Current directory: %CD%
echo 2. Current date: %DATE%
echo 3. Current time: %TIME%
echo 4. Command line arguments: %*
echo.
echo 5. Creating test file...
echo This is a test file. > test_file.txt
echo 6. Test file created. Contents:
type test_file.txt
echo.
echo 7. Deleting test file...
del test_file.txt
echo 8. Test file deleted.
echo.
echo 9. Directory listing:
dir /b
echo.
echo Test completed successfully.
pause
