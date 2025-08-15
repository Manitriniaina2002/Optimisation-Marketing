@echo off
echo Direct test script started > test_output.txt
echo Current directory: %CD% >> test_output.txt
date /t >> test_output.txt
time /t >> test_output.txt
echo Script completed successfully >> test_output.txt
type test_output.txt
pause
