@echo off
REM TalkDB - Natural Language Database Interface
REM Executable batch file that reads environment variables

title TalkDB Server

echo.
echo ============================================
echo  TalkDB - Natural Language Database Interface
echo ============================================
echo.

REM Set the Python executable path
set PYTHON_EXE=C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe

REM Check if Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    echo Please update the PYTHON_EXE path in TalkDB.bat
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please run this batch file from the TalkDB directory
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Using system environment variables only
    echo.
)

echo Starting TalkDB Server...
echo Press Ctrl+C to stop the server
echo.

REM Start the server using main.py
"%PYTHON_EXE%" main.py

REM If we reach here, the server stopped
echo.
echo TalkDB Server stopped.
pause