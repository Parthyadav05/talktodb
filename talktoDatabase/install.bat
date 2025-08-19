@echo off
REM TalkDB Installation Script

echo Installing TalkDB Dependencies...
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ===============================================
echo  TalkDB Installation Complete!
echo ===============================================
echo.
echo 1. Edit .env file with your database credentials
echo 2. Run TalkDB.bat to start the server
echo 3. Access at http://localhost:8080
echo.
pause
