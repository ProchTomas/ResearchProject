@echo off
REM Portfolio Optimization - Simple Startup

title Portfolio Optimization

echo.
echo ================================================
echo Portfolio Optimization System
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install Python from python.org
    pause
    exit /b 1
)

echo Python found
echo.

REM Install packages
echo Installing required packages...
python -m pip install flask flask-cors numpy yfinance pandas scipy --quiet

if not exist "saved_statistics" mkdir saved_statistics

echo.
echo ================================================
echo Starting server...
echo ================================================
echo.
echo Open your browser to: http://localhost:5000
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.

python server.py

pause