@echo off
echo ========================================
echo  Sales Forecasting Application
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Make sure Python is installed and in your PATH
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
if not exist "venv\Lib\site-packages\fastapi\" (
    echo.
    echo Installing dependencies (this may take a few minutes)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the application
echo.
echo ========================================
echo Starting application...
echo ========================================
echo.
echo Application will be available at:
echo http://localhost:8002
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

uvicorn app:app --host 0.0.0.0 --port 8002 --reload

pause
