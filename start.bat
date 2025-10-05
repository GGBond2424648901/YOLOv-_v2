@echo off
title YOLOv AI Annotation Tool

echo ========================================
echo    YOLOv AI Annotation Tool - Windows
echo ========================================
echo.

rem Smart Python environment detection
echo Detecting Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Warning: Python not found, installing automatically...
    goto install_python
) else (
    rem Found Python environment, perform detailed detection
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "detected_version=%%v"
    echo Success: Found Python environment %detected_version%
    
    rem Check version compatibility
    python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Error: Python version too low (need >=3.8^), installing new version
        goto install_python
    )
    
    rem Check pip availability
    echo Checking pip tool...
    python -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo Error: pip tool not available, installing new version
        goto install_python
    )
    
    rem Test key modules
    echo Testing Python environment integrity...
    python -c "import sys, os, subprocess, json, importlib" >nul 2>&1
    if errorlevel 1 (
        echo Error: Python environment incomplete, installing new version
        goto install_python
    )
    
    echo Success: Current Python environment available, continue using existing environment
    goto check_python_done
)

:install_python
echo Starting Python auto-installer...

rem Check if install script exists
if not exist "install_python.bat" (
    echo Error: Python install script not found
    echo Tip: Please ensure install_python.bat file exists
    pause
    exit /b 1
)

rem Execute Python installation
call install_python.bat
if errorlevel 1 (
    echo Error: Python installation failed
    pause
    exit /b 1
)

echo Success: Python installation completed, continue starting program...

:check_python_done
echo Success: Python environment ready

rem Check dependencies
echo.
echo Checking dependencies...

rem Check each dependency package
set "missing_deps=0"

echo Checking Flask...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo   Flask: Not installed
    set "missing_deps=1"
) else (
    echo   Flask: Installed
)

echo Checking PyTorch...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo   PyTorch: Not installed
    set "missing_deps=1"
) else (
    echo   PyTorch: Installed
)

echo Checking OpenCV...
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo   OpenCV: Not installed
    set "missing_deps=1"
) else (
    echo   OpenCV: Installed
)

echo Checking other dependencies...
python -c "import PIL, numpy, ultralytics, flask_cors" >nul 2>&1
if errorlevel 1 (
    echo   Other dependencies: Missing
    set "missing_deps=1"
) else (
    echo   Other dependencies: Installed
)

if "%missing_deps%"=="1" (
    echo.
    echo Installing missing dependencies automatically...
    echo Info: This may take several minutes, please wait...
    echo.
    
    rem First try to upgrade pip
    echo Upgrading pip to latest version...
    python -m pip install --upgrade pip
    
    rem Try to install dependencies using domestic mirror
    echo Installing dependencies using Tsinghua mirror...
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Warning: Mirror installation failed, trying official source...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo.
            echo Error: Dependencies installation failed
            echo Suggestions:
            echo    1. Check network connection
            echo    2. Run this script as administrator
            echo    3. Manual execute: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
            echo    4. If still fails, install dependencies one by one
            echo.
            echo Dependency List:
            echo    - Flask==2.3.3
            echo    - torch (latest version)
            echo    - opencv-python==4.8.1.78
            echo    - ultralytics (latest version)
            echo    - Other packages see requirements.txt
            pause
            exit /b 1
        )
    )
    
    echo.
    echo Success: Dependencies installed successfully!
    echo Verify: Re-verifying dependencies...
    echo.
    
    rem Re-check key dependencies
    set "verification_failed=0"
    
    echo Verifying Flask...
    python -c "import flask; print('Flask version:', flask.__version__)" >nul 2>&1
    if errorlevel 1 (
        echo   Failed: Flask verification failed
        set "verification_failed=1"
    ) else (
        echo   Passed: Flask verification passed
    )
    
    echo Verifying PyTorch...
    python -c "import torch; print('PyTorch version:', torch.__version__)" >nul 2>&1
    if errorlevel 1 (
        echo   Failed: PyTorch verification failed
        set "verification_failed=1"
    ) else (
        echo   Passed: PyTorch verification passed
    )
    
    echo Verifying OpenCV...
    python -c "import cv2; print('OpenCV version:', cv2.__version__)" >nul 2>&1
    if errorlevel 1 (
        echo   Failed: OpenCV verification failed
        set "verification_failed=1"
    ) else (
        echo   Passed: OpenCV verification passed
    )
    
    echo Verifying Ultralytics...
    python -c "import ultralytics; print('Ultralytics installed successfully')" >nul 2>&1
    if errorlevel 1 (
        echo   Failed: Ultralytics verification failed
        set "verification_failed=1"
    ) else (
        echo   Passed: Ultralytics verification passed
    )
    
    if "%verification_failed%"=="1" (
        echo.
        echo Warning: Some dependencies may not be installed completely
        echo Suggestion: Recommend re-running this script or manually install failed packages
        echo.
    ) else (
        echo.
        echo Success: All dependencies verified successfully!
        echo.
    )
) else (
    echo.
    echo Success: All dependencies ready
    echo.
)

rem Start server
echo Starting YOLOv AI Annotation Tool...
echo Info: Browser will open automatically, if not please visit http://localhost:5000
echo Tip: Press Ctrl+C to stop server
echo ========================================
echo.

python start_server.py

pause