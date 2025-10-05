@echo off
rem Python自动下载安装脚本
chcp 936 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
title Python环境自动安装

echo ========================================
echo     Python 3.11.7 自动安装程序
echo ========================================
echo.

rem 检查Python是否已安装
python --version >nul 2>&1
if not errorlevel 1 (
    echo [发现] 已安装的Python环境
    echo [检测] 检查版本兼容性...
    
    :: 显示当前版本
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "current_version=%%v"
    echo 当前Python版本: %current_version%
    
    :: 检查版本是否符合要求 (>=3.8)
    python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo [成功] Python版本符合要求 (>=3.8)
        echo [测试] 测试当前环境的可用性...
        
        :: 测试pip是否可用
        python -m pip --version >nul 2>&1
        if not errorlevel 1 (
            echo [成功] pip工具可用
            
            :: 测试基础模块导入
            python -c "import sys, os, subprocess, json" >nul 2>&1
            if not errorlevel 1 (
                echo [成功] 基础模块正常
                echo [完成] 当前Python环境可用，将使用现有环境
                goto :end
            ) else (
                echo [错误] 基础模块导入失败
                echo [提示] 可能是Python安装不完整，将安装新的Python环境
            )
        ) else (
            echo [错误] pip工具不可用
            echo [提示] 这可能导致依赖包安装失败，将安装新的Python环境
        )
    ) else (
        echo [错误] Python版本过低 (需要 >=3.8)
        echo [提示] 将安装新版本以确保兼容性
    )
    echo.
    echo [警告] 检测到问题，准备安装Python 3.11.7以确保最佳兼容性...
    echo [提示] 您的原有Python环境不会受到影响
    timeout /t 3 /nobreak >nul
)

echo [检测] 检测系统架构...
set "ARCH=x86"
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" set "ARCH=amd64"
if "%PROCESSOR_ARCHITEW6432%"=="AMD64" set "ARCH=amd64"

echo 系统架构: %ARCH%

:: 设置Python下载链接（Python 3.11.7）
if "%ARCH%"=="amd64" (
    set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    set "PYTHON_FILE=python-3.11.7-amd64.exe"
) else (
    set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/python-3.11.7.exe"
    set "PYTHON_FILE=python-3.11.7.exe"
)

echo [下载] 下载Python安装包...
echo 下载地址: %PYTHON_URL%
echo 这可能需要几分钟时间，请耐心等待...
echo.

:: 使用PowerShell下载Python
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest '%PYTHON_URL%' -OutFile '%PYTHON_FILE%' -UseBasicParsing; Write-Host '[成功] 下载完成'; exit 0 } catch { Write-Host '[错误] 下载失败:' $_.Exception.Message; exit 1 }}"

if errorlevel 1 (
    echo.
    echo [错误] Python下载失败
    echo [解决方案]
    echo    1. 检查网络连接
    echo    2. 手动下载Python: %PYTHON_URL%
    echo    3. 运行下载的安装包，勾选"Add Python to PATH"
    pause
    exit /b 1
)

echo [安装] 开始安装Python...
echo [警告] 请在安装过程中勾选 "Add Python to PATH" 选项
echo.

:: 静默安装Python，自动添加到PATH
"%PYTHON_FILE%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

if errorlevel 1 (
    echo [错误] Python自动安装失败，正在启动手动安装...
    echo [提示] 请确保在安装时勾选 "Add Python to PATH"
    "%PYTHON_FILE%"
    pause
) else (
    echo [成功] Python安装完成
)

:: 刷新环境变量
call refreshenv >nul 2>&1

:: 等待安装完成
echo 等待安装完成...
timeout /t 3 /nobreak >nul

:: 验证安装
echo [验证] 验证Python安装...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python安装验证失败
    echo [提示] 请重启命令行或重新启动电脑后再试
    pause
    exit /b 1
) else (
    echo [成功] Python安装验证成功
    python --version
)

:: 清理安装文件
if exist "%PYTHON_FILE%" (
    echo [清理] 清理安装文件...
    del "%PYTHON_FILE%"
)

:end
echo.
echo ========================================
echo     Python环境准备完成
echo ========================================
exit /b 0
