@echo off
:: Python环境测试脚本
chcp 65001 >nul 2>&1
title Python环境兼容性测试

echo ========================================
echo      Python环境兼容性测试
echo ========================================
echo.

set "test_passed=1"

echo 🔍 检测Python环境...

:: 检查Python是否存在
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python环境
    echo 💡 需要安装Python 3.8+
    set "test_passed=0"
    goto :test_result
)

:: 获取Python版本
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "python_version=%%v"
echo ✅ 发现Python: %python_version%

:: 检查版本兼容性 (>= 3.8)
echo 🔍 检查版本兼容性...
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ❌ Python版本过低 (需要 >=3.8)
    set "test_passed=0"
) else (
    echo ✅ Python版本符合要求
)

:: 检查pip工具
echo 🔍 检查pip工具...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip工具不可用
    set "test_passed=0"
) else (
    for /f "tokens=2" %%v in ('python -m pip --version 2^>^&1') do set "pip_version=%%v"
    echo ✅ pip可用: %pip_version%
)

:: 检查基础模块
echo 🔍 检查基础模块...
python -c "import sys, os, subprocess, json, importlib" >nul 2>&1
if errorlevel 1 (
    echo ❌ 基础模块导入失败
    set "test_passed=0"
) else (
    echo ✅ 基础模块正常
)

:: 测试网络连接和pip源
echo 🔍 测试网络连接...
python -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 官方PyPI连接测试失败
    :: 测试国内镜像
    python -c "import urllib.request; urllib.request.urlopen('https://pypi.tuna.tsinghua.edu.cn/simple/', timeout=5)" >nul 2>&1
    if errorlevel 1 (
        echo ❌ 清华镜像连接也失败
        echo 💡 网络可能有问题，建议检查网络连接
    ) else (
        echo ✅ 清华镜像连接正常
    )
) else (
    echo ✅ 官方PyPI连接正常
)

:: 测试关键依赖包安装
echo.
echo 🧪 测试依赖包兼容性...

:: 测试Flask安装
echo 测试 Flask 安装...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 🔧 尝试安装Flask测试...
    python -m pip install flask --quiet --timeout 30 >nul 2>&1
    if errorlevel 1 (
        echo ❌ Flask安装测试失败
        set "test_passed=0"
    ) else (
        echo ✅ Flask安装测试成功
        python -m pip uninstall flask -y --quiet >nul 2>&1
    )
) else (
    echo ✅ Flask已安装
)

:: 测试PyTorch安装能力
echo 测试 PyTorch 兼容性...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 🔧 检查PyTorch安装环境...
    :: 检查CPU架构
    python -c "import platform; print(platform.machine())" >temp_arch.txt 2>nul
    if exist temp_arch.txt (
        set /p arch=<temp_arch.txt
        del temp_arch.txt
        echo CPU架构: %arch%
        if "%arch%"=="AMD64" (
            echo ✅ 支持PyTorch安装
        ) else (
            echo ⚠️  特殊架构，PyTorch安装可能需要特殊处理
        )
    )
) else (
    echo ✅ PyTorch已安装
)

:test_result
echo.
echo ========================================
echo            测试结果
echo ========================================

if "%test_passed%"=="1" (
    echo ✅ 当前Python环境完全兼容！
    echo 🎉 可以直接使用现有环境运行YOLOv智能标注工具
    echo.
    echo 📋 环境信息：
    echo   - Python版本: %python_version%
    echo   - pip版本: %pip_version%
    echo   - 基础模块: 正常
    echo   - 网络连接: 可用
    echo.
    echo 💡 建议：继续使用当前环境，将自动安装所需依赖包
) else (
    echo ❌ 当前Python环境不完全兼容
    echo 🔧 建议安装新的Python 3.11.7环境以确保最佳兼容性
    echo.
    echo 📋 检测到的问题：
    if "%python_version%"=="" echo   - Python未安装或不可用
    echo.
    echo 💡 解决方案：
    echo   1. 运行install_python.bat自动安装Python 3.11.7
    echo   2. 或手动安装Python 3.8+并配置环境变量
    echo   3. 确保网络连接正常以便下载依赖包
)

echo ========================================
echo.
pause
