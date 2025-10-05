#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv智能标注工具 - 安装测试脚本
检查所有依赖包和功能是否正常工作
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("   ❌ 版本太低，需要Python 3.8+")
        return False
    else:
        print("   ✅ Python版本符合要求")
        return True

def test_imports():
    """测试所有依赖包导入"""
    print("\n📦 测试依赖包导入...")
    
    packages = [
        ('flask', 'Flask'),
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'), 
        ('ultralytics', 'Ultralytics'),
        ('flask_cors', 'Flask-CORS'),
        ('werkzeug', 'Werkzeug'),
        ('requests', 'Requests')
    ]
    
    failed_packages = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name} - {e}")
            failed_packages.append(name)
    
    if failed_packages:
        print(f"\n⚠️  发现 {len(failed_packages)} 个缺失的依赖包:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        return False
    else:
        print("\n✅ 所有依赖包导入成功")
        return True

def check_files():
    """检查必要文件是否存在"""
    print("\n📁 检查必要文件...")
    
    required_files = [
        'yolo_annotation_server.py',
        'yolo_annotation_tool.html',
        'requirements.txt',
        'start_server.py',
        'start.bat'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  发现 {len(missing_files)} 个缺失的文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("\n✅ 所有必要文件存在")
        return True

def check_directories():
    """检查并创建必要目录"""
    print("\n📂 检查必要目录...")
    
    directories = ['uploads', 'models', 'results']
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"   ✅ {directory}/ (已存在)")
        else:
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ {directory}/ (已创建)")
            except Exception as e:
                print(f"   ❌ {directory}/ - 创建失败: {e}")
                return False
    
    print("\n✅ 所有目录检查完成")
    return True

def test_torch_functionality():
    """测试PyTorch基本功能"""
    print("\n🧠 测试PyTorch功能...")
    
    try:
        import torch
        
        # 检查PyTorch版本
        print(f"   PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"   ✅ CUDA可用 - GPU: {torch.cuda.get_device_name()}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            print("   ⚠️  CUDA不可用，将使用CPU")
        
        # 测试基本张量操作
        x = torch.randn(2, 3)
        y = x + 1
        print("   ✅ 张量操作正常")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch测试失败: {e}")
        return False

def test_opencv_functionality():
    """测试OpenCV基本功能"""
    print("\n📸 测试OpenCV功能...")
    
    try:
        import cv2
        import numpy as np
        
        print(f"   OpenCV版本: {cv2.__version__}")
        
        # 创建测试图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
        
        # 测试图像操作
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("   ✅ 图像处理功能正常")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV测试失败: {e}")
        return False

def test_flask_functionality():
    """测试Flask基本功能"""
    print("\n🌐 测试Flask功能...")
    
    try:
        from flask import Flask
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/test')
        def test_route():
            return {'status': 'ok'}
        
        print("   ✅ Flask应用创建成功")
        return True
        
    except Exception as e:
        print(f"   ❌ Flask测试失败: {e}")
        return False

def print_system_info():
    """显示系统信息"""
    print("\n💻 系统信息:")
    print(f"   操作系统: {os.name}")
    print(f"   Python执行路径: {sys.executable}")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   Python路径: {':'.join(sys.path[:3])}...")

def main():
    """主测试函数"""
    print("🧪 YOLOv智能标注工具 - 安装测试")
    print("=" * 50)
    
    # 显示系统信息
    print_system_info()
    
    # 运行所有测试
    tests = [
        ("Python版本", check_python_version),
        ("依赖包导入", test_imports),
        ("必要文件", check_files),
        ("目录结构", check_directories),
        ("PyTorch功能", test_torch_functionality),
        ("OpenCV功能", test_opencv_functionality),
        ("Flask功能", test_flask_functionality)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {test_name} 测试出错: {e}")
            failed += 1
    
    # 显示总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print(f"   ✅ 通过: {passed}")
    print(f"   ❌ 失败: {failed}")
    print(f"   📊 总计: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！YOLOv智能标注工具已准备就绪")
        print("💡 现在可以运行 start.bat 或 python start_server.py 启动服务")
        return True
    else:
        print(f"\n⚠️  发现 {failed} 个问题，请先解决后再启动服务")
        print("💡 建议:")
        print("   1. 运行 install_dependencies.bat 安装依赖")
        print("   2. 检查Python版本是否为3.8+")
        print("   3. 确保网络连接正常")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
