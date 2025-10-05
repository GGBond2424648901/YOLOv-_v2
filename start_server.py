#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv智能标注工具 - 启动脚本
一键启动服务器并打开浏览器
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"当前版本：{sys.version}")
        return False
    return True

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        ('flask', 'Flask'),
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('ultralytics', 'Ultralytics'),
        ('flask_cors', 'Flask-CORS')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n🔧 检测到 {len(missing_packages)} 个缺失的依赖包")
        print("正在自动安装依赖...")
        
        return install_dependencies()
    else:
        print("✅ 所有依赖包检查通过")
        return True

def install_dependencies():
    """自动安装依赖包"""
    try:
        print("📦 正在安装依赖包，请稍候...")
        
        # 检查requirements.txt是否存在
        if not Path("requirements.txt").exists():
            print("❌ 错误：找不到 requirements.txt 文件")
            return False
        
        # 使用subprocess安装依赖
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ 依赖包安装成功！")
            print("🔄 重新检查依赖...")
            
            # 重新检查依赖（不递归调用install_dependencies）
            required_packages = [
                ('flask', 'Flask'),
                ('torch', 'PyTorch'),
                ('cv2', 'OpenCV'),
                ('PIL', 'Pillow'),
                ('numpy', 'NumPy'),
                ('ultralytics', 'Ultralytics'),
                ('flask_cors', 'Flask-CORS')
            ]
            
            for package, name in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    print(f"⚠️  {name} 安装可能失败，请手动安装")
                    return False
            
            print("✅ 所有依赖包验证通过")
            return True
        else:
            print("❌ 依赖包安装失败")
            print("错误信息:", result.stderr)
            print("\n🔧 请手动运行以下命令:")
            print("   pip install -r requirements.txt")
            return False
            
    except Exception as e:
        print(f"❌ 安装依赖时出错: {e}")
        print("\n🔧 请手动运行以下命令:")
        print("   pip install -r requirements.txt")
        return False

def create_directories():
    """创建必要的目录"""
    directories = ['uploads', 'models', 'results']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 创建目录: {directory}")

def start_server():
    """启动服务器"""
    try:
        print("🚀 启动YOLOv智能标注工具服务器...")
        print("=" * 60)
        
        # 检查端口是否被占用
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', 5000)) == 0:
                print("⚠️  警告：端口5000已被占用")
                print("💡 解决方案：")
                print("   1. 关闭其他占用5000端口的程序")
                print("   2. 或者修改配置使用其他端口")
                return False
        
        # 导入并启动Flask服务器
        from yolo_annotation_server import app
        
        print("✅ 服务器启动成功！")
        print("🌐 请在浏览器中访问: http://localhost:5000")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        return True
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("💡 可能的原因：")
        print("   1. yolo_annotation_server.py 文件缺失或损坏")
        print("   2. 某些依赖包安装不完整")
        print("   3. Python路径配置问题")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 建议：")
        print("   1. 检查防火墙设置")
        print("   2. 确保有足够的系统权限")
        print("   3. 查看详细错误信息进行排查")
        return False

def open_browser():
    """在默认浏览器中打开应用"""
    url = "http://localhost:5000"
    time.sleep(2)  # 等待服务器启动
    try:
        webbrowser.open(url)
        print(f"🌐 已在浏览器中打开: {url}")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print(f"请手动在浏览器中访问: {url}")

def main():
    """主函数"""
    print("🎯 YOLOv智能标注工具 - 启动程序")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    print("🔍 检查依赖包...")
    if not check_dependencies():
        print("\n❌ 依赖包检查失败，无法启动服务器")
        print("💡 解决方案：")
        print("   1. 运行: pip install -r requirements.txt")
        print("   2. 或使用国内镜像: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt")
        print("   3. 确保网络连接正常")
        sys.exit(1)
    
    print("\n✅ 所有依赖包准备完成")
    
    # 创建目录
    print("📁 创建必要目录...")
    create_directories()
    
    # 检查必要文件
    required_files = ['yolo_annotation_server.py', 'yolo_annotation_tool.html']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ 缺少必要文件: {file}")
            sys.exit(1)
    
    print("✅ 文件检查通过")
    print()
    
    # 显示系统信息
    import torch
    print("💻 系统信息:")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name()}")
    print()
    
    # 显示使用说明
    print("📖 使用说明:")
    print("1. 服务器启动后会自动打开浏览器")
    print("2. 在网页中上传您的YOLOv模型文件(.pt)")
    print("3. 上传图片或视频进行标注")
    print("4. 调整参数并点击自动标注")
    print("5. 导出标注数据为训练格式")
    print("6. 按Ctrl+C停止服务器")
    print("=" * 60)
    
    # 启动浏览器（在后台线程中）
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动服务器
    if not start_server():
        print("\n❌ 服务器启动失败")
        print("🔧 请检查上述错误信息并尝试解决")
        sys.exit(1)

if __name__ == '__main__':
    main()
