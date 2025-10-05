#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv智能标注工具 - 演示脚本
下载示例模型和图片进行演示
"""

import os
import requests
from pathlib import Path

def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ {filename} 下载完成")
        return True
    except Exception as e:
        print(f"✗ {filename} 下载失败: {e}")
        return False

def setup_demo():
    """设置演示环境"""
    print("🎯 YOLOv智能标注工具 - 演示设置")
    print("=" * 50)
    
    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('demo_images', exist_ok=True)
    
    # 下载示例模型（YOLOv5s）
    model_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    model_path = "models/yolov5s.pt"
    
    if not Path(model_path).exists():
        print("📥 下载YOLOv5s示例模型...")
        if download_file(model_url, model_path):
            print(f"模型已保存到: {model_path}")
        else:
            print("模型下载失败，请手动下载并放入models目录")
    else:
        print("✓ 示例模型已存在")
    
    # 示例图片URL列表
    image_urls = [
        ("https://ultralytics.com/images/bus.jpg", "demo_images/bus.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "demo_images/zidane.jpg"),
    ]
    
    # 下载示例图片
    print("\n📥 下载示例图片...")
    for url, path in image_urls:
        if not Path(path).exists():
            download_file(url, path)
        else:
            print(f"✓ {path} 已存在")
    
    print("\n" + "=" * 50)
    print("🎉 演示环境设置完成！")
    print("\n📖 使用说明:")
    print("1. 运行 'python start_server.py' 启动服务器")
    print("2. 在浏览器中上传 models/yolov5s.pt 模型")
    print("3. 上传 demo_images/ 中的图片进行测试")
    print("4. 体验自动标注和导出功能")
    print("=" * 50)

if __name__ == '__main__':
    setup_demo()
