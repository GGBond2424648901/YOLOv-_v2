# 🎯 YOLOv智能标注工具

一个功能强大的网页端YOLOv自动标注工具，支持加载自定义训练模型，自动标注图片和视频，并导出为多种训练格式。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.13+-red.svg)

## ✨ 主要特性

- 🧠 **支持多种YOLO模型**: YOLOv5、YOLOv8等预训练模型
- 🖼️ **智能图像标注**: 自动识别和标注图片中的目标
- 🎥 **视频处理支持**: 视频帧提取和批量标注
- 📊 **多格式导出**: 支持YOLO JSON、COCO JSON、YOLO TXT、Pascal VOC格式
- 🚀 **批量处理**: 高效处理大量图片文件
- 🎨 **现代化界面**: 响应式设计，操作简单直观
- ⚡ **实时预览**: 实时显示检测结果和置信度
- 🛠️ **参数调优**: 可调节置信度阈值、IoU阈值等参数
- 📋 **图片列表管理**: 可视化展示所有图片，标注状态一目了然
- 🔇 **静默批量检测**: 大量图片处理时不轮播，只显示进度
- 🎯 **快速跳转**: 点击列表直接定位到指定图片

## 🛠️ 安装指南

### 1. 环境要求

- Python 3.8 或更高版本
- 推荐使用 Python 3.9 或 3.10
- GPU支持（可选，推荐用于更快的推理速度）

### 2. 快速启动（推荐）

#### Windows用户
双击运行 `start.bat` 文件，脚本将自动：
- ✅ 检查Python环境
- ✅ 自动安装所需依赖包
- ✅ 启动服务器
- ✅ 自动打开浏览器

#### 如果遇到问题
运行 `install_dependencies.bat` 来专门安装依赖包，然后再运行 `start.bat`

#### Python用户
```bash
python start_server.py
```

### 3. 手动安装（可选）

如果自动安装失败，可以手动执行以下步骤：

#### 克隆项目

```bash
git clone https://github.com/your-username/yolo-annotation-tool.git
cd yolo-annotation-tool
```

### 3. 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 如果您有NVIDIA GPU，安装CUDA版本的PyTorch（可选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. 启动服务

```bash
# 方式1：使用启动脚本（推荐）
python start_server.py

# 方式2：直接启动服务器
python yolo_annotation_server.py
```

### 5. 访问应用

启动后自动打开浏览器，或手动访问 [http://localhost:5000](http://localhost:5000)

## 🚀 快速开始

### 第一步：上传模型

1. 在左侧"模型管理"区域点击上传区域
2. 选择您的YOLOv模型文件（.pt格式）
3. 等待模型加载完成

### 第二步：上传图片

1. 在中央面板点击"选择图片或视频"
2. 选择要标注的图片文件
3. 图片将在画布中显示

### 第三步：自动标注

1. 调整右侧的置信度阈值（推荐0.5）
2. 点击"自动标注"按钮
3. 等待模型处理完成

### 第四步：导出数据

1. 在右侧选择导出格式
2. 点击"导出标注数据"
3. 文件将自动下载

## 📁 项目结构

```
yolo-annotation-tool/
├── yolo_annotation_tool.html    # 前端界面
├── yolo_annotation_server.py    # 后端API服务
├── start_server.py              # 启动脚本
├── requirements.txt             # Python依赖
├── README.md                    # 项目文档
├── uploads/                     # 上传文件目录
├── models/                      # 模型文件目录
└── results/                     # 导出结果目录
```

## 🎮 使用指南

### 模型管理

- **支持格式**: `.pt` 文件（YOLOv5/YOLOv8）
- **模型来源**: 
  - [YOLOv5官方模型](https://github.com/ultralytics/yolov5)
  - [YOLOv8官方模型](https://github.com/ultralytics/ultralytics)
  - 自己训练的模型

### 参数设置

- **置信度阈值**: 控制检测的敏感度（0.1-1.0）
- **IoU阈值**: 控制重叠框的过滤（0.1-1.0）
- **输入尺寸**: 模型输入图像大小（320-1280）

### 批量处理

1. 选择多个图片文件
2. 设置处理参数
3. 点击"开始批量处理"
4. 等待处理完成

### 导出格式

| 格式 | 描述 | 用途 | 文件结构 |
|------|------|------|---------|
| YOLO JSON | LabelMe标准JSON格式 | 通用标注工具 | 图片和JSON在同一目录 |
| COCO JSON | COCO数据集标准格式 | 对象检测基准 | 图片和JSON在同一目录 |
| YOLO TXT | YOLO训练格式 | YOLOv5/v8训练 | 分离的images/和labels/目录 |
| Pascal VOC | XML格式 | 传统目标检测 | 分离的images/和labels/目录 |

**导出数据集结构：**
```
# YOLO JSON/COCO JSON格式
my_dataset/
├── images/
│   ├── image1.jpg
│   ├── image1.json    # 和图片同名的JSON标注文件
│   ├── image2.jpg
│   └── image2.json
└── README.md

# YOLO TXT格式
my_dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
├── labels/
│   ├── image1.txt     # YOLO格式标注
│   └── image2.txt
├── dataset.yaml       # YOLO配置文件
└── README.md
```

## 🔧 高级功能

### 手动标注

- 在图片上拖拽鼠标绘制边界框
- 双击标注框进行编辑
- 支持删除和修改标注

### 视频处理

- 支持多种视频格式
- 实时提取视频帧
- 批量标注视频序列

### API接口

后端提供RESTful API接口：

- `POST /api/upload_model` - 上传模型
- `POST /api/upload_media` - 上传媒体文件
- `POST /api/predict` - 单张图片预测
- `POST /api/batch_predict` - 批量预测
- `POST /api/export_annotations` - 导出标注

## 🔧 故障排除

### 启动问题

#### 1. 缺少依赖包错误
```
❌ 缺少依赖包: No module named 'flask'
```
**解决方案**：
- **Windows用户**：运行 `install_dependencies.bat`
- **手动安装**：`pip install -r requirements.txt`
- **使用国内镜像**：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

#### 2. 中文显示乱码（Windows）
**已修复**：最新版 `start.bat` 自动设置UTF-8编码  
如仍有问题，请以管理员身份运行

#### 3. 端口5000被占用
```
⚠️ 警告：端口5000已被占用
```
**解决方案**：
- 关闭占用5000端口的其他程序
- 或修改 `start_server.py` 中的端口号

#### 4. 安装依赖时网络超时
**解决方案**：
```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 或使用阿里镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

### 自动安装功能
- **start.bat** 会自动检查并安装缺失的依赖包
- 如果自动安装失败，会提示手动安装命令
- 提供了详细的错误信息和解决建议

## 🐛 常见问题

### Q: 模型加载失败怎么办？
A: 请确保模型文件是有效的.pt格式，并检查是否安装了ultralytics包。

### Q: GPU不可用怎么办？
A: 工具会自动使用CPU，但处理速度较慢。可以安装CUDA版本的PyTorch来启用GPU加速。

### Q: 支持哪些图片格式？
A: 支持PNG、JPG、JPEG、GIF、BMP、TIFF等常见格式。

### Q: 如何训练自己的模型？
A: 可以使用导出的标注数据配合YOLOv5或YOLOv8进行训练。

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5) - 强大的目标检测框架
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 最新的YOLO实现
- [Flask](https://flask.palletsprojects.com/) - 轻量级Web框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/yolo-annotation-tool/issues)
- 发送邮件到 your-email@example.com

---

⭐ 如果这个项目对您有帮助，请考虑给它一个星标！
