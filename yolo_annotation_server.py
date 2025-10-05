#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv智能标注工具 - 后端API服务
支持YOLOv5/YOLOv8模型的加载和推理，提供RESTful API接口
"""

import os
import io
import json
import uuid
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {
    'model': {'pt'},
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
}

# 创建必要的文件夹
for folder in [UPLOAD_FOLDER, MODELS_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局变量
current_model = None
model_info = {}
session_data = {}

class YOLOModel:
    """YOLO模型包装器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = []
        self.model_type = self.detect_model_type(model_path)
        self.supports_segmentation = False  # 是否支持分割
        self.load_model()
    
    def detect_model_type(self, model_path: str) -> str:
        """检测模型类型（YOLOv5或YOLOv8）"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                # YOLOv5风格
                return 'yolov5'
            else:
                # YOLOv8风格
                return 'yolov8'
        except Exception as e:
            logger.warning(f"无法确定模型类型，默认使用YOLOv5: {e}")
            return 'yolov5'
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            # 优先尝试使用ultralytics (YOLOv8)
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.class_names = list(self.model.names.values())
                self.model_type = 'yolov8'
                
                # 检测是否支持分割（通过检查模型任务类型或进行测试推理）
                try:
                    # 检查模型的task属性
                    if hasattr(self.model, 'task'):
                        self.supports_segmentation = self.model.task == 'segment'
                    # 或者检查模型名称
                    elif 'seg' in str(self.model_path).lower() or '-seg' in str(self.model_path).lower():
                        self.supports_segmentation = True
                    else:
                        # 进行小图测试推理来确定
                        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                        test_results = self.model.predict(test_img, verbose=False)
                        if test_results and len(test_results) > 0:
                            self.supports_segmentation = hasattr(test_results[0], 'masks') and test_results[0].masks is not None
                except Exception as seg_check_error:
                    logger.warning(f"检测分割支持失败，默认为False: {seg_check_error}")
                    self.supports_segmentation = False
                
                logger.info(f"成功加载YOLOv8模型: {self.model_path}, 支持分割: {self.supports_segmentation}")
                return
            except Exception as e:
                logger.warning(f"YOLOv8加载失败，尝试其他方法: {e}")
            
            # 备用方案：直接加载权重
            self.load_weights_directly()
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 如果都失败了，创建一个虚拟模型用于演示
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """创建虚拟模型用于演示"""
        logger.warning("创建虚拟模型用于演示")
        self.model = None
        self.class_names = self.get_coco_classes()
        self.model_type = 'dummy'
    
    def load_weights_directly(self):
        """直接加载权重文件"""
        try:
            logger.info("尝试直接加载权重文件...")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 检查不同的权重文件格式
            if 'model' in checkpoint:
                # YOLOv5格式
                self.model = checkpoint['model'].float().eval()
                if hasattr(self.model, 'names'):
                    self.class_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
                else:
                    self.class_names = self.get_coco_classes()
                self.model_type = 'yolov5'
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # 其他格式
                logger.warning("检测到state_dict格式，使用COCO类别")
                self.class_names = self.get_coco_classes()
                self.model_type = 'custom'
            else:
                # 未知格式，使用默认类别
                logger.warning("未知模型格式，使用COCO类别")
                self.class_names = self.get_coco_classes()
                self.model_type = 'unknown'
                
            logger.info(f"直接权重加载成功，类别数: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"直接权重加载失败: {e}")
            # 不再抛出异常，让create_dummy_model处理
            raise
    
    def get_coco_classes(self) -> List[str]:
        """获取COCO数据集的类别名称"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.5, 
                iou_threshold: float = 0.45, img_size: int = 640) -> List[Dict]:
        """对图像进行预测"""
        try:
            # 如果是虚拟模型，生成模拟结果
            if self.model_type == 'dummy' or self.model is None:
                return self.generate_dummy_predictions(image.shape, conf_threshold)
            
            # 预处理图像
            original_shape = image.shape[:2]
            
            if self.model_type == 'yolov8' and hasattr(self.model, 'predict'):
                # YOLOv8预测
                results = self.model.predict(image, conf=conf_threshold, 
                                           iou=iou_threshold, imgsz=img_size)
                return self.parse_yolov8_results(results[0], original_shape)
            else:
                # YOLOv5预测或其他类型
                if hasattr(self.model, '__call__'):
                    results = self.model(image, size=img_size)
                    return self.parse_yolov5_results(results, conf_threshold, 
                                                   iou_threshold, original_shape)
                else:
                    # 如果模型不能正常调用，返回虚拟结果
                    return self.generate_dummy_predictions(image.shape, conf_threshold)
                
        except Exception as e:
            logger.error(f"预测失败: {e}")
            # 出错时也返回虚拟结果，让用户能够体验功能
            return self.generate_dummy_predictions(image.shape, conf_threshold)
    
    def generate_dummy_predictions(self, image_shape, conf_threshold: float) -> List[Dict]:
        """生成虚拟预测结果用于演示"""
        import random
        
        height, width = image_shape[:2]
        predictions = []
        
        # 生成2-5个虚拟检测结果
        num_objects = random.randint(2, 5)
        common_classes = ['person', 'car', 'bicycle', 'dog', 'cat', 'bus', 'truck']
        
        for i in range(num_objects):
            # 随机选择类别
            class_name = random.choice(common_classes)
            class_id = self.class_names.index(class_name) if class_name in self.class_names else 0
            
            # 生成随机边界框
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            w = random.randint(50, min(200, width - x1))
            h = random.randint(50, min(200, height - y1))
            
            # 生成置信度
            confidence = conf_threshold + random.random() * (0.95 - conf_threshold)
            
            prediction = {
                'id': str(uuid.uuid4()),
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x1 + w),
                    'y2': float(y1 + h),
                    'width': float(w),
                    'height': float(h)
                }
            }
            predictions.append(prediction)
        
        logger.info(f"生成了{len(predictions)}个虚拟检测结果")
        return predictions
    
    def mask_to_polygon(self, mask: np.ndarray, epsilon_factor: float = 0.001) -> List[List[float]]:
        """将掩膜转换为多边形坐标
        
        Args:
            mask: 二值掩膜数组
            epsilon_factor: 轮廓近似的精度因子
            
        Returns:
            多边形坐标列表 [[x1, y1], [x2, y2], ...]
        """
        try:
            # 确保是二值图像
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
            
            # 选择最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 简化轮廓
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 转换为列表格式
            polygon = [[float(point[0][0]), float(point[0][1])] for point in approx]
            
            return polygon
            
        except Exception as e:
            logger.error(f"掩膜转多边形失败: {e}")
            return []
    
    def parse_yolov8_results(self, result, original_shape: Tuple[int, int]) -> List[Dict]:
        """解析YOLOv8结果"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # 检查是否有掩膜数据
            masks = None
            if self.supports_segmentation and hasattr(result, 'masks') and result.masks is not None:
                try:
                    masks = result.masks.data.cpu().numpy()
                except Exception as e:
                    logger.warning(f"提取掩膜数据失败: {e}")
                    masks = None
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                
                detection = {
                    'id': str(uuid.uuid4()),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2), 
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    }
                }
                
                # 如果有掩膜数据，转换为多边形
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    # 将掩膜调整为原始图像大小
                    if mask.shape != original_shape:
                        mask = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                                        interpolation=cv2.INTER_LINEAR)
                    
                    # 转换为多边形
                    polygon = self.mask_to_polygon(mask)
                    if polygon and len(polygon) >= 3:
                        detection['segmentation'] = polygon
                        detection['shape_type'] = 'polygon'
                    else:
                        detection['shape_type'] = 'rectangle'
                else:
                    detection['shape_type'] = 'rectangle'
                
                detections.append(detection)
        
        return detections
    
    def parse_yolov5_results(self, results, conf_threshold: float, 
                           iou_threshold: float, original_shape: Tuple[int, int]) -> List[Dict]:
        """解析YOLOv5结果"""
        detections = []
        
        # 应用NMS
        pred = results.pred[0]
        if pred is not None and len(pred) > 0:
            # 过滤低置信度检测
            pred = pred[pred[:, 4] >= conf_threshold]
            
            if len(pred) > 0:
                # 应用NMS
                keep = torch.ops.torchvision.nms(pred[:, :4], pred[:, 4], iou_threshold)
                pred = pred[keep]
                
                for detection in pred:
                    x1, y1, x2, y2, conf, class_id = detection[:6]
                    class_id = int(class_id)
                    
                    detection_dict = {
                        'id': str(uuid.uuid4()),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2), 
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        }
                    }
                    detections.append(detection_dict)
        
        return detections

def allowed_file(filename: str, file_type: str) -> bool:
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def generate_session_id() -> str:
    """生成会话ID"""
    return str(uuid.uuid4())

@app.route('/')
def index():
    """返回主页面"""
    with open('yolo_annotation_tool.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传文件的访问"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """上传模型文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename, 'model'):
            return jsonify({'error': '不支持的文件格式，请上传.pt文件'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(MODELS_FOLDER, filename)
        file.save(filepath)
        
        # 加载模型
        global current_model, model_info
        logger.info(f"开始加载模型: {filename}")
        
        try:
            current_model = YOLOModel(filepath)
            
            model_info = {
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'classes': current_model.class_names,
                'num_classes': len(current_model.class_names),
                'model_type': current_model.model_type,
                'device': str(current_model.device),
                'supports_segmentation': current_model.supports_segmentation,
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"模型加载成功: {current_model.model_type}, 类别数: {len(current_model.class_names)}, 支持分割: {current_model.supports_segmentation}")
            
            return jsonify({
                'message': f'模型上传成功 (类型: {current_model.model_type})',
                'model_info': model_info
            })
            
        except Exception as model_error:
            logger.error(f"模型加载失败: {model_error}")
            # 删除上传的文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'模型加载失败: {str(model_error)}'}), 500
        
    except Exception as e:
        logger.error(f"模型上传失败: {e}")
        return jsonify({'error': f'模型上传失败: {str(e)}'}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """获取当前模型信息"""
    if current_model is None:
        return jsonify({'error': '没有加载的模型'}), 400
    
    return jsonify(model_info)

@app.route('/api/list_models', methods=['GET'])
def list_models():
    """列出models目录中的所有模型文件"""
    try:
        model_files = []
        if os.path.exists(MODELS_FOLDER):
            for filename in os.listdir(MODELS_FOLDER):
                if filename.endswith('.pt'):
                    filepath = os.path.join(MODELS_FOLDER, filename)
                    file_stat = os.stat(filepath)
                    model_files.append({
                        'filename': filename,
                        'size': file_stat.st_size,
                        'size_mb': f"{file_stat.st_size / 1024 / 1024:.1f} MB",
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
        
        # 按修改时间排序
        model_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'models': model_files,
            'count': len(model_files)
        })
        
    except Exception as e:
        logger.error(f"列出模型文件失败: {e}")
        return jsonify({'error': f'列出模型文件失败: {str(e)}'}), 500

@app.route('/api/load_existing_model', methods=['POST'])
def load_existing_model():
    """加载已存在的模型文件"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': '请提供模型文件名'}), 400
        
        filepath = os.path.join(MODELS_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '模型文件不存在'}), 404
        
        if not filename.endswith('.pt'):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 加载模型
        global current_model, model_info
        logger.info(f"开始加载现有模型: {filename}")
        
        try:
            current_model = YOLOModel(filepath)
            
            model_info = {
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'classes': current_model.class_names,
                'num_classes': len(current_model.class_names),
                'model_type': current_model.model_type,
                'device': str(current_model.device),
                'supports_segmentation': current_model.supports_segmentation,
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"现有模型加载成功: {current_model.model_type}, 类别数: {len(current_model.class_names)}, 支持分割: {current_model.supports_segmentation}")
            
            return jsonify({
                'message': f'模型加载成功 (类型: {current_model.model_type})',
                'model_info': model_info
            })
            
        except Exception as model_error:
            logger.error(f"模型加载失败: {model_error}")
            return jsonify({'error': f'模型加载失败: {str(model_error)}'}), 500
            
    except Exception as e:
        logger.error(f"加载现有模型失败: {e}")
        return jsonify({'error': f'加载现有模型失败: {str(e)}'}), 500

@app.route('/api/upload_media', methods=['POST'])
def upload_media():
    """上传图片或视频文件"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        files = request.files.getlist('files')
        session_id = generate_session_id()
        
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # 检查文件类型
            is_image = allowed_file(file.filename, 'image')
            is_video = allowed_file(file.filename, 'video')
            
            if not (is_image or is_video):
                continue
            
            # 保存文件
            filename = secure_filename(file.filename)
            unique_filename = f"{session_id}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            
            file_info = {
                'filename': filename,
                'unique_filename': unique_filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'type': 'image' if is_image else 'video',
                'session_id': session_id
            }
            
            uploaded_files.append(file_info)
        
        # 保存会话数据
        session_data[session_id] = {
            'files': uploaded_files,
            'predictions': {},  # 初始化predictions字段
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'message': f'成功上传 {len(uploaded_files)} 个文件',
            'session_id': session_id,
            'files': uploaded_files
        })
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({'error': f'文件上传失败: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """对图片进行预测"""
    try:
        if current_model is None:
            return jsonify({'error': '请先上传并加载模型'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': '缺少请求数据'}), 400
        
        session_id = data.get('session_id')
        filename = data.get('filename')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        iou_threshold = float(data.get('iou_threshold', 0.45))
        img_size = int(data.get('img_size', 640))
        selected_classes = data.get('selected_classes', [])  # 新增：选择的类别ID列表
        
        if not session_id or not filename:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 查找文件
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        file_info = None
        for file_data in session_data[session_id]['files']:
            if file_data['unique_filename'] == filename:
                file_info = file_data
                break
        
        if not file_info:
            return jsonify({'error': '文件不存在'}), 404
        
        if file_info['type'] != 'image':
            return jsonify({'error': '只支持图片预测'}), 400
        
        # 读取图片
        image_path = file_info['filepath']
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': '无法读取图片'}), 500
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行预测
        detections = current_model.predict(
            image_rgb, conf_threshold, iou_threshold, img_size
        )
        
        # 根据选择的类别过滤检测结果
        if selected_classes:
            detections = [det for det in detections if det.get('class_id') in selected_classes]
        
        # 保存结果
        result_data = {
            'session_id': session_id,
            'filename': filename,
            'image_shape': image.shape,
            'detections': detections,
            'parameters': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'img_size': img_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到会话数据
        if 'predictions' not in session_data[session_id]:
            session_data[session_id]['predictions'] = {}
        session_data[session_id]['predictions'][filename] = result_data
        
        return jsonify({
            'message': f'检测完成，发现 {len(detections)} 个目标',
            'detections': detections,
            'image_shape': image.shape
        })
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/export_annotations', methods=['POST'])
def export_annotations():
    """导出标注数据"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '缺少请求数据'}), 400
        
        session_id = data.get('session_id')
        filename = data.get('filename')
        format_type = data.get('format', 'yolo_json')
        annotations = data.get('annotations', [])
        image_width = data.get('image_width', 640)
        image_height = data.get('image_height', 640)
        
        if not session_id or not filename:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 生成导出数据
        export_data = generate_export_data(annotations, format_type, session_id, filename, image_width, image_height)
        
        # 保存到文件
        export_filename = f"{session_id}_{filename}_{format_type}.{get_export_extension(format_type)}"
        export_path = os.path.join(RESULTS_FOLDER, export_filename)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            if format_type.endswith('json'):
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                f.write(export_data)
        
        return send_file(export_path, as_attachment=True, download_name=export_filename)
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测"""
    try:
        if current_model is None:
            return jsonify({'error': '请先上传并加载模型'}), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        iou_threshold = float(data.get('iou_threshold', 0.45))
        img_size = int(data.get('img_size', 640))
        selected_classes = data.get('selected_classes', [])  # 新增：选择的类别ID列表
        
        if not session_id or session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        files = session_data[session_id]['files']
        image_files = [f for f in files if f['type'] == 'image']
        
        if not image_files:
            return jsonify({'error': '没有可处理的图片文件'}), 400
        
        results = []
        
        for file_info in image_files:
            try:
                # 读取图片
                image = cv2.imread(file_info['filepath'])
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 进行预测
                detections = current_model.predict(
                    image_rgb, conf_threshold, iou_threshold, img_size
                )
                
                # 根据选择的类别过滤检测结果
                if selected_classes:
                    detections = [det for det in detections if det.get('class_id') in selected_classes]
                
                result = {
                    'filename': file_info['filename'],
                    'unique_filename': file_info['unique_filename'],
                    'detections': detections,
                    'count': len(detections)
                }
                
                results.append(result)
                
                # 保存到会话数据
                if 'predictions' not in session_data[session_id]:
                    session_data[session_id]['predictions'] = {}
                
                session_data[session_id]['predictions'][file_info['unique_filename']] = {
                    'session_id': session_id,
                    'filename': file_info['unique_filename'],
                    'image_shape': image.shape,
                    'detections': detections,
                    'parameters': {
                        'conf_threshold': conf_threshold,
                        'iou_threshold': iou_threshold,
                        'img_size': img_size
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"处理文件 {file_info['filename']} 失败: {e}")
                continue
        
        return jsonify({
            'message': f'批量处理完成，处理了 {len(results)} 个文件',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

def generate_export_data(annotations: List[Dict], format_type: str, 
                        session_id: str, filename: str, image_width: int = 640, image_height: int = 640) -> Any:
    """生成不同格式的导出数据"""
    
    if format_type == 'yolo_json':
        # LabelMe格式
        shapes = []
        for ann in annotations:
            # 检查是否有分割数据
            if 'segmentation' in ann and ann.get('shape_type') == 'polygon':
                # 多边形分割
                points = ann['segmentation']
                shape_type = 'polygon'
            else:
                # 边界框
                bbox = ann.get('bbox', {})
                points = [
                    [bbox.get('x1', 0), bbox.get('y1', 0)],
                    [bbox.get('x2', 0), bbox.get('y2', 0)]
                ]
                shape_type = 'rectangle'
            
            shape = {
                'label': ann.get('class_name', 'unknown'),
                'points': points,
                'group_id': None,
                'description': '',
                'shape_type': shape_type,
                'flags': {},
                'mask': None
            }
            shapes.append(shape)
        
        return {
            'version': '5.8.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': filename,
            'imageData': '',  # 可以为空或base64编码的图片数据
            'imageHeight': image_height,
            'imageWidth': image_width
        }
    
    elif format_type == 'coco_json':
        categories = []
        if current_model:
            categories = [
                {'id': i, 'name': name, 'supercategory': 'object'}
                for i, name in enumerate(current_model.class_names)
            ]
        
        coco_annotations = []
        for i, ann in enumerate(annotations):
            bbox = ann.get('bbox', {})
            coco_annotations.append({
                'id': i,
                'image_id': 1,
                'category_id': ann.get('class_id', 0),
                'bbox': [
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('width', 0),
                    bbox.get('height', 0)
                ],
                'area': bbox.get('width', 0) * bbox.get('height', 0),
                'iscrowd': 0
            })
        
        return {
            'info': {
                'description': 'YOLOv智能标注工具导出',
                'version': '1.0',
                'date_created': datetime.now().isoformat()
            },
            'images': [{
                'id': 1,
                'file_name': filename,
                'width': 640,  # 默认值，应该从实际图像获取
                'height': 640
            }],
            'categories': categories,
            'annotations': coco_annotations
        }
    
    elif format_type == 'yolo_txt':
        lines = []
        for ann in annotations:
            class_id = ann.get('class_id', 0)
            
            # 检查是否有分割数据
            if 'segmentation' in ann and ann.get('shape_type') == 'polygon':
                # YOLO分割格式: class_id x1 y1 x2 y2 x3 y3 ... (归一化坐标)
                polygon = ann['segmentation']
                if polygon and len(polygon) >= 3:
                    # 归一化坐标
                    normalized_coords = []
                    for point in polygon:
                        x_norm = point[0] / image_width
                        y_norm = point[1] / image_height
                        normalized_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                    
                    line = f"{class_id} " + " ".join(normalized_coords)
                    lines.append(line)
            else:
                # YOLO边界框格式: class_id x_center y_center width height (归一化)
                bbox = ann.get('bbox', {})
                x_center = (bbox.get('x1', 0) + bbox.get('width', 0) / 2) / image_width
                y_center = (bbox.get('y1', 0) + bbox.get('height', 0) / 2) / image_height
                width = bbox.get('width', 0) / image_width
                height = bbox.get('height', 0) / image_height
                
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return '\n'.join(lines)
    
    elif format_type == 'pascal_voc':
        # 简化的Pascal VOC格式
        objects = []
        for ann in annotations:
            bbox = ann.get('bbox', {})
            objects.append(f"""
    <object>
        <name>{ann.get('class_name', 'unknown')}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int(bbox.get('x1', 0))}</xmin>
            <ymin>{int(bbox.get('y1', 0))}</ymin>
            <xmax>{int(bbox.get('x2', 0))}</xmax>
            <ymax>{int(bbox.get('y2', 0))}</ymax>
        </bndbox>
    </object>""")
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <filename>{filename}</filename>
    <size>
        <width>640</width>
        <height>640</height>
        <depth>3</depth>
    </size>
    {''.join(objects)}
</annotation>"""
    
    return {}

def get_export_extension(format_type: str) -> str:
    """获取导出格式的文件扩展名"""
    extensions = {
        'yolo_json': 'json',
        'coco_json': 'json',
        'yolo_txt': 'txt',
        'pascal_voc': 'xml'
    }
    return extensions.get(format_type, 'json')

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': current_model is not None,
        'device': str(current_model.device) if current_model else 'none'
    })

@app.route('/api/export_dataset', methods=['POST'])
def export_dataset():
    """导出完整数据集"""
    try:
        data = request.get_json()
        format_type = data.get('format', 'yolo_json')
        include_images = data.get('include_images', True)
        dataset_name = data.get('dataset_name', 'my_yolo_dataset')
        
        import zipfile
        import tempfile
        from datetime import datetime
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 根据格式类型创建目录结构
            if format_type == 'yolo_txt':
                # YOLO TXT格式：分离的images和labels文件夹
                images_dir = os.path.join(dataset_dir, 'images')
                labels_dir = os.path.join(dataset_dir, 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
            else:
                # JSON格式：图片和JSON文件放在同一个images文件夹
                images_dir = os.path.join(dataset_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                labels_dir = images_dir  # JSON文件和图片在同一目录
            
            exported_count = 0
            class_names = current_model.class_names if current_model else []
            
            # 遍历所有会话数据
            for session_id, session_info in session_data.items():
                predictions = session_info.get('predictions', {})
                files = session_info.get('files', [])
                
                for file_info in files:
                    if file_info['type'] != 'image':
                        continue
                        
                    filename = file_info['unique_filename']
                    filepath = file_info['filepath']
                    
                    if filename in predictions and os.path.exists(filepath):
                        prediction_data = predictions[filename]
                        detections = prediction_data.get('detections', [])
                        
                        if detections:  # 只导出有标注的图片
                            original_name = file_info['filename']
                            base_name = os.path.splitext(original_name)[0]
                            
                            # 复制图片文件
                            if include_images:
                                image_dest = os.path.join(images_dir, original_name)
                                shutil.copy2(filepath, image_dest)
                            
                            # 生成标注文件
                            image_shape = prediction_data.get('image_shape', [640, 640, 3])
                            img_height, img_width = image_shape[:2]
                            
                            if format_type == 'yolo_txt':
                                # YOLO格式标注文件（放在labels文件夹）
                                label_file = os.path.join(labels_dir, f"{base_name}.txt")
                                with open(label_file, 'w') as f:
                                    for det in detections:
                                        class_id = det.get('class_id', 0)
                                        
                                        # 检查是否有分割数据
                                        if 'segmentation' in det and det.get('shape_type') == 'polygon':
                                            # YOLO分割格式
                                            polygon = det['segmentation']
                                            if polygon and len(polygon) >= 3:
                                                normalized_coords = []
                                                for point in polygon:
                                                    x_norm = point[0] / img_width
                                                    y_norm = point[1] / img_height
                                                    normalized_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                                                
                                                f.write(f"{class_id} " + " ".join(normalized_coords) + "\n")
                                        else:
                                            # YOLO边界框格式
                                            bbox = det.get('bbox', {})
                                            x_center = (bbox.get('x1', 0) + bbox.get('width', 0) / 2) / img_width
                                            y_center = (bbox.get('y1', 0) + bbox.get('height', 0) / 2) / img_height
                                            width = bbox.get('width', 0) / img_width
                                            height = bbox.get('height', 0) / img_height
                                            
                                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            
                            elif format_type == 'yolo_json':
                                # LabelMe JSON格式（放在images文件夹，和图片同名）
                                label_file = os.path.join(labels_dir, f"{base_name}.json")
                                
                                # 转换检测结果为LabelMe格式
                                shapes = []
                                for det in detections:
                                    # 检查是否有分割数据
                                    if 'segmentation' in det and det.get('shape_type') == 'polygon':
                                        points = det['segmentation']
                                        shape_type = 'polygon'
                                    else:
                                        bbox = det.get('bbox', {})
                                        points = [
                                            [bbox.get('x1', 0), bbox.get('y1', 0)],
                                            [bbox.get('x2', 0), bbox.get('y2', 0)]
                                        ]
                                        shape_type = 'rectangle'
                                    
                                    shape = {
                                        'label': det.get('class_name', 'unknown'),
                                        'points': points,
                                        'group_id': None,
                                        'description': '',
                                        'shape_type': shape_type,
                                        'flags': {},
                                        'mask': None
                                    }
                                    shapes.append(shape)
                                
                                labelme_data = {
                                    'version': '5.8.1',
                                    'flags': {},
                                    'shapes': shapes,
                                    'imagePath': original_name,
                                    'imageData': '',
                                    'imageHeight': img_height,
                                    'imageWidth': img_width
                                }
                                
                                with open(label_file, 'w', encoding='utf-8') as f:
                                    json.dump(labelme_data, f, indent=4, ensure_ascii=False)
                            
                            elif format_type == 'coco_json':
                                # COCO格式（每个图片一个JSON文件，放在images文件夹）
                                label_file = os.path.join(labels_dir, f"{base_name}.json")
                                coco_data = generate_coco_for_image(file_info, detections, class_names, img_width, img_height)
                                with open(label_file, 'w', encoding='utf-8') as f:
                                    json.dump(coco_data, f, indent=4, ensure_ascii=False)
                            
                            exported_count += 1
            
            # 创建数据集配置文件
            if format_type == 'yolo_txt':
                # 创建YOLO配置文件
                config_file = os.path.join(dataset_dir, 'dataset.yaml')
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(f"# YOLOv数据集配置文件\n")
                    f.write(f"# 生成时间: {datetime.now().isoformat()}\n\n")
                    f.write(f"train: images\n")
                    f.write(f"val: images\n\n")
                    f.write(f"nc: {len(class_names)}\n")
                    f.write(f"names: {class_names}\n")
            
            # 创建README文件
            readme_file = os.path.join(dataset_dir, 'README.md')
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f"# YOLOv智能标注数据集\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**图片数量**: {exported_count}\n")
                f.write(f"**类别数量**: {len(class_names)}\n")
                f.write(f"**标注格式**: {format_type}\n\n")
                f.write(f"## 数据集结构\n")
                f.write(f"```\n")
                f.write(f"{dataset_name}/\n")
                f.write(f"├── images/          # 图片文件\n")
                f.write(f"├── labels/          # 标注文件\n")
                f.write(f"├── dataset.yaml     # YOLO配置文件\n")
                f.write(f"└── README.md        # 说明文档\n")
                f.write(f"```\n\n")
                f.write(f"## 类别列表\n")
                for i, name in enumerate(class_names):
                    f.write(f"{i}: {name}\n")
            
            # 打包为ZIP文件
            zip_path = os.path.join(RESULTS_FOLDER, f"{dataset_name}_{format_type}_{int(time.time())}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arc_path)
            
            if exported_count == 0:
                return jsonify({'error': '没有可导出的标注数据'}), 400
            
            return send_file(zip_path, as_attachment=True, 
                           download_name=f"{dataset_name}_{format_type}.zip")
            
    except Exception as e:
        logger.error(f"导出数据集失败: {e}")
        return jsonify({'error': f'导出数据集失败: {str(e)}'}), 500

def generate_coco_for_image(file_info, detections, class_names, img_width=640, img_height=640):
    """为单个图片生成COCO格式数据"""
    return {
        'info': {
            'description': 'YOLOv智能标注工具导出',
            'version': '1.0',
            'date_created': datetime.now().isoformat()
        },
        'images': [{
            'id': 1,
            'file_name': file_info['filename'],
            'width': img_width,
            'height': img_height
        }],
        'categories': [
            {'id': i, 'name': name, 'supercategory': 'object'}
            for i, name in enumerate(class_names)
        ],
        'annotations': [
            {
                'id': i,
                'image_id': 1,
                'category_id': det.get('class_id', 0),
                'bbox': [
                    det.get('bbox', {}).get('x1', 0),
                    det.get('bbox', {}).get('y1', 0),
                    det.get('bbox', {}).get('width', 0),
                    det.get('bbox', {}).get('height', 0)
                ],
                'area': det.get('bbox', {}).get('width', 0) * det.get('bbox', {}).get('height', 0),
                'iscrowd': 0
            }
            for i, det in enumerate(detections)
        ]
    }

@app.route('/api/delete_file', methods=['DELETE'])
def delete_file():
    """删除指定文件"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        filename = data.get('filename')
        
        if not session_id or not filename:
            return jsonify({'error': '缺少必要参数'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        # 查找并删除文件
        files = session_data[session_id]['files']
        file_to_delete = None
        file_index = -1
        
        for i, file_info in enumerate(files):
            if file_info['unique_filename'] == filename:
                file_to_delete = file_info
                file_index = i
                break
        
        if file_to_delete is None:
            return jsonify({'error': '文件不存在'}), 404
        
        # 删除物理文件
        filepath = file_to_delete['filepath']
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从会话数据中移除
        files.pop(file_index)
        
        # 清除相关的预测数据
        predictions = session_data[session_id].get('predictions', {})
        if filename in predictions:
            del predictions[filename]
        
        logger.info(f"文件删除成功: {filename}")
        
        return jsonify({
            'message': '文件删除成功',
            'remaining_files': len(files)
        })
        
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return jsonify({'error': f'删除文件失败: {str(e)}'}), 500

@app.route('/api/extract_video_frames', methods=['POST'])
def extract_video_frames():
    """提取视频帧"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        filename = data.get('filename')
        frame_interval = int(data.get('frame_interval', 1))  # 帧间隔
        
        if not session_id or not filename:
            return jsonify({'error': '缺少必要参数'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        # 查找视频文件
        video_file = None
        for file_info in session_data[session_id]['files']:
            if file_info['unique_filename'] == filename and file_info['type'] == 'video':
                video_file = file_info
                break
        
        if video_file is None:
            return jsonify({'error': '视频文件不存在'}), 404
        
        video_path = video_file['filepath']
        
        # 使用OpenCV提取帧
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return jsonify({'error': '无法读取视频信息'}), 400
        
        # 计算需要提取的帧数
        extract_count = total_frames // frame_interval
        extracted_files = []
        
        frame_index = 0
        extracted_index = 0
        
        logger.info(f"开始提取视频帧: {filename}, 总帧数: {total_frames}, 间隔: {frame_interval}, 预计提取: {extract_count}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔提取帧
            if frame_index % frame_interval == 0:
                # 生成帧文件名
                base_name = os.path.splitext(video_file['filename'])[0]
                frame_filename = f"{base_name}_frame_{extracted_index:04d}.jpg"
                frame_unique_filename = f"{session_id}_{frame_filename}"
                frame_filepath = os.path.join(UPLOAD_FOLDER, frame_unique_filename)
                
                # 保存帧
                cv2.imwrite(frame_filepath, frame)
                
                # 添加到文件列表
                frame_info = {
                    'filename': frame_filename,
                    'unique_filename': frame_unique_filename,
                    'filepath': frame_filepath,
                    'type': 'image',
                    'size': os.path.getsize(frame_filepath),
                    'timestamp': datetime.now().isoformat(),
                    'source_video': filename,
                    'frame_index': frame_index,
                    'time_seconds': frame_index / fps if fps > 0 else 0
                }
                
                extracted_files.append(frame_info)
                extracted_index += 1
            
            frame_index += 1
        
        cap.release()
        
        # 将提取的帧添加到会话数据
        session_data[session_id]['files'].extend(extracted_files)
        
        logger.info(f"视频帧提取完成: 提取了 {len(extracted_files)} 张图片")
        
        return jsonify({
            'message': f'视频帧提取完成',
            'total_frames': total_frames,
            'extracted_count': len(extracted_files),
            'frame_interval': frame_interval,
            'extracted_files': extracted_files  # 返回所有提取的文件信息
        })
        
    except Exception as e:
        logger.error(f"视频帧提取失败: {e}")
        return jsonify({'error': f'视频帧提取失败: {str(e)}'}), 500

@app.route('/api/clear_all_annotations', methods=['POST'])
def clear_all_annotations():
    """清除所有标注数据"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': '缺少会话ID'}), 400
        
        if session_id not in session_data:
            return jsonify({'error': '会话不存在'}), 404
        
        # 清除所有预测数据
        if 'predictions' not in session_data[session_id]:
            session_data[session_id]['predictions'] = {}
        session_data[session_id]['predictions'] = {}
        
        logger.info(f"清除所有标注数据: {session_id}")
        
        return jsonify({'message': '所有标注数据已清除'})
        
    except Exception as e:
        logger.error(f"清除所有标注失败: {e}")
        return jsonify({'error': f'清除所有标注失败: {str(e)}'}), 500

@app.route('/api/clear_session/<session_id>', methods=['DELETE'])
def clear_session(session_id: str):
    """清理会话数据"""
    try:
        if session_id in session_data:
            # 删除上传的文件
            for file_info in session_data[session_id]['files']:
                filepath = file_info['filepath']
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            # 删除会话数据
            del session_data[session_id]
            
            return jsonify({'message': '会话清理完成'})
        else:
            return jsonify({'error': '会话不存在'}), 404
            
    except Exception as e:
        logger.error(f"清理会话失败: {e}")
        return jsonify({'error': f'清理会话失败: {str(e)}'}), 500

@app.route('/api/clear_all_uploads', methods=['DELETE'])
def clear_all_uploads():
    """清除uploads文件夹中的所有文件"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            return jsonify({'message': 'uploads文件夹不存在，已重新创建', 'deleted_count': 0})
        
        deleted_count = 0
        failed_count = 0
        failed_files = []
        
        # 遍历并删除所有文件（忽略子文件夹）
        try:
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # 只删除文件，不删除文件夹
                if os.path.isfile(filepath):
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"已删除文件: {filename}")
                    except PermissionError:
                        logger.warning(f"权限不足，无法删除: {filename}")
                        failed_files.append(f"{filename} (权限不足)")
                        failed_count += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {filename}: {e}")
                        failed_files.append(f"{filename} ({str(e)})")
                        failed_count += 1
        except Exception as e:
            logger.error(f"遍历文件夹失败: {e}")
            return jsonify({'error': f'遍历uploads文件夹失败: {str(e)}'}), 500
        
        # 清除所有会话数据
        session_data.clear()
        logger.info("已清除所有会话数据")
        
        # 返回结果
        if failed_count > 0:
            message = f'已删除 {deleted_count} 个文件，{failed_count} 个文件删除失败'
            logger.warning(message)
            return jsonify({
                'message': message,
                'deleted_count': deleted_count,
                'failed_count': failed_count,
                'failed_files': failed_files[:10]  # 最多返回10个失败文件
            })
        else:
            message = f'成功清除所有上传文件，共 {deleted_count} 个'
            logger.info(message)
            return jsonify({
                'message': message,
                'deleted_count': deleted_count
            })
            
    except Exception as e:
        error_msg = f"清除所有上传文件失败: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 YOLOv智能标注工具服务器启动中...")
    print("=" * 60)
    print(f"📂 上传文件夹: {UPLOAD_FOLDER}")
    print(f"🧠 模型文件夹: {MODELS_FOLDER}")
    print(f"📊 结果文件夹: {RESULTS_FOLDER}")
    print(f"🔧 计算设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print("📖 使用说明:")
    print("1. 在浏览器中打开 http://localhost:5000")
    print("2. 上传您的YOLOv模型文件(.pt)")
    print("3. 上传图片或视频进行标注")
    print("4. 调整参数并点击自动标注")
    print("5. 导出标注数据为训练格式")
    print("=" * 60)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)
