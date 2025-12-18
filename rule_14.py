from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient
import cv2
import statistics
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/wsw/gyx/code_12.17/MinerU2.5-2509-1.2B",
    dtype="auto", # use `torch_dtype` instead of `dtype` for transformers<4.56.0
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/home/wsw/gyx/code_12.17/MinerU2.5-2509-1.2B",
    use_fast=True
)

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)
import torch
import os
import cv2
import numpy as np
import csv
from skimage.measure import shannon_entropy
import numpy as np

    
def detect_modules(image_path):
    # image_path='/home/wsw/gyx/code_11.28/test_data/排布间距/2025091704124681414A4C7CF24648B3109557C45D065B.jpg'
    cv_image = cv2.imread(image_path)  # 直接使用 OpenCV 读取
    image = Image.open(image_path)
    extracted_blocks = client.two_step_extract(image)
    print(extracted_blocks)

    text_boxes = []
    img_with_boxes = cv_image.copy()
    for i, block in enumerate(extracted_blocks):
        try:
            # 获取边界框坐标
            bbox = block.bbox
            # print(f"  模块 {i} 原始bbox: {bbox}")
            
            # 确保 bbox 是 [x_min, y_min, x_max, y_max] 格式
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                
                # 确保坐标在合理范围内
                height, width = img_with_boxes.shape[:2]

                # --- 修改开始 ---
                # 判断是否为归一化坐标 (假设如果坐标都小于等于1，则是归一化坐标)
                if all(x <= 1.0 for x in [x_min, y_min, x_max, y_max]):
                    # print(f"  检测到归一化坐标，正在转换...")
                    x_min = int(x_min * width)
                    x_max = int(x_max * width)
                    y_min = int(y_min * height)
                    y_max = int(y_max * height)
                else:
                    # 已经是像素坐标，直接取整
                    x_min = int(x_min)
                    y_min = int(y_min)
                    x_max = int(x_max)
                    y_max = int(y_max)

                # # 限制坐标在图像范围内 (防止越界)
                # x_min = max(0, min(x_min, width - 1))
                # y_min = max(0, min(y_min, height - 1))
                # x_max = max(0, min(x_max, width - 1))
                # y_max = max(0, min(y_max, height - 1))
                # --- 修改结束 ---
                text_boxes.append([x_min, y_min, x_max, y_max])
                                                
                # # 绘制边界框
                # color = (0, 255, 0)  # 绿色 (BGR格式)
                # thickness = 2
                # cv2.rectangle(img_with_boxes, 
                #             (x_min, y_min), 
                #             (x_max, y_max), 
                #             color, thickness)
                
                # # 添加模块编号标签
                # label = f"Module {i}"
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.5
                # label_thickness = 2
                
                # # 计算文本大小
                # (text_width, text_height), baseline = cv2.getTextSize(
                #     label, font, font_scale, label_thickness
                # )
                
                # # 绘制文本背景
                # cv2.rectangle(img_with_boxes,
                #             (x_min, y_min - text_height - 10),
                #             (x_min + text_width, y_min),
                #             color, -1)  # -1 表示填充
                
                # # 绘制文本
                # cv2.putText(img_with_boxes, label,
                #             (x_min, y_min - 5),
                #             font, font_scale,
                #             (0, 0, 0),  # 黑色文本
                #             label_thickness)
                
                # # print(f"  模块 {i} 边界框已绘制")
            else:
                print(f"  模块 {i} 的bbox格式不正确: {bbox}")
                
        except Exception as e:
            print(f"  绘制模块 {i} 时出错: {e}")
            continue


    # 汇总所有模块
    modules = []  
    for i, b in enumerate(text_boxes):
        if extracted_blocks[i].type=='image':
            modules.append({'type': 'image', 'box': b})
        elif extracted_blocks[i].type!='list':
            modules.append({'type': 'text', 'box': b})

    return modules,height, width,cv_image

if __name__ == "__main__":

    folders = ["/home/wsw/gyx/code_11.28/test_data/排布间距"]
    # folders = ["/home/wsw/gyx/code_11.28/test_data/正常-normal_data"]
    
    image_files = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
    
    image_complexities = []
    N=0
    for img_path in image_files:
        # img_path='/home/wsw/gyx/code_11.28/test_data/排布间距/dsp_url_20251124134001FF36D9206A0B475098B2655616CD4880.jpg'
        print(f"处理图像: {img_path}")
        # 1. 检测广告模块
        modules,H, W, img = detect_modules(img_path)

        # if calculate_features(modules, H, W,img):
        #     N=N+1
        # print(N)


