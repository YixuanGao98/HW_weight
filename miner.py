

import io
import asyncio
import aiofiles
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from PIL import Image
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
import os
import cv2
import numpy as np
import json

# 设置GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

class MinerUProcessor:
    def __init__(self, model_path="opendatalab/MinerU2.5-2509-1.2B"):
        """初始化MinerU处理器"""
        self.model_path = model_path
        self.async_llm = None
        self.client = None
        
    async def initialize(self):
        """初始化AsyncLLM和MinerUClient"""
        print("正在初始化AsyncLLM...")
        # 创建async_llm
        self.async_llm = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=self.model_path,
                logits_processors=[MinerULogitsProcessor]
            )
        )
        
        print("正在初始化MinerUClient...")
        # 创建client
        self.client = MinerUClient(
            backend="vllm-async-engine",
            vllm_async_llm=self.async_llm,
        )
        print("初始化完成!")
        
    async def load_image(self, image_path):
        """异步加载图片，返回 PIL Image 和 OpenCV 格式"""
        print(f"正在加载图片: {image_path}")
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
        
        # 使用BytesIO创建PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 转换为 OpenCV 格式 (BGR)
        cv_image = cv2.imread(image_path)  # 直接使用 OpenCV 读取
        
        if cv_image is None:
            # 如果 OpenCV 读取失败，从 PIL 转换
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        print(f"图片加载完成，PIL尺寸: {pil_image.size}, OpenCV形状: {cv_image.shape}")
        return pil_image, cv_image
    
    async def extract_blocks(self, image_path):
        """完整的提取流程：加载图片 -> 提取块"""
        try:
            # 1. 确保已初始化
            if self.client is None:
                await self.initialize()
            
            # 2. 加载图片
            pil_image, cv_image = await self.load_image(image_path)
            
            # 3. 提取块
            print("正在提取块...")
            extracted_blocks = await self.client.aio_two_step_extract(pil_image)
            
            print("提取完成!")
            return extracted_blocks, cv_image
            
        except Exception as e:
            print(f"提取过程中出现错误: {e}")
            raise
    
    async def process_multiple_images(self, image_paths):
        """处理多张图片"""
        results = {}
        
        # 先初始化模型
        await self.initialize()
        
        for image_path in image_paths:
            try:
                print(f"\n{'='*50}")
                print(f"处理图片: {image_path}")
                
                # 加载图片
                pil_image, cv_image = await self.load_image(image_path)
                
                # 提取块
                extracted_blocks = await self.client.aio_two_step_extract(pil_image)
                
                # 调试：打印提取结果
                print(f"提取到 {len(extracted_blocks)} 个模块")
                
                # if len(extracted_blocks) > 0:
                #     print("\n模块详细信息:")
                #     for i, block in enumerate(extracted_blocks):
                #         print(f"模块 {i}:")
                #         print(f"  BBox: {block.bbox}")
                #         print(f"  类型: {block.type}")
                #         # print(f"  内容: {block.content[:50]}...")  # 只显示前50个字符
                
                results[image_path] = extracted_blocks
                
                # 在图像上绘制边界框
                img_with_boxes = cv_image.copy()
                
                if len(extracted_blocks) > 0:
                    print("\n正在绘制边界框...")
                    for i, block in enumerate(extracted_blocks):
                        try:
                            # 获取边界框坐标
                            bbox = block.bbox
                            print(f"  模块 {i} 原始bbox: {bbox}")
                            
                            # 确保 bbox 是 [x_min, y_min, x_max, y_max] 格式
                            if len(bbox) == 4:
                                x_min, y_min, x_max, y_max = bbox
                                
# 确保坐标在合理范围内
height, width = img_with_boxes.shape[:2]

# --- 修改开始 ---
# 判断是否为归一化坐标 (假设如果坐标都小于等于1，则是归一化坐标)
if all(x <= 1.0 for x in [x_min, y_min, x_max, y_max]):
    print(f"  检测到归一化坐标，正在转换...")
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

# 限制坐标在图像范围内 (防止越界)
x_min = max(0, min(x_min, width - 1))
y_min = max(0, min(y_min, height - 1))
x_max = max(0, min(x_max, width - 1))
y_max = max(0, min(y_max, height - 1))
# --- 修改结束 ---

print(f"  模块 {i} 修正后bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
                                
                                # 绘制边界框
                                color = (0, 255, 0)  # 绿色 (BGR格式)
                                thickness = 2
                                cv2.rectangle(img_with_boxes, 
                                            (x_min, y_min), 
                                            (x_max, y_max), 
                                            color, thickness)
                                
                                # 添加模块编号标签
                                label = f"Module {i}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.5
                                label_thickness = 2
                                
                                # 计算文本大小
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label, font, font_scale, label_thickness
                                )
                                
                                # 绘制文本背景
                                cv2.rectangle(img_with_boxes,
                                            (x_min, y_min - text_height - 10),
                                            (x_min + text_width, y_min),
                                            color, -1)  # -1 表示填充
                                
                                # 绘制文本
                                cv2.putText(img_with_boxes, label,
                                          (x_min, y_min - 5),
                                          font, font_scale,
                                          (0, 0, 0),  # 黑色文本
                                          label_thickness)
                                
                                print(f"  模块 {i} 边界框已绘制")
                            else:
                                print(f"  模块 {i} 的bbox格式不正确: {bbox}")
                                
                        except Exception as e:
                            print(f"  绘制模块 {i} 时出错: {e}")
                            continue
                else:
                    print("警告: 未提取到任何模块!")
                    # 在没有模块时，添加一个提示文本
                    cv2.putText(img_with_boxes, "No modules detected",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 0, 255), 2)
                
                # 保存带框图像
                output_dir = "/home/wsw/gyx/code_12.17/layout_paibu"
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成输出文件名
                base_name = os.path.basename(image_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}_annotated.jpg")
                
                # 保存图像
                cv2.imwrite(output_path, img_with_boxes)
                print(f"保存带框图像到: {output_path}")
                
                # 同时保存JSON结果
                json_path = os.path.join(output_dir, f"{name_without_ext}_results.json")
                json_results = []
                for block in extracted_blocks:
                    json_results.append({
                        "bbox": block.bbox,
                        "type": block.type,
                        "content": block.content[:500]  # 限制内容长度
                    })
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, ensure_ascii=False, indent=2)
                print(f"保存结果到: {json_path}")
                
                print(f"图片 {image_path} 处理完成")
                
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                results[image_path] = None
        
        return results
    
    async def shutdown(self):
        """关闭资源"""
        if self.async_llm is not None:
            print("正在关闭AsyncLLM...")
            self.async_llm.shutdown()
            print("资源已释放")

def test_single_image():
    """测试单张图片的边界框绘制"""
    test_image_path = "/home/wsw/gyx/code_11.28/test_data/排布间距/20241212112442EECEB9747D62434FBDC1F1CA71BE7829 (1).jpg"
    
    # 简单测试：先不用异步，直接检查图像
    print("测试单张图片...")
    
    # 读取图像
    cv_image = cv2.imread(test_image_path)
    if cv_image is None:
        print("错误: 无法读取图像")
        return
    
    print(f"图像尺寸: {cv_image.shape}")
    
    # 手动添加一个测试边界框（确认OpenCV工作正常）
    height, width = cv_image.shape[:2]
    test_box = [100, 100, 300, 200]  # [x_min, y_min, x_max, y_max]
    
    # 绘制测试框
    x_min, y_min, x_max, y_max = test_box
    cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # 红色框
    
    # 保存测试图像
    test_output = "/home/wsw/gyx/code_12.17/layout_paibu/test_box.jpg"
    cv2.imwrite(test_output, cv_image)
    print(f"测试图像已保存到: {test_output}")
    print("请检查这个测试图像是否包含红色边界框")
    
    # 如果测试图像正常，说明OpenCV工作正常
    # 问题可能在提取的blocks为空或坐标格式不对

async def debug_extraction(image_path):
    """调试提取过程"""
    processor = MinerUProcessor(model_path="opendatalab/MinerU2.5-2509-1.2B")
    
    try:
        await processor.initialize()
        
        # 加载图片
        pil_image, cv_image = await processor.load_image(image_path)
        
        # 提取块
        print("正在提取块...")
        extracted_blocks = await processor.client.aio_two_step_extract(pil_image)
        
        print(f"\n提取结果分析:")
        print(f"提取到 {len(extracted_blocks)} 个模块")
        
        if len(extracted_blocks) == 0:
            print("警告: 没有提取到任何模块!")
            print("可能原因:")
            print("1. 图像中没有可识别的模块")
            print("2. 模型参数可能需要调整")
            print("3. 图像质量或格式问题")
        else:
            # 详细打印每个模块的信息
            for i, block in enumerate(extracted_blocks):
                print(f"\n模块 {i}:")
                print(f"  BBox: {block.bbox}")
                print(f"  Type: {block.type}")
                # print(f"  Content (前100字符): {block.content[:100]}")
                
                # 检查BBox格式
                if hasattr(block.bbox, '__len__'):
                    print(f"  BBox长度: {len(block.bbox)}")
                    if len(block.bbox) >= 4:
                        x_min, y_min, x_max, y_max = block.bbox[:4]
                        print(f"  坐标范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
                        
                        # 检查坐标是否合理
                        height, width = cv_image.shape[:2]
                        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                            print(f"  警告: 坐标超出图像范围!")
                        if x_min >= x_max or y_min >= y_max:
                            print(f"  警告: 坐标值无效 (x_min >= x_max 或 y_min >= y_max)!")
        
        return extracted_blocks
        
    finally:
        await processor.shutdown()

async def main_batch(image_paths):
    """批量处理图片的示例"""
    processor = MinerUProcessor(model_path="opendatalab/MinerU2.5-2509-1.2B")
    
    try:
        print("批量处理图片...")
        results = await processor.process_multiple_images(image_paths)
        
        print("\n" + "="*50)
        print("处理结果汇总:")
        for path, blocks in results.items():
            print(f"\n图片: {path}")
            print(f"提取到 {len(blocks) if blocks else 0} 个模块")
            
    finally:
        await processor.shutdown()

if __name__ == "__main__":
    # # 第一步：测试OpenCV是否能正常绘制边界框
    # print("第一步：测试OpenCV功能...")
    # test_single_image()
    
    # # 第二步：调试提取过程
    # print("\n" + "="*50)
    # print("第二步：调试提取过程...")
    # test_image = "/home/wsw/gyx/code_11.28/test_data/排布间距/20241212112442EECEB9747D62434FBDC1F1CA71BE7829 (1).jpg"
    # asyncio.run(debug_extraction(test_image))
    
    # 第三步：批量处理
    print("\n" + "="*50)
    print("第三步：批量处理所有图片...")
    
    folders = ["/home/wsw/gyx/code_11.28/test_data/排布间距"]
    image_files = []
    
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图像文件
                    image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 先处理几张测试
    if len(image_files) > 0:
        # 先处理前3张看看效果
        test_files = image_files[:3]
        print(f"先处理前 {len(test_files)} 张图片作为测试...")
        asyncio.run(main_batch(test_files))
    else:
        print("没有找到图片文件!")

模块 3 原始bbox: [0.301, 0.346, 0.686, 0.389]
  模块 3 修正后bbox: [0, 0, 0, 0]
