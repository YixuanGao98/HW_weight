  {
    "image": "/home/wsw/gyx/code_11.28/test_data/正常-normal_data/dsp_url_20250819140833284B5521FA3143868FE411B113CA155A.jpg",
    "response": "这张广告图的排版布局**不合理**。\n\n理由如下：\n\n1.  **字与标签logo之间离得太近**：图中“赔付限额 600万元”这个标签（蓝色圆角矩形框）与上方的“责任范围内大病小病都能保”等宣传语文字在视觉上距离过近。宣传语文字位于图片上半部分，而标签位于图片下半部分，中间隔着一位女性模特。虽然模特是图像主体，但文字与标签之间没有足够的留白或视觉分隔，导致视觉上显得拥挤、层次不清，缺乏呼吸感。\n\n2.  **字与商品之间离得太近**：这里的“商品”指代的是图中女性模特所代表的“保险产品”或“保障服务”。宣传语文字（如“责任范围内大病小病都能保”）直接位于模特头部上方，而“赔付限额 600万元”标签位于模特身体下方。文字与模特形象（作为产品代言人）之间没有足够的空间缓冲，容易让观者感觉文字“贴”在模特身上，显得不够专业和疏离，也削弱了广告的视觉美感和信息层级。\n\n综上，由于存在“字与"
  },
  {
    "image": "/home/wsw/gyx/code_11.28/test_data/正常-normal_data/dsp_url_2025082116555265CE45116D6D4CFEBB84379BDFF6E765.jpg",
    "response": "1. 字与标签logo之间：**没有离得太近**。  \n   图中“TASAKI”品牌文字位于图片底部中央，与上方的模特图像之间有明显的留白空间，视觉上分离清晰，符合广告设计中品牌标识与主体内容的常规布局，不会产生拥挤或压迫感。\n\n2. 字与商品之间：**没有离得太近**。  \n   “TASAKI”文字位于模特图像正下方，与模特佩戴的项链、耳环等商品之间隔着模特的肩部和服装，距离适中。这种布局是典型的“品牌标识置于产品下方”的广告排版，既突出商品，又保留品牌辨识度，不会造成视觉干扰。\n\n✅ **最终判断：这张广告图的排版布局合理。**\n\n理由：  \n- 品牌文字与图像内容（模特+商品）之间有充足留白，层次分明。  \n- 文字位置居中且处于视觉重心下方，符合广告视觉动线（从产品→品牌），不抢夺主体，也不显得孤立。  \n- 整体构图简洁、专业，符合高端珠宝品牌的调性。\n\n因此，该广告图排版布局合理，无明显"
  },
  {
    "image": "/home/wsw/gyx/code_11.28/test_data/正常-normal_data/dsp_url_2025081511090977BACB55DB734F1EBD9D271FD20D246B.jpg",
    "response": "这张广告图的排版布局**不合理**。\n\n理由如下：\n\n1.  **字与标签logo之间离得太近**：图中“芒果tv”的logo位于抽油烟机下方，而“花儿与少年 同心季”的节目标题文字则叠加在人物胸前和锅具上方。这两个元素在视觉上距离非常近，几乎重叠在同一个垂直空间内。logo和文字（尤其是“花儿与少年”这几个大字）在构图上没有足够的留白或视觉分隔，显得拥挤，破坏了画面的层次感和专业感。\n\n2.  **字与商品之间离得太近**：节目标题文字“花儿与少年 同心季”直接覆盖在人物手中的盘子（商品/道具）和锅具之上，文字与商品（盘子、番茄、锅）有直接的重叠。这种设计不仅遮挡了商品的视觉呈现，也使文字与商品在视觉上“打架”，不符合广告图中“突出商品/场景”的常见排版原则。\n\n综上，由于**字与标签logo之间离得太近**，且**字与商品之间离得太近**，这两个问题同时存在，因此可以判定这张"
  },
  {
    "image": "/home/wsw/gyx/code_11.28/test_data/正常-normal_data/20250820112913E9C27F4B1CB8491FBAA72D9B5C4E7644.jpg",
    "response": "1. **字与标签logo之间是否离得太近？**  \n   - 图中“支”字（支付宝的“支”）位于顶部，是品牌标识，下方是宣传语“动动手指上支付宝”。两者之间有明显的留白和视觉间距，符合广告设计中品牌标识与正文的常规布局，**不近**。\n\n2. **字与商品之间是否离得太近？**  \n   - 宣传语“动动手指上支付宝”和“天天必得20元外卖红包”位于画面中上部，而商品（炸鸡、薯条、鸡尾酒）位于画面中下部。两者之间有充足的视觉空间，文字没有压在商品上，也没有遮挡商品，**不近**。\n\n✅ **最终判断：这张广告图的排版布局合理。**\n\n理由：  \n- 品牌标识（“支”）与正文之间留白充足，符合视觉层级。  \n- 正文与商品之间有明确的视觉分隔，商品主体突出，文字不干扰商品展示。  \n- 整体布局层次清晰，符合广告设计的“品牌—信息—产品”三段式结构，引导用户视线从"
  },
 
https://www.paddlepaddle.org.cn/packages/stable/cu126/
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
