import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from tqdm import tqdm

# ===============================
# 1. 模型与 Processor 加载
# ===============================
model_path = "/home/wsw/jikaiyuan/stage2/code/code_2025_12_17/Qwen3-VL-8B-Instruct"

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_path)

# ===============================
# 2. 输入文件夹
# ===============================
image_dir = "/home/wsw/gyx/code_11.28/test_data/排布间距"
assert os.path.isdir(image_dir)

image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"共发现 {len(image_paths)} 张图片")

# ===============================
# 3. 固定问题模板
# ===============================
question_text = (
    "1. 请判断这张广告里字与标签logo之间是否离得太近。"
    "2. 请判断这张广告里字与商品之间是否离得太近。"
    "如果有一个是，则认为这张广告图的排版布局不合理。"
    "请最终判断这张图的排版布局是否合理。"
)

# ===============================
# 4. 批量推理
# ===============================
results = []

for img_path in tqdm(image_paths, desc="Processing images"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"\n[{os.path.basename(img_path)}]")
    print(output_text)

    results.append({
        "image": img_path,
        "response": output_text
    })

# ===============================
# 5. 保存结果（强烈推荐）
# ===============================
import json

save_path = os.path.join(image_dir, "layout_reasoning_results.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n推理完成，结果已保存至: {save_path}")



import math
# ... 保持之前的 import 不变 ...

def calculate_box_distance(box1, box2):
    """
    计算两个矩形框 [x_min, y_min, x_max, y_max] 之间的最短欧式距离
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算水平方向的最短距离 (dx)
    # 如果 box1 在 box2 左侧，距离为 x2_min - x1_max
    # 如果 box1 在 box2 右侧，距离为 x1_min - x2_max
    # 如果重叠，dx = 0
    dx = max(0, x1_min - x2_max, x2_min - x1_max)
    
    # 计算垂直方向的最短距离 (dy)
    dy = max(0, y1_min - y2_max, y2_min - y1_max)

    # 返回欧式距离
    return math.sqrt(dx**2 + dy**2)

def detect_modules(image_path):
    # ... 此部分逻辑保持不变 ...
    cv_image = cv2.imread(image_path)
    image = Image.open(image_path)
    extracted_blocks = client.two_step_extract(image)
    
    text_boxes = []
    height, width = cv_image.shape[:2]

    for i, block in enumerate(extracted_blocks):
        try:
            bbox = block.bbox
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                # 坐标转换逻辑
                if all(x <= 1.0 for x in [x_min, y_min, x_max, y_max]):
                    x_min, x_max = int(x_min * width), int(x_max * width)
                    y_min, y_max = int(y_min * height), int(y_max * height)
                else:
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                
                text_boxes.append([x_min, y_min, x_max, y_max])
        except Exception as e:
            continue

    modules = []  
    for i, b in enumerate(text_boxes):
        if extracted_blocks[i].type == 'image':
            modules.append({'id': i, 'type': 'image', 'box': b})
        elif extracted_blocks[i].type != 'list':
            modules.append({'id': i, 'type': 'text', 'box': b})

    return modules, height, width, cv_image

if __name__ == "__main__":
    # ... 文件夹遍历逻辑保持不变 ...
    folders = ["/home/wsw/gyx/code_11.28/test_data/排布间距"]
    image_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

    for img_path in image_files:
        print(f"\n{'='*30}\n处理图像: {img_path}")
        
        # 1. 检测模块
        modules, H, W, img = detect_modules(img_path)
        print(f"检测到 {len(modules)} 个模块")

        # 2. 计算模块间距离
        # 使用两两组合 (Pairwise) 计算
        module_distances = []
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                m1 = modules[i]
                m2 = modules[j]
                
                dist = calculate_box_distance(m1['box'], m2['box'])
                
                # 记录结果
                dist_info = {
                    'pair': (m1['id'], m2['id']),
                    'types': (m1['type'], m2['type']),
                    'distance': round(dist, 2)
                }
                module_distances.append(dist_info)
                
                # 打印距离较近的模块（可选，例如距离小于100像素的）
                print(f"模块 {m1['id']}({m1['type']}) 与 模块 {m2['id']}({m2['type']}) 的距离: {dist_info['distance']} px")

        # 这里可以继续你的后续逻辑，比如 image_complexities.append(...)
