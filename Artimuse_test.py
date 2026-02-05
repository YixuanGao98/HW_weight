import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image, ImageFile
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
import sys
import numpy as np
# 允许加载损坏或截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 导入自定义模型组件
sys.path.append("src")
sys.path.append("src/artimuse")
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

# 图像预处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)

def main(args):
    # 1. 路径配置（根据你之前的脚本整理）
    model_path = os.path.join("/home/wsw/model", args.model_name)
    true_dir = '/home/wsw/gyx/code_11.28/清晰度-splash_clarity'
    false_dir = '/home/wsw/gyx/code_11.28/正常-normal_data'
    
    # 2. 加载模型和分词器
    print(f"[INFO] Loading model from: {model_path}")
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    generation_config = dict(
        max_new_tokens=8192,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 3. 准备评估任务
    # 将两个文件夹及其对应的 Ground Truth 标签放入列表
    tasks = [
        (true_dir, True),   # 这里的图片标签应为 True (清晰)
        (false_dir, False)  # 这里的图片标签应为 False (模糊/普通)
    ]

    tp = fp = tn = fn = 0
    results = []

    # 4. 开始遍历文件夹并测试
    for folder_path, true_label in tasks:
        if not os.path.exists(folder_path):
            print(f"[WARN] Folder not found: {folder_path}")
            continue
            
        print(f"\n[INFO] Evaluating folder: {folder_path} (GT: {true_label})")
        
        # 获取文件夹内所有图片名
        img_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        for filename in tqdm(img_list, desc="Processing"):
            image_path = os.path.join(folder_path, filename)

            
            try:
                # 推理
                pixel_values = load_image(image_path).to(torch.bfloat16).to(args.device)
                score = model.score(args.device, tokenizer, pixel_values, generation_config)
                
                # 判定逻辑 (根据你原代码：score < 54.5 为 True)
                predicted_label = True if score < np.float(args.threshold) else False

                # 统计 Precision / Recall
                if predicted_label == True and true_label == True:
                    tp += 1
                elif predicted_label == True and true_label == False:
                    fp += 1
                elif predicted_label == False and true_label == False:
                    tn += 1
                elif predicted_label == False and true_label == True:
                    fn += 1
                
                # 记录结果（可选）
                results.append({
                    "image": filename,
                    "score": float(score),
                    "predicted": predicted_label,
                    "gt": true_label
                })
                
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")

    # 5. 计算最终指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "="*30)
    print(f"Evaluation Complete")
    print(f"Total processed: {len(results)}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ArtMuse_AVA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=str, default="53")
    args = parser.parse_args()

    main(args)
  
