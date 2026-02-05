import os
import json
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

import sys
sys.path.append("src")
sys.path.append("src/artimuse")
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

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


def get_model_params(model):
    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(args):
    # Setup paths
    model_path = os.path.join("/home/wsw/model", args.model_name)
    results_dir = os.path.join("results", "dataset_results")
    # input_json_path = f"/home/wsw/gyx/code_11.28/hw_test.json"
    # test_image_path = f"/home/wsw/gyx/code_1.15_2/数量满足要求"
    input_json_path = f"/home/wsw/gyx/code_11.28/ArtiMuse-master/hw_test_清晰.json"
    test_image_path = f"/home/wsw/gyx/code_11.28"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{args.dataset}_{args.model_name}.json")

    # Load model & tokenizer
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

    # Calculate model parameters
    total_params = get_model_params(model)
    print(f"Total number of parameters in the model: {total_params / 1e6:.2f}M")

    with open(input_json_path, 'r') as f:
        test_data = json.load(f)

    results = []
    tp = fp = tn = fn = 0  # Initialize counts for precision and recall

    for item in tqdm(test_data, desc=f"Evaluating {args.dataset}"):
        print(item)
        image_name = item['image']
        true_label = item['gt_score']  # True label (True or False)
        image_path = os.path.join(test_image_path, image_name)

        if not os.path.exists(image_path):
            print(f"{image_path} not exists.")
            continue

        pixel_values = load_image(image_path).to(torch.bfloat16).to(args.device)

        score = model.score(args.device, tokenizer, pixel_values, generation_config)
        print(score)
        
        # Prediction based on score threshold
        predicted_label = True if score < 54.5 else False #精美都53

        # Calculate precision and recall
        if predicted_label == True and true_label == True:
            tp += 1  # True Positive
        elif predicted_label == True and true_label == False:
            fp += 1  # False Positive
        elif predicted_label == False and true_label == False:
            tn += 1  # True Negative
        elif predicted_label == False and true_label == True:
            fn += 1  # False Negative
            
        results.append({
            "image": image_name,
            "score": score,
            "predicted_label": predicted_label,
            "true_label": true_label
        })
        
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ArtMuse_AVA", help="Name of the model (must exist in checkpoints/)")
    parser.add_argument("--dataset", type=str, default="AVA", help="AVA | TAD66K | PARA | FLICKR-AES")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device number")
    args = parser.parse_args()

    main(args)
# Precision: 0.9474
# Recall: 0.9474
