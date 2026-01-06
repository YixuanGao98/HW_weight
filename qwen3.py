# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# =========================
# 全局设置
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = "model_hub/Qwen3-VL-235B-A22B-Instruct"
DATA_JSON = "train_instruction_data.json"
OUT_DIR = "./qwen_outputs_vllm"
FINAL_FILENAME = "predictions_ad_violation_check.json"

TENSOR_PARALLEL_SIZE = 8 
CHUNK_SIZE = 256  

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 提示词模板 (更新维度)
# =========================

def build_user_text_for_exquisiteness() -> str:
    """精美度检测 Prompt"""
    return (
        "Evaluate the image for 'Exquisiteness' (精美度).\n"
        "If the design looks like a low-effort template, lacks visual depth, or feels disconnected, you MUST return \"is_violation\": true.\n\n"
        "**STRICT VIOLATION CRITERIA:**\n"
        "1. Simplistic & 'Flat' Design: Overly basic layout, generic icons, no professional lighting/shadows.\n"
        "2. Poor Overall Aesthetics: Muddy colors, unbalanced composition, pixelated/unrefined quality.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a strictly valid JSON object:\n"
        "{\"is_violation\": true, \"reason\": \"...\"}"
    )

def build_user_text_for_spacing() -> str:
    """间距检测 Prompt"""
    return (
        "Analyze the image for 'Composition & Spacing' standards.\n"
        "If the elements feel 'squeezed' or 'crowded,' you MUST return \"is_violation\": true.\n\n"
        "**STRICT VIOLATION CRITERIA:**\n"
        "1. Lack of Breathing Room: Subject, headline, and logo are too close.\n"
        "2. Edge Tension: Elements touching/tangent to borders.\n"
        "3. Information Overload: Too many blocks with no clear visual path.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a strictly valid JSON object:\n"
        "{\"is_violation\": true, \"reason\": \"...\"}"
    )

def build_user_text_for_position() -> str:
    """位置/易读性检测 Prompt"""
    return (
        "Evaluate 'Information Accessibility'. Ensure copy is instantly readable and strategically placed.\n"
        "If the text is buried, hard to read, or poorly positioned, you MUST return \"is_violation\": true.\n\n"
        "**STRICT VIOLATION CRITERIA:**\n"
        "1. Overlay on Complex Background: Text over busy textures without professional treatment.\n"
        "2. Poor Visual Catchiness: Message in 'visual dead zones'.\n"
        "3. Low Contrast: Text color similar to background.\n"
        "4. Unclear Small Text: Footnotes difficult to read.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a strictly valid JSON object:\n"
        "{\"is_violation\": true, \"reason\": \"...\"}"
    )

def build_user_text_for_post_production() -> str:
    """后期处理检测 Prompt"""
    return (
        "Analyze the image for 'Post-Production Quality'. Identify raw, unprocessed amateur snapshots.\n"
        "If the image looks like a 'casual snapshot' without professional polishing, you MUST return \"is_violation\": true.\n\n"
        "**STRICT VIOLATION CRITERIA:**\n"
        "1. Lack of Professional Post-Processing: Raw Photo appearance, no optimization of lighting, color, or depth of field.\n"
        "2. Amateur Snapshot Aesthetic: Lacks high-end texture and artistic polish. Feels 'Cheap'.\n"
        "3. Absence of Value Conveyance: Fails to evoke a sense of high quality or luxury.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a strictly valid JSON object:\n"
        "{\"is_violation\": true, \"reason\": \"...\"}"
    )

def build_user_text_for_realism() -> str:
    """真实感检测 Prompt"""
    return (
        "Analyze the image for 'Picture-Realism and Compositional Integrity'.\n"
        "If the subject and background feel disconnected, amateurish, or poorly composited, you MUST return \"is_violation\": true.\n\n"
        "**STRICT VIOLATION CRITERIA:**\n"
        "1. Obvious 'Photoshopped' Traces: Sharp, unnatural cutout edges or visible 'fringing'.\n"
        "2. Unnatural Lighting & Shadows: Light direction conflict or missing contact/cast shadows (floating effect).\n"
        "3. Illogical Scene Composition: 'Sticker Effect' where the subject looks arbitrarily pasted.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a strictly valid JSON object:\n"
        "{\"is_violation\": true, \"reason\": \"...\"}"
    )

SYSTEM_PROMPT = (
    "You are a highly critical Senior Art Director and Visual Auditor. "
    "Your goal is to identify low-quality, amateur, or poorly designed advertising materials. "
    "Strictly follow instructions and output ONLY a valid JSON object with 'is_violation' and 'reason' keys. "
    "Do NOT output any preambles or extra text."
)

# =========================
# 数据加载与主逻辑
# =========================

def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = [item for item in data if "file_path" in item and "question_type" in item and os.path.exists(item["file_path"])]
    return cleaned

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ds = load_dataset(DATA_JSON)

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[151645, 151643]
    )

    final_output_path = os.path.join(OUT_DIR, FINAL_FILENAME)
    processed_results = []
    finished_ids = set()

    if os.path.exists(final_output_path):
        try:
            with open(final_output_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            for item in loaded_data:
                curr_id = item.get("id") if item.get("id") is not None else item.get("path")
                finished_ids.add(curr_id)
                processed_results.append(item)
        except: pass

    ds_to_process = [item for item in ds if (item.get("id") if item.get("id") is not None else item.get("file_path")) not in finished_ids]
    if not ds_to_process: return

    chunks = [ds_to_process[i : i + CHUNK_SIZE] for i in range(0, len(ds_to_process), CHUNK_SIZE)]

    for chunk in tqdm(chunks, desc="Audit Inference"):
        prompts = []
        valid_chunk_items = []
        for item in chunk:
            qtype = item["question_type"]
            img_path = item["file_path"]
            
            # 扩展的分支逻辑
            if qtype == "exquisiteness":
                user_text = build_user_text_for_exquisiteness()
            elif qtype == "spacing":
                user_text = build_user_text_for_spacing()
            elif qtype == "position":
                user_text = build_user_text_for_position()
            elif qtype == "post_production": # 新增：后期处理
                user_text = build_user_text_for_post_production()
            elif qtype == "realism":         # 新增：真实感
                user_text = build_user_text_for_realism()
            else:
                continue

            try:
                image_obj = Image.open(img_path).convert("RGB")
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": user_text}
                    ]}
                ]
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append({"prompt": text_prompt, "multi_modal_data": {"image": image_obj}})
                valid_chunk_items.append(item)
            except: continue

        if not prompts: continue
        outputs = llm.generate(prompts, sampling_params)

        for original_item, output_obj in zip(valid_chunk_items, outputs):
            generated_text = output_obj.outputs[0].text.strip()
            if generated_text.startswith("```json"):
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()
            
            processed_results.append({
                "id": original_item.get("id"),
                "path": original_item.get("file_path"),
                "question_type": original_item.get("question_type"),
                "model_output": generated_text
            })

        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
