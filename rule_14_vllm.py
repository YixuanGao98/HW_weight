# -*- coding: utf-8 -*-
"""
VLLM Design Rule Auditor (Zero-shot)
功能：批量扫描文件夹内的图片，判断是否违反"文字-设计搭配"规则。
输出：终端统计报告 + 详细 JSON 结果文件
python rule_14_vllm.py \
  --input_dir "project_huawei/stage2/文字-设计搭配协调" \
  --model_path "/path/to/your/Qwen2.5-VL-7B-Instruct"
"""

import os
import re
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any

# ==========================================
# 【核心配置】环境与并发设置
# ==========================================
# 必须在导入 vllm 前设置 spawn，防止 CUDA 初始化死锁
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 如果需要指定显卡，请取消下面注释并修改 ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# 防止部分图片因截断而报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的图片格式
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ==========================================
# 【提示词工程】针对 "文字-设计搭配协调"
# ==========================================
# 将中文规则翻译为模型易懂的英文指令，并强制 JSON 输出
SYS_PROMPT_TEXT = """
You are a highly critical Senior Art Director. Your job is to flag "Low-Quality / Amateur" advertising designs.
You have ZERO TOLERANCE for "Cheap Ad Styles" (often called "Niu Pi Xian" in Chinese context).

**YOUR TASK:**
Determine if the image violates the "Text-Design Harmony" standards. 
If the design looks cheap, messy, or outdated, you MUST return "is_violation": true.

**STRICT VIOLATION CRITERIA (If ANY are found, return TRUE):**

1. **The "WordArt" Effect (廉价特效):**
   - Text uses heavy, amateurish strokes (thick white/colored outlines) that look jagged or pixelated.
   - Text has aggressive "Outer Glow" (neon glow) that makes it look blurry.
   - Text uses outdated "Pseudo-3D" gradients (e.g., shiny gold/silver metal texture) that clash with a flat background.
   - Text is stretched, squeezed, or distorted strictly to fit a space.

2. **Visual Clutter & Conflict (背景冲突):**
   - Text is placed directly on top of a "Busy Photograph" (leaves, city streets, crowds) without a sufficient background mask, making it hard to read.
   - The color of the text vibrates against the background (e.g., bright red text on bright green background).
   - "Patchwork Style": The text background looks like a sticker arbitrarily pasted onto a photo, completely ignoring the photo's lighting and perspective.


3. **Inconsistent Aesthetic (风格割裂):**
   - Foreground is a cartoon/gaming style "Button" or "Banner", but the background is a realistic, high-res nature/human photo. They do not belong in the same world.

**NON-VIOLATION (Good Design):**
- Clean, flat vector art is OK.
- Text on a solid, clean color background is OK.
- Professional typography with no cheap effects is OK.


**OUTPUT FORMAT:**
Return a strictly valid JSON object:
{
    "is_violation": true, // true = Bad/Cheap Design; false = Professional/Clean Design
    "reason": "Describe the specific visual flaw (e.g., 'Cheap 3D gold text on complex floral background')."
}
"""

SYS_PROMPT_TEXT = """
You are a highly critical Senior Art Director. Your job is to flag "Low-Quality / Amateur" advertising designs.
You have ZERO TOLERANCE for "Cheap Ad Styles" (often called "Niu Pi Xian" in Chinese context).

**YOUR TASK:**
Determine if the image violates the "Text-Design Harmony" standards. 
If the design looks cheap, messy, or outdated, you MUST return "is_violation": true.

**STRICT VIOLATION CRITERIA (If ANY are found, return TRUE):**

1. **The "WordArt" Effect (廉价特效):**
   - Text uses heavy, amateurish strokes (thick white/colored outlines) that look jagged or pixelated.
   - Text uses outdated "Pseudo-3D" gradients (e.g., shiny gold/silver metal texture) that clash with a flat background.
   - Text has aggressive "Outer Glow" (neon glow) that makes it look blurry.
   - Text is stretched, squeezed, or distorted strictly to fit a space.

2. **Visual Clutter & Conflict (背景冲突):**
   - Text is placed directly on top of a "Busy Photograph" (leaves, city streets, crowds) without a sufficient background mask, making it hard to read.
   - The color of the text vibrates against the background (e.g., bright red text on bright green background).
   - "Patchwork Style": The text background looks like a sticker arbitrarily pasted onto a photo, completely ignoring the photo's lighting and perspective.


3. **Inconsistent Aesthetic (风格割裂):**
   - Foreground is a cartoon/gaming style "Button" or "Banner", but the background is a realistic, high-res nature/human photo. They do not belong in the same world.

**NON-VIOLATION (Good Design):**
- Clean, flat vector art is OK.
- Text on a solid, clean color background is OK.
- Professional typography with no cheap effects is OK (e.g. xx折扣 is OK).


**OUTPUT FORMAT:**
Return a strictly valid JSON object:
{
    "is_violation": true, // true = Bad/Cheap Design; false = Professional/Clean Design
    "reason": "Describe the specific visual flaw (e.g., 'Cheap 3D gold text on complex floral background')."
}
"""

# ==========================================
# 辅助函数
# ==========================================

def collect_images(input_dir: Path) -> List[Dict[str, str]]:
    """扫描目录下所有图片"""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()  # 排序，保证每次运行顺序一致
    
    print(f"[Info] Found {len(files)} images in {input_dir}")
    return [{"path": str(p), "filename": p.name} for p in files]

def parse_llm_output(text: str) -> Dict[str, Any]:
    """解析模型输出，提取 JSON"""
    default_res = {"is_violation": None, "reason": "Parse Error"}
    
    if not text:
        return default_res

    # 1. 尝试清洗 Markdown 标记
    text_clean = re.sub(r"```json|```", "", text).strip()
    
    # 2. 尝试直接解析 JSON
    try:
        # 有时候模型会在 JSON 后加废话，尝试只截取第一个 {...}
        match = re.search(r"\{.*?\}", text_clean, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            
            # 归一化 boolean 值
            is_v = data.get("is_violation")
            if isinstance(is_v, str):
                is_v = (is_v.lower() == "true")
            
            return {
                "is_violation": bool(is_v),
                "reason": data.get("reason", "No reason provided")
            }
    except Exception:
        pass

    # 3. 兜底策略：如果 JSON 解析失败，用正则暴力匹配关键词
    # 这是一个 Fallback，防止模型只输出了文本
    text_lower = text.lower()
    if "true" in text_lower and "false" not in text_lower:
        return {"is_violation": True, "reason": "Parsed from text (fallback)"}
    elif "false" in text_lower:
        return {"is_violation": False, "reason": "Parsed from text (fallback)"}
    
    return default_res

def prepare_vllm_inputs(batch_meta: List[Dict], processor) -> List[Dict]:
    """构建 vLLM 输入格式"""
    vllm_inputs = []
    user_query = "Analyze this image against the design rules and return the JSON decision."
    
    for item in batch_meta:
        img_path = item["path"]
        try:
            image_obj = Image.open(img_path).convert("RGB")
            
            # 构建对话模板
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYS_PROMPT_TEXT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_query}
                ]}
            ]
            
            # 应用 Chat Template
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            vllm_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": image_obj}
            })
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
            vllm_inputs.append(None)
            
    return vllm_inputs

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AI Design Auditor")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images to check")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local Qwen-VL model")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size (GPU count)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    meta_data = collect_images(input_path)
    
    if not meta_data:
        print("[Info] No images found. Exiting.")
        return

    # ---------------------------
    # 初始化模型
    # ---------------------------
    print(f"\n[Init] Loading Model: {args.model_path}")
    print(f"[Init] Tensor Parallel Size: {args.tp_size}")
    
    # 建议 max_model_len 设置大一些，VL 模型上下文消耗大
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.90, 
        max_model_len=8192, 
        enforce_eager=True, # Qwen-VL 推荐开启 Eager 模式
        limit_mm_per_prompt={"image": 1}
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 采样参数：温度设低，保证确定性
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
        top_p=0.85,
        repetition_penalty=1.05
    )

    # ---------------------------
    # 批量推理
    # ---------------------------
    results = []
    print(f"\n[Run] Starting Inference on {len(meta_data)} images...")
    
    for i in tqdm(range(0, len(meta_data), args.batch_size), desc="Processing Batches"):
        batch_meta = meta_data[i : i + args.batch_size]
        batch_inputs = prepare_vllm_inputs(batch_meta, processor)
        
        # 移除加载失败的 None 项，但需要保留索引对应关系比较麻烦
        # 这里简化处理：只推理成功的，失败的跳过
        valid_inputs = [inp for inp in batch_inputs if inp is not None]
        valid_indices = [idx for idx, inp in enumerate(batch_inputs) if inp is not None]
        
        if not valid_inputs:
            continue
            
        outputs = llm.generate(valid_inputs, sampling_params=sampling_params, use_tqdm=False)
        
        for local_idx, out in enumerate(outputs):
            # 找回原始 meta 数据
            original_meta = batch_meta[valid_indices[local_idx]]
            
            generated_text = out.outputs[0].text
            parsed_res = parse_llm_output(generated_text)
            
            results.append({
                "filename": original_meta["filename"],
                "path": original_meta["path"],
                "is_violation": parsed_res["is_violation"], # True = 违规, False = 合规
                "reason": parsed_res["reason"],
                "raw_output": generated_text
            })

    # ---------------------------
    # 统计与输出
    # ---------------------------
    total = len(results)
    true_count = sum(1 for r in results if r["is_violation"] is True)   # 违规
    false_count = sum(1 for r in results if r["is_violation"] is False) # 合规
    none_count = sum(1 for r in results if r["is_violation"] is None)   # 解析失败

    true_rate = (true_count / total * 100) if total > 0 else 0
    false_rate = (false_count / total * 100) if total > 0 else 0

    print("\n" + "="*60)
    print(f"AUDIT REPORT FOR: {input_path.name}")
    print("="*60)
    print(f"{'Total Images':<25}: {total}")
    print(f"{'Analysis Failed':<25}: {none_count}")
    print("-" * 60)
    print(f"{'VIOLATIONS (True)':<25}: {true_count}  ({true_rate:.2f}%)  [Hit Rules]")
    print(f"{'COMPLIANT (False)':<25}: {false_count}  ({false_rate:.2f}%)  [Safe]")
    print("="*60)

    # 保存详细 JSON
    output_file = input_path / f"audit_result_{input_path.name}.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Detailed JSON report saved to:\n-> {output_file}")
    except Exception as e:
        print(f"[Error] Could not save JSON: {e}")

if __name__ == "__main__":
    # 再次确保多进程启动方式
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()




You are a highly critical Senior Art Director specializing in Layout and Visual Hierarchy. 
Your job is to identify "Suffocating Designs" where elements are too cramped, lack breathing room, or feel disorganized due to poor spacing.

**YOUR TASK:**
Determine if the image violates the "Composition & Spacing" standards. 
A professional ad must have a clear "Sense of Breath" (Negative Space). If the elements feel "squeezed" or "crowded," you MUST return "is_violation": true.

**STRICT VIOLATION CRITERIA (If ANY are found, return TRUE):**

1. **Lack of Breathing Room (模块拥挤):**
   - The main subject (product/person), the headline text, and the logo/call-to-action are too close to each other.
   - Visually, there is almost no negative space (empty space) between major design modules.
   - The design feels "heavy" or "claustrophobic" because elements are packed too tightly.

2. **Edge Tension (贴边风险):**
   - Crucial text or logos are placed too close to the edges of the canvas (insufficient margins), making the layout feel unstable or amateurish.
   - Elements are "touching" or "tangent" to each other or the border without intentional overlapping.

3. **Information Overload (信息堆砌):**
   - The layout is filled with too many text blocks or icons with no clear separation. 
   - There is no clear "visual path"; the eye doesn't know where to rest because everything is competing for space.

4. **Scale Imbalance (比例失调):**
   - The main subject is so large that it forces the text into tiny, cramped corners, resulting in an uncomfortable distribution of weight.

**NON-VIOLATION (Good Design):**
- **Generous White Space:** Clear separation between the headline, the subject, and the footer information.
- **Defined Margins:** At least 10% of the canvas width/height is kept clear as a "safe zone" around the edges.
- **Structured Layout:** Elements follow a clear grid or intentional alignment that allows the design to "breathe."

**OUTPUT FORMAT:**
Return a strictly valid JSON object:
{
    "is_violation": true, // true = Crowded/Suffocating; false = Balanced/Spacious
    "reason": "Describe why the spacing fails (e.g., 'The logo is too close to the headline, creating visual clutter and a lack of breathing room')."
}
You are a highly critical Senior Art Director. Your goal is to evaluate "Information Accessibility." 
You must ensure that the advertising copy is not just "present," but "instantly readable and strategically placed."

**YOUR TASK:**
Analyze the image to determine if the text placement violates professional readability standards. 
If the text is buried, hard to read, or poorly positioned, you MUST return "is_violation": true.

**STRICT VIOLATION CRITERIA (If ANY are found, return TRUE):**

1. **Direct Overlay on Complex Background (复杂背景叠加):**
   - Text is placed directly over a "busy" part of a photograph (e.g., human faces, detailed textures, high-contrast patterns, or cluttered cityscapes) without a semi-transparent mask or solid background block.
   - The background details "cut through" the strokes of the text, making it difficult to identify the characters quickly.

2. **Poor Visual Catchiness (视线捕捉力弱):**
   - The primary marketing message (the "Hook") is placed in a "visual dead zone" (e.g., the very bottom edge or extreme corners) where the eye does not naturally land.
   - Important copy is too small or lacks contrast relative to its position, failing to be "eye-catching".

3. **Low Contrast / Visual Camouflage (识别度缺失):**
   - The text color is too similar to the background colors, causing the text to "camouflage" into the image.
   - There is no clear visual hierarchy; the eye has to "search" for the text instead of being drawn to it immediately.

**NON-VIOLATION (Good Design):**
- Text is placed on a "clean" area of the image (e.g., sky, plain wall, or a blurred background).
- Important text is placed in a focal point (e.g., top-middle or center) and is clearly legible at a glance.
- Text has a solid color backing/container when the background is complex.

**OUTPUT FORMAT:**
Return a strictly valid JSON object:
{
    "is_violation": true, // true = Hard to read/Poor placement; false = Legible/Strategic placement
    "reason": "Describe the specific placement issue (e.g., 'The primary headline is overlaid on a high-contrast floral pattern, making it nearly invisible')."
}
