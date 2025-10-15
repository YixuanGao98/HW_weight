#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple

# 允许的评级标签（小写）
ALLOWED_LABELS = ["bad", "poor", "fair", "good", "excellent"]

# 英文问题文本（按要求固定两句）
QUESTION_TEXT = {
    "aes": "Please rate the aesthetic quality of this advertisement image.",
    "ads": "Please rate the advertising attributes of this advertisement image.",
}

# 规则：只要包含关键词就视为对应类型（兼容“并提供您的理由”等扩展说法）
KW_AESTHETIC = "美观度进行评级"
KW_ADS = "广告属性进行评级"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Format-1 JSON to Format-2 JSON.")
    p.add_argument("--input", "-i", required=True, help="Path to the input JSON (Format-1).")
    p.add_argument("--output", "-o", required=True, help="Path to the output JSON (Format-2).")
    return p.parse_args()

def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 既支持最外层是 list，也兼容不规范情况
    if isinstance(data, dict):
        # 允许单对象也转为列表
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input JSON should be a list of objects in 'Format-1'.")
    return data

def question_type_and_text(user_content: str) -> Tuple[str, str]:
    """
    根据用户提示文本判定 question_type（aes/ads），并给出英文 question 文本。
    规则：包含关键词即可匹配（兼容“并提供您的理由”等）。
    """
    if KW_AESTHETIC in user_content:
        return "aes", QUESTION_TEXT["aes"]
    if KW_ADS in user_content:
        return "ads", QUESTION_TEXT["ads"]
    # 若两者都未命中，尽力匹配（更宽松：只要包含“美观度/广告属性”四字之一）
    if "美观度" in user_content:
        return "aes", QUESTION_TEXT["aes"]
    if "广告属性" in user_content:
        return "ads", QUESTION_TEXT["ads"]
    # 实在无法判断则抛错，提示检查数据
    raise ValueError(f"无法从提问中判断 question_type：{user_content}")

def extract_answer_label(assistant_content: str) -> str:
    """
    从 assistant 的 content 中提取评级标签（bad/poor/fair/good/excellent）。
    策略：
      1) 全文低化后用词边界优先匹配一个标签。
      2) 若找不到，尝试用常见中文词→英文标签的映射（可按需扩展）。
    """
    text = assistant_content.strip()
    low = text.lower()

    # 优先：词边界匹配（避免匹配到 'goodness' 等）
    for label in ALLOWED_LABELS:
        if re.search(rf"\b{label}\b", low):
            return label.capitalize()

    # 兜底：粗糙前缀（示例数据常见 'Fair,' 'Good,'）
    for label in ALLOWED_LABELS:
        if low.startswith(label):
            return label.capitalize()

    # 进一步兜底：中文关键词粗映射（可扩展）
    cn2en = {
        "较差": "poor",
        "很差": "bad",
        "差": "poor",
        "一般": "fair",
        "中等": "fair",
        "良好": "good",
        "很好": "good",
        "优秀": "excellent",
        "极好": "excellent",
    }
    for k, v in cn2en.items():
        if k in text:
            return v.capitalize()

    # 如果仍未识别，尝试逗号/空白前的首词
    head = re.split(r"[,\s，。；;！!？?\n\r]+", text, maxsplit=1)[0].lower()
    if head in ALLOWED_LABELS:
        return head.capitalize()

    raise ValueError(f"无法从回答中提取评级标签（bad/poor/fair/good/excellent）：{assistant_content}")

def get_first_user_and_assistant(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    返回第一个 user.content 和 第一个 assistant.content。
    """
    user_content = None
    assistant_content = None
    for m in messages:
        role = m.get("role", "").strip().lower()
        content = m.get("content", "")
        if role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and assistant_content is None:
            assistant_content = content
        if user_content is not None and assistant_content is not None:
            break
    if user_content is None or assistant_content is None:
        raise ValueError("messages 中缺少 user 或 assistant 的 content。")
    return user_content, assistant_content

def convert(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    主转换逻辑：从格式一到格式二，并在同一 question_type 下按 image_id 去重。
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[str, set] = {"aes": set(), "ads": set()}  # 记录在各类型下出现过的 image_id

    next_id = 1
    for rec in records:
        # 取图片路径（只取第一张）
        images = rec.get("images", [])
        if not images:
            # 跳过无图样本
            continue
        file_path = images[0]
        image_id = os.path.basename(file_path)

        # 取问答
        messages = rec.get("messages", [])
        try:
            user_q, assistant_a = get_first_user_and_assistant(messages)
        except Exception as e:
            # 跳过异常样本
            # 若需要严格模式可改为 raise
            continue

        # question_type & 英文 question 文本
        try:
            qtype, qtext = question_type_and_text(user_q)
        except Exception:
            # 跳过无法判断的问题
            continue

        # 提取答案标签
        try:
            label = extract_answer_label(assistant_a)  # 返回首字母大写
        except Exception:
            # 跳过无法抽取评级的样本
            continue

        # 去重（在同一 question_type 下，image_id 只保留第一次）
        if image_id in seen[qtype]:
            continue
        seen[qtype].add(image_id)

        # 构造格式二条目
        out.append({
            "id": next_id,
            "image_id": image_id,
            "quetion": qtext,            # 注意：按你的键名保留 'quetion'
            "answer": label,             # 规范化为首字母大写
            "question_type": qtype,      # 'aes' 或 'ads'
            "file_path": file_path,      # 原路径
        })
        next_id += 1

    return out

def main():
    args = parse_args()
    in_data = load_input(args.input)
    out_data = convert(in_data)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {len(out_data)} items to {args.output}")

if __name__ == "__main__":
    main()
