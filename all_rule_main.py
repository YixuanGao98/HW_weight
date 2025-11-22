#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
from typing import Callable, Dict, Tuple, List, Optional
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ===== 你的环境变量 / 模型路径 =====
os.environ["YOLO_FONT"] = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
os.environ["U2NET_HOME"] = "model/rembg"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第0号GPU
_YUNET_ONNX = os.environ.get("YUNET_ONNX", "yunet.onnx")

YOLO_BEST = "model/yolo/train8/weights/best.pt"
Qalign_BEST='/home/wsw/model' #q-align权重
Qalign_BEST='/mnt/sda/gyx/huggingface' #q-align权重
Qualiclip_BEST='model/rule_5_6_7/QualiCLIP+_koniq.pth' #Qualiclip权重
REAL_BEST='model/rule_5_6_7/zhenshigan.bin' #真实感权重
POSTPRO_BEST='model/rule_5_6_7/houqi.bin' #后期处理权重
# ===== 你的规则函数导入 =====
from rule_1 import is_uncomfortable_palette_image
from rule_2 import is_low_hue_diversity_image
from rule_3 import is_promotion_image
from rule_4 import is_low_value_image
from rule_8 import is_promo_with_yolo_image
from rule_9 import is_subject_too_large_image
from rule_10 import is_outside_safe_area_image
from rule_5 import is_bad_quality
from rule_6_7 import is_realistic_or_postprocessing

# ---------- 模型加载 ----------

#加载清晰度模型
from transformers import AutoModelForCausalLM
import torch
from pyiqa import create_metric

global Qalign_model, Qualiclip_model
Qalign_model = None
Qualiclip_model = None
def quality_model():
    global Qalign_model, Qualiclip_model

    ####下载one-align
    from huggingface_hub import snapshot_download
    # # 设置镜像地址（国内加速）
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 目标模型和本地路径
    repo_id = 'q-future/one-align'
    local_dir = os.path.join(Qalign_BEST, 'one-align')
    # 检查本地目录是否存在
    if not os.path.exists(local_dir):
        print(f"模型目录 {local_dir} 不存在，开始下载...")
        snapshot_download(
            repo_id=repo_id,
            repo_type='model',
            local_dir=local_dir,
            resume_download=True
        )
        print("下载完成！")
    else:
        print(f"模型目录 {local_dir} 已存在，跳过下载。")
    # 加载模型
    Qalign_model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    ###load qualiclip
    Qualiclip_model = create_metric(
        'qualiclip+', metric_mode='NR', device=None, pretrained=False
    )
    Qualiclip_model.load_weights(Qualiclip_BEST, weight_keys='params')

    return Qalign_model,Qualiclip_model


from realistic_postprocess.vit.models.modeling import VisionTransformer, CONFIGS
global ViT_model_real,ViT_model_post
ViT_model_real = None
ViT_model_post = None
def vit_model(pretrained_dir):
    # Prepare model
    config = CONFIGS['R50-ViT-B_16']
    num_classes = 100
    ViT_model = VisionTransformer(config, 384, zero_head=True, num_classes=num_classes)
    
    # Load weights from PyTorch checkpoint (.bin, .pth)
    if pretrained_dir:
        checkpoint = torch.load(pretrained_dir)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:  # Common in PyTorch Lightning
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:     # Some checkpoints use "model"
            state_dict = checkpoint["model"]
        else:                           # Direct state_dict
            state_dict = checkpoint

        # Remove "module." prefix if present (from DataParallel/DDP)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load weights into the model
        ViT_model.load_state_dict(state_dict, strict=True)
    
    ViT_model.cuda()

    return ViT_model


# ---------- 工具函数 ----------
def iter_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
    for fn in sorted(os.listdir(folder)[:]):###50
        if fn.lower().endswith(exts):
            yield os.path.join(folder, fn)


def safe_read(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[警告] 图片读取失败: {img_path}")
    return img


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------- 主流程 ----------
def build_ocr_if_needed() -> PaddleOCR:
    # 采用你给的 PP-OCRv5 server 配置
    ocr = PaddleOCR(
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir="model/paddleocr/det/PP-OCRv5_server_det_infer",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir="model/paddleocr/rec/PP-OCRv5_server_rec_infer",
        lang="ch",
    )
    return ocr


def build_yolo_if_needed() -> YOLO:
    return YOLO(YOLO_BEST)


def get_rule_mapping(ocr: Optional[PaddleOCR], yolo: Optional[YOLO], qalign_threshold1: int, qalign_threshold2: int, qualiclip_threshold: int):
    """
    返回：
      rule_name -> {
        'title': 中文描述（用于打印）,
        'needs': set({'ocr','yolo'})，
        'fn': 接收 (img, idx) -> bool 的函数
      }
    """
    mapping: Dict[str, Dict] = {}

    # 1 配色不舒适
    mapping["rule_1"] = {
        "title": "配色不舒适",
        "needs": set(),
        "fn": lambda img, idx=None: not bool(is_uncomfortable_palette_image(img))
    }

    # 2 色相少（低色相多样性）
    mapping["rule_2"] = {
        "title": "色相多",
        "needs": set(),
        "fn": lambda img, idx=None: not bool(is_low_hue_diversity_image(img))
    }

    # 3 促销感强（OCR）
    if ocr is not None:
        mapping["rule_3"] = {
            "title": "促销感强",
            "needs": {"ocr"},
            "fn": lambda img, idx=None: bool(is_promotion_image(img, ocr))
        }

    # 4 低价值（OCR）
    if ocr is not None:
        mapping["rule_4"] = {
            "title": "低价值",
            "needs": {"ocr"},
            "fn": lambda img, idx=None: bool(is_low_value_image(img, ocr))
        }

    # 8 存在促销元素（YOLO + OCR）
    if (ocr is not None) and (yolo is not None):
        mapping["rule_8"] = {
            "title": "存在促销元素",
            "needs": {"ocr", "yolo"},
            "fn": lambda img, idx=None: bool(is_promo_with_yolo_image(img, model=yolo, idx=idx or 0, ocr=ocr))
        }

    # 9 占比过大
    mapping["rule_9"] = {
        "title": "占比过大",
        "needs": set(),
        "fn": lambda img, idx=None: bool(is_subject_too_large_image(img))
    }

    # 10 超出安全区（OCR）
    if ocr is not None:
        mapping["rule_10"] = {
            "title": "超出安全区",
            "needs": {"ocr"},
            "fn": lambda img, idx=None: bool(is_outside_safe_area_image(img, ocr_instance=ocr, idx=idx or 0))
        }

    # 5 清晰度
    mapping["rule_5"] = {
        "title": "清晰度差",
        "needs": set(),
        "fn": lambda img, idx=None: bool(is_bad_quality(img,Qalign_model,Qualiclip_model, qalign_threshold1,qalign_threshold2,qualiclip_threshold))
    }
    # 6 清晰度
    mapping["rule_6"] = {
        "title": "真实感差",
        "needs": set(),
        "fn": lambda img, idx=None: bool(is_realistic_or_postprocessing(ViT_model_real,img))
    }
    
    # 7 清晰度
    mapping["rule_7"] = {
        "title": "未经后期处理",
        "needs": set(),
        "fn": lambda img, idx=None: bool(is_realistic_or_postprocessing(ViT_model_post,img))
    }
    return mapping

def main(
    data_root: str = "data_self",
    rules_to_run: Optional[List[str]] = None,
    labels_to_run: Optional[List[str]] = None,
    verbose: bool = True,
    qalign_threshold1: int = 3.5,
    qalign_threshold2: int = 5,
    qualiclip_threshold: int = 5,
):

    need_ocr = False
    need_yolo = False
    need_quality_model = False
    need_vit_model_real = False
    need_vit_model_post = False
    _target_rules = rules_to_run or [f"rule_{i}" for i in range(1, 11)]
    if any(r in _target_rules for r in ("rule_3", "rule_4", "rule_8", "rule_10")):
        need_ocr = True
    if any(r in _target_rules for r in ("rule_8",)):
        need_yolo = True
    if any(r in _target_rules for r in ("rule_5",)):
        need_quality_model = True
    if any(r in _target_rules for r in ("rule_6",)):
        need_vit_model_real = True
    if any(r in _target_rules for r in ("rule_7",)):
        need_vit_model_post = True
        
    ocr = build_ocr_if_needed() if need_ocr else None
    yolo = build_yolo_if_needed() if need_yolo else None
    quality_model() if need_quality_model else None
    global ViT_model_real
    ViT_model_real=vit_model(REAL_BEST) if need_vit_model_real else None
    global ViT_model_post
    ViT_model_post=vit_model(POSTPRO_BEST) if need_vit_model_post else None
    
    rule_map = get_rule_mapping(ocr, yolo, qalign_threshold1,qalign_threshold2,qualiclip_threshold)

    available_rules = [r for r in _target_rules if r in rule_map]
    if not available_rules:
        print("[错误] 所选规则均不可用（可能缺 OCR/YOLO 依赖）。")
        return

    # 允许只处理 True/False
    labels = labels_to_run or [ "False","True"]

    # 全局计数（用于保持你最后的汇总行）
    global_total = 0
    count_promo = 0                     # rule_3
    count_low_value = 0                 # rule_4
    count_uncomfortable_palette = 0     # rule_1
    count_low_hue_diversity = 0         # rule_2
    count_is_promo_with_yolo = 0        # rule_8
    count_is_subject_too_large = 0      # rule_9
    count_is_outside_safe_area = 0      # rule_10
    count_is_bad_quality = 0      # rule_5
    count_is_low_realistic = 0      # rule_6
    count_is_need_postprocessing = 0      # rule_7
    
    
    # 每个规则评估
    for rule_name in available_rules:
        title = rule_map[rule_name]["title"]
        fn: Callable = rule_map[rule_name]["fn"]

        tp = fp = tn = fn_cnt = 0
        rule_total = 0
        running_idx = 0  # 仅给 rule_8 的 idx 使用

        # 遍历 True/False 子目录
        labels=[rule_name,'normal']
        for lab in labels:
            lab_dir = os.path.join(data_root,  lab)
            if not os.path.isdir(lab_dir):
                if verbose:
                    print(f"[提示] 跳过不存在的目录：{lab_dir}")
                continue

            for img_path in iter_images(lab_dir):
                img = safe_read(img_path)
                if img is None:
                    continue
                rule_total += 1
                global_total += 1
                running_idx += 1

                pred = bool(fn(img, running_idx))
                gt = (lab != "normal")

                if pred and gt:
                    tp += 1
                elif pred and (not gt):
                    fp += 1
                elif (not pred) and (not gt):
                    tn += 1
                else:  # not pred and gt
                    fn_cnt += 1

                # 同时更新“总览行”中的各个计数（按预测为 True 计数）
                if pred:
                    if rule_name == "rule_3":
                        count_promo += 1
                    elif rule_name == "rule_4":
                        count_low_value += 1
                    elif rule_name == "rule_1":
                        count_uncomfortable_palette += 1
                    elif rule_name == "rule_2":
                        count_low_hue_diversity += 1
                    elif rule_name == "rule_8":
                        count_is_promo_with_yolo += 1
                    elif rule_name == "rule_9":
                        count_is_subject_too_large += 1
                    elif rule_name == "rule_10":
                        count_is_outside_safe_area += 1
                    elif rule_name == "rule_5":
                        count_is_bad_quality += 1
                    elif rule_name == "rule_6":
                        count_is_low_realistic += 1
                    elif rule_name == "rule_7":
                        count_is_need_postprocessing += 1
                        
                if verbose:
                    print(f"[{rule_name}/{lab}] {os.path.basename(img_path)} -> 预测:{pred} | 真实:{gt}")

        # 规则级别统计输出
        precision, recall, f1 = prf1(tp, fp, fn_cnt)
        acc = (tp + tn) / rule_total if rule_total > 0 else 0.0

        print(
            f"\n=== 规则 {rule_name}（{title}）统计 ===\n"
            f"总数: {rule_total} | TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn_cnt}\n"
            f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n"
        )

    # 你的“总览行”（尽量保持你原来的风格与字段）
    if global_total > 0:
        print(
            f"\n统计：总图片数={global_total}，"
            f"        促销感强数={count_promo}|{round(count_promo / global_total * 100, 2)}%，"
            f"        低价值数={count_low_value}|{round(count_low_value / global_total * 100, 2)}%，"
            f"        配色不舒适数={count_uncomfortable_palette}|{round(count_uncomfortable_palette / global_total * 100, 2)}%，"
            f"        色相多={count_low_hue_diversity}|{round(count_low_hue_diversity / global_total * 100, 2)}%，"
            f"        存在促销元素={count_is_promo_with_yolo}|{round(count_is_promo_with_yolo / global_total * 100, 2)}%，"
            f"        占比过大={count_is_subject_too_large}|{round(count_is_subject_too_large / global_total * 100, 2)}%，"
            f"        超出安全区={count_is_outside_safe_area}|{round(count_is_outside_safe_area / global_total * 100, 2)}%，"
            f"        低清晰度={count_is_bad_quality}|{round(count_is_bad_quality / global_total * 100, 2)}%，"
            f"        真实感差={count_is_low_realistic}|{round(count_is_low_realistic / global_total * 100, 2)}%，"
            f"        未经后期处理={count_is_need_postprocessing}|{round(count_is_need_postprocessing / global_total * 100, 2)}%，"
            f"        end"
        )
    else:
        print("未检测到有效图片。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遍历 data_self/rule_xx/True|False 评估各规则")
    parser.add_argument("--data_root", type=str, default="/home/wsw/jikaiyuan/data/object_stage3", help="根目录（包含 rule_1..rule_10）")
    parser.add_argument("--rules", type=str, default="", help="只跑指定规则，逗号分隔，例如: rule_1,rule_8；留空=全部")
    parser.add_argument("--labels", type=str, default="", help="只跑指定标签，逗号分隔: True,False；可只给 True 或 False；留空=两者都跑")
    parser.add_argument("--quiet", action="store_true", help="少打印明细，仅打印统计")
    
    # 添加三个清晰度的阈值参数
    parser.add_argument("--qalign_threshold1", type=float, default=3.5, help="范围[0,5], 清晰度第一个阈值，可以用于筛选低质量图片")
    parser.add_argument("--qalign_threshold2", type=float, default=4.17, help="范围[0,5], 清晰度第二个阈值，可以用于筛选中等质量图片")
    parser.add_argument("--qualiclip_threshold", type=float, default=0.68, help="范围[0,1], 清晰度第三个阈值，可以用于筛选中等质量图片")
    args = parser.parse_args()

    rules_to_run = [r.strip() for r in args.rules.split(",") if r.strip()] if args.rules else None
    labels_to_run = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else None

    main(
        data_root=args.data_root,
        rules_to_run=rules_to_run,
        labels_to_run=labels_to_run,
        qalign_threshold1=args.qalign_threshold1,
        qalign_threshold2=args.qalign_threshold2,
        qualiclip_threshold=args.qualiclip_threshold,
        verbose=not args.quiet
    )
