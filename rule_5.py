
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

Could not fetch URL https://pypi.org/simple/ml-collections/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/ml-collections/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))) - skipping
ERROR: Could not find a version that satisfies the requirement ml-collections (from versions: none)
ERROR: No matching distribution found for ml-collections
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ml-collections
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/vit/train.py", line 334, in <module>
    main()
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/vit/train.py", line 330, in main
    train(args, model)
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/vit/train.py", line 199, in train
    for step, batch in enumerate(epoch_iterator):
  File "/root/miniconda3/lib/python3.12/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1344, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/data/dataloader.py", line 1370, in _process_data
    data.reraise()
  File "/root/miniconda3/lib/python3.12/site-packages/torch/_utils.py", line 706, in reraise
    raise exception
PIL.UnidentifiedImageError: Caught UnidentifiedImageError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 309, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torchvision/datasets/folder.py", line 245, in __getitem__
    sample = self.loader(path)
             ^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torchvision/datasets/folder.py", line 284, in default_loader
    return pil_loader(path)
           ^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/torchvision/datasets/folder.py", line 263, in pil_loader
    img = Image.open(f)
          ^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/PIL/Image.py", line 3532, in open
    raise UnidentifiedImageError(msg)
PIL.UnidentifiedImageError: cannot identify image file <_io.BufferedReader name='/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/vit/dataset/huawei_ad/objective_database/train/真实感3/FALSE/IMG_0218.JPG'>

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from PIL import Image
import torch

import pandas as pd
from pyiqa import create_metric
import cv2

def calculate_image_quality(model, image_np):
    """
    计算numpy格式图像的质量分数
    
    参数:
        weight_path: 模型权重路径
        image_np: numpy格式的图像数组 (H, W, C)或(N, H, W, C)
    """

    
    # 确保输入是4维 (N, H, W, C)
    if len(image_np.shape) == 3:
        image_np = np.expand_dims(image_np, axis=0)  # 添加batch维度
        
    # 收集结果
    results = []
    
    for i in range(image_np.shape[0]):
        # 转换为PIL Image
        img = Image.fromarray(image_np[i]).convert("RGB")
        
        # 计算质量分数
        with torch.no_grad():
            quality_score = model.score([img], task_="quality", input_="image")
            # aesthetics_score = model.score([img], task_="aesthetics", input_="image")

    return quality_score.item()
    
def calculate_image_quality_qualiclip(image_np):
    """Inference demo for pyiqa."""
    metric_name = 'qualiclip+'
    # 确保输入是4维 (N, H, W, C)
    if len(image_np.shape) == 3:
        image_np = np.expand_dims(image_np, axis=0)  # 添加batch维度
        
    for i in range(image_np.shape[0]):
        # 转换为PIL Image
        img = Image.fromarray(image_np[i]).convert("RGB")
        

    # set up IQA model
    iqa_model = create_metric(
        metric_name, metric_mode='NR', device=None
    )

    ref_img_path = None
    score = iqa_model(img, ref_img_path).cpu().item()
    return score
    
def is_bad_quality(img,model):
        #####清晰度，真实感，后期处理，需要转换图像
    # 1. 转换通道顺序: BGR -> RGB（如果模型需要RGB输入）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, C), RGB格式
    
    # 2. 添加batch维度 (N=1)
    img_np = np.expand_dims(img_rgb, axis=0)  # shape: (1, H, W, C)
    
    image_quality_qalign = calculate_image_quality(model,img_np)#需要从q-future/one-align下载相应权重
    image_quality_qualiclip = calculate_image_quality_qualiclip(img_np)
    
    # if (image_quality_qalign < 4.3) & (image_quality_qualiclip < 0.54):
    if (image_quality_qualiclip < 0.53)| (image_quality_qalign < 3.5):
        return True
    else:
        return False
        
# 使用示例
if __name__ == "__main__":
    # 加载图像为numpy数组的示例
    from PIL import Image
    img_path = "/mnt/sda/gyx/huawei_ad/3_6/y60073309_102_258208558_10_20250210192633.jpg"
    img_np = cv2.imread(img_path)  # (H, W, C)
    
    results = is_bad_quality(img_np)
    print(results)
    
# image_quality_qualiclip < 0.52
# === 规则 rule_5（清晰度差）统计 ===
# 总数: 100 | TP: 40 | FP: 1 | TN: 49 | FN: 10
# Accuracy: 0.8900 | Precision: 0.9756 | Recall: 0.8000 | F1: 0.8791

#  (image_quality_qualiclip < 0.53)| (image_quality_qalign < 3.5):
# === 规则 rule_5（清晰度差）统计 ===
# 总数: 100 | TP: 46 | FP: 4 | TN: 46 | FN: 4
# Accuracy: 0.9200 | Precision: 0.9200 | Recall: 0.9200 | F1: 0.9200
