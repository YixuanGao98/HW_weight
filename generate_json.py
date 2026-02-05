import os
import json

def generate_json(true_dir, false_dir, output_file):
    result = []
    
    # 处理true文件夹中的图片（支持中文文件名）
    for filename in os.listdir(true_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 保留原始中文文件名，确保JSON能正确存储
            result.append({
                "image": '清晰度-splash_clarity/'+filename,  # 直接使用中文文件名
                "gt_score": True
            })
    
    # 处理false文件夹中的图片（支持中文文件名）
    for filename in os.listdir(false_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            result.append({
                "image": '正常-normal_data/'+filename,  # 直接使用中文文件名
                "gt_score": False
            })
    
    # 写入JSON文件（确保中文正常写入）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)  # ensure_ascii=False保留中文

# 使用示例
generate_json('/home/wsw/gyx/code_11.28/清晰度-splash_clarity', '/home/wsw/gyx/code_11.28/正常-normal_data', 'hw_test_清晰.json')
