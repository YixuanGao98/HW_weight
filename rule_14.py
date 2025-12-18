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
