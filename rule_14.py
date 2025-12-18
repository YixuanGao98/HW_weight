import cv2
import numpy as np
import math
# 如果你没有安装 paddleocr，请注释掉下面这几行，直接使用我模拟的 blocks 数据测试
try:
    from paddleocr import PaddleOCRVL
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    print("未检测到 PaddleOCR 环境，将使用模拟数据演示距离计算逻辑。")

def setup_ocr():
    """初始化模型"""
    if not HAS_PADDLE:
        return None
        
    model_path = "/home/wsw/jikaiyuan/stage2/code/code_2025_12_18/PaddleOCR-VL"
    # 目录下需要包含 pdmodel, pdiparams, infer_cfg.yml
    ocr = PaddleOCRVL(
        vl_rec_model_dir=model_path,
        layout_detection_model_dir="/home/wsw/jikaiyuan/stage2/code/code_2025_12_18/PaddleOCR-VL/PP-DocLayoutV2"
    )
    return ocr

def _extract_text_blocks_from_ocr_(ocr_results):
    """提取文本块逻辑"""
    blocks = []
    # 适配 PaddleOCR-VL 的返回结构
    # 注意：根据版本不同，有时直接是 list，有时是对象。这里沿用你提供的逻辑。
    if hasattr(ocr_results, 'json'): # 单个结果对象
        res_list = [ocr_results]
    elif isinstance(ocr_results, list):
        res_list = ocr_results
    else:
        res_list = []

    for r in res_list:
        # 兼容不同层级的返回格式
        parsing_list = r.json["res"]["parsing_res_list"] if hasattr(r, 'json') else r.get("res", {}).get("parsing_res_list", [])
        
        for block in parsing_list:
            # 过滤空内容
            if not block.get("block_content", "").strip():
                continue
            blocks.append(block)
    return blocks

def calculate_min_distance(bbox1, bbox2):
    """
    计算两个矩形框之间的最短距离 (Edge-to-Edge)
    bbox 格式: [xmin, ymin, xmax, ymax]
    """
    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2
    
    # 1. 计算水平方向的间距 (x_gap)
    # 如果两个块在水平方向有重叠，gap 为 0；否则为 l2-r1 或 l1-r2
    x_gap = max(0, l2 - r1, l1 - r2)
    
    # 2. 计算垂直方向的间距 (y_gap)
    y_gap = max(0, t2 - b1, t1 - b2)
    
    # 3. 欧几里得距离
    dist = math.sqrt(x_gap**2 + y_gap**2)
    
    return dist, x_gap, y_gap

def analyze_layout_distances(blocks):
    """
    分析所有 Block 之间的距离
    """
    n = len(blocks)
    print(f"\n--- 开始计算 {n} 个文本块之间的距离 ---\n")
    
    results = []
    
    for i in range(n):
        block_a = blocks[i]
        bbox_a = block_a['block_bbox']
        content_a = block_a['block_content']
        
        # 寻找最近的邻居
        min_dist = float('inf')
        nearest_idx = -1
        
        # 存储该 block 与其他所有 block 的距离
        distances = []
        
        for j in range(n):
            if i == j:
                continue
                
            block_b = blocks[j]
            bbox_b = block_b['block_bbox']
            content_b = block_b['block_content']
            
            # 计算距离
            dist, dx, dy = calculate_min_distance(bbox_a, bbox_b)
            
            distances.append({
                'target_index': j,
                'target_content': content_b,
                'distance': dist,
                'dx': dx,
                'dy': dy
            })
            
            # 更新最近邻居
            if dist < min_dist:
                min_dist = dist
                nearest_idx = j
        
        # 打印当前 Block 的分析结果
        print(f"Block {i} [{content_a[:10]}...]:")
        if nearest_idx != -1:
            nearest_content = blocks[nearest_idx]['block_content']
            print(f"  -> 最近邻居: Block {nearest_idx} [{nearest_content[:10]}...]")
            print(f"  -> 最短距离: {min_dist:.2f} px")
        
        # 如果需要打印所有距离（可选，如果 block 太多建议注释掉）
        # for d in distances:
        #     print(f"     vs Block {d['target_index']}: dist={d['distance']:.1f} (dx={d['dx']}, dy={d['dy']})")
            
        results.append({
            'source_index': i,
            'nearest_index': nearest_idx,
            'min_distance': min_dist,
            'all_distances': distances
        })
        print("-" * 30)
        
    return results

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 1. 尝试初始化 OCR 并读取图片
    ocr = setup_ocr()
    blocks = []

    if ocr:
        print("正在运行 OCR 推理...")
        test_image = "/home/wsw/gyx/code_11.28/test_data/排布间距/20241212112442EECEB9747D62434FBDC1F1CA71BE7829 (1).jpg"
        
        try:
            img = cv2.imread(test_image)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {test_image}")
                
            results = ocr.predict(img)
            blocks = _extract_text_blocks_from_ocr_(results)
            print(f"成功提取 {len(blocks)} 个文本块。")
            
        except Exception as e:
            print(f"OCR 运行出错: {e}")
    else:
        # 2. 如果没有环境，使用你提供的模拟数据进行测试（为了演示代码逻辑）
        print("使用模拟数据演示...")
        blocks = [
            {'block_label': 'paragraph_title', 'block_content': '00首经典老歌', 'block_bbox': [100, 50, 300, 80], 'block_id': 0}, 
            {'block_label': 'text', 'block_content': '1.雾里看花', 'block_bbox': [100, 100, 200, 130], 'block_id': 1}, 
            {'block_label': 'text', 'block_content': '2. 灰姑娘', 'block_bbox': [300, 100, 400, 130], 'block_id': 2},
            {'block_label': 'text', 'block_content': '3. 下沙', 'block_bbox': [100, 150, 200, 180], 'block_id': 3}
        ]

    # 3. 核心功能：计算距离
    if blocks:
        dist_results = analyze_layout_distances(blocks)
        
        # 示例：获取 Block 1 到 Block 2 的具体距离
        if len(blocks) >= 3:
            b1 = blocks[1]['block_bbox']
            b2 = blocks[2]['block_bbox']
            d, dx, dy = calculate_min_distance(b1, b2)
            print(f"\n[特定查询] Block 1 和 Block 2 之间的距离: {d:.2f} (水平间距: {dx}, 垂直间距: {dy})")
            
            
            
            
            import cv2
import numpy as np
from paddleocr import PaddleOCRVL
model_path = "/home/wsw/jikaiyuan/stage2/code/code_2025_12_18/PaddleOCR-VL"
# 目录下需要包含 pdmodel, pdiparams, infer_cfg.yml

ocr = PaddleOCRVL(
    vl_rec_model_dir = model_path,   # 这个目录必须包含 pdmodel, pdiparams, infer_cfg.yml
    layout_detection_model_dir = "/home/wsw/jikaiyuan/stage2/code/code_2025_12_18/PaddleOCR-VL/PP-DocLayoutV2"
)

def _extract_text_blocks_from_ocr_(ocr_results):
    """
    从 OCR 结果中提取非 image 的文本块，返回 list[dict]，每个 dict 至少包含：
    - block_content
    - block_bbox = [x1, y1, x2, y2]
    """
    blocks = []
    for r in ocr_results:
        parsing_list = r.json["res"]["parsing_res_list"]
        for block in parsing_list:
            # if block.get("block_label") == "image":
            #     continue
            if not block.get("block_content", "").strip():
                continue
            blocks.append(block)
    return blocks

# ========= OCR 文本块 =========
test_image = "/home/wsw/gyx/code_11.28/test_data/排布间距/20241212112442EECEB9747D62434FBDC1F1CA71BE7829 (1).jpg"
img=cv2.imread(test_image)
results = ocr.predict(img)
blocks = _extract_text_blocks_from_ocr_(results)
print(blocks)
[{'block_label': 'paragraph_title', 'block_content': '00首经典老歌', 'block_bbox': [...], 'block_id': 0, 'block_order': 1}, {'block_label': 'text', 'block_content': '1.雾里看花', 'block_bbox': [...], 'block_id': 1, 'block_order': 2}, {'block_label': 'text', 'block_content': '2. 灰姑娘', 'block_bbox': [...], 'block_id': 2, 'block_order': 3}, {'block_label': 'text', 'block_content': '3. 下沙', 'block_bbox': [...], 'block_id': 3, 'block_order': 4}, {'block_label': 'text', 'block_content': '5.心雨', 'block_bbox': [...], 'block_id': 4, 'block_order': 5}, {'block_label': 'text', 'block_content': '7.外面的世界', 'block_bbox': [...], 'block_id': 5, 'block_order': 6}, {'block_label': 'text', 'block_content': '9. 安妮', 'block_bbox': [...], 'block_id': 6, 'block_order': 7}, {'block_label': 'text', 'block_content': '11.禁锢', 'block_bbox': [...], 'block_id': 7, 'block_order': 8}, {'block_label': 'text', 'block_content': '13.女人花', 'block_bbox': [...], 'block_id': 8, 'block_order': 9}, {'block_label': 'text', 'block_content': '15. 红豆', 'block_bbox': [...], 'block_id': 9, 'block_order': 10}, {'block_label': 'text', 'block_content': '4. 舞女泪', 'block_bbox': [...], 'block_id': 10, 'block_order': 11}, {'block_label': 'text', 'block_content': '6. 海阔天空', 'block_bbox': [...], 'block_id': 11, 'block_order': 12}, {'block_label': 'text', 'block_content': '8. 单身情歌', 'block_bbox': [...], 'block_id': 12, 'block_order': 13}, {'block_label': 'text', 'block_content': '10.心如刀割', 'block_bbox': [...], 'block_id': 13, 'block_order': 14}, {'block_label': 'text', 'block_content': '12. 大中国', 'block_bbox': [...], 'block_id': 14, 'block_order': 15}, {'block_label': 'text', 'block_content': '14.约定', 'block_bbox': [...], 'block_id': 15, 'block_order': 16}, {'block_label': 'text', 'block_content': '16. 恋曲1990', 'block_bbox': [...], 'block_id': 16, 'block_order': 17}, {'block_label': 'text', 'block_content': '听一听 >', 'block_bbox': [...], 'block_id': 17, 'block_order': 18}, {'block_label': 'text', 'block_content': '000', 'block_bbox': [...], 'block_id': 18, 'block_order': 19}]


import cv2
import numpy as np
from paddleocr import PaddleOCRVL

def _extract_text_blocks_from_ocr_(ocr_results):
    """
    从 OCR 结果中提取非 image 的文本块，返回 list[dict]，每个 dict 至少包含：
    - block_content
    - block_bbox = [x1, y1, x2, y2]
    """
    blocks = []
    for r in ocr_results:
        parsing_list = r.json["res"]["parsing_res_list"]
        for block in parsing_list:
            if block.get("block_label") == "image":
                continue
            if not block.get("block_content", "").strip():
                continue
            blocks.append(block)
    return blocks


def is_text_design_uncoordinated_image(
    img,
    ocr,
    bg_uniform_ratio_thresh=0.7,      # KMeans 主色占比分数（色块）
    bg_uniform_std_thresh=12.0,       # 主色亮度 std（越小越均匀）
    context_edge_ratio_thresh=1.3,    # 背景复杂度 / ROI 复杂度
    multihue_delta_h_thresh=25.0,     # 字体多色 ΔH 阈值
    plate_sat_mean_thresh=80.0,       # 色块饱和度均值下限
    plate_edge_simple_thresh=5.0,     # ROI 边缘密度上限（越小越平滑）
    plate_val_std_thresh=18.0,        # 色块亮度方差上限（避免照片纹理）
    color_clash_delta_h_thresh=30.0   # 色块主色与全局主色的最小色相差
):
    """
    文字-设计搭配协调（增强版启发式）：

    子规则：
    A) 色块承载文案且色块颜色与整体调性差异较大（突兀色块）
    B) 色块叠加在相对复杂的背景上（原规则）
    C) 字体存在明显多色/渐变（粗略）

    True  = 命中任意子规则（疑似设计不协调）
    False = 暂未发现明显问题
    """
    h_img, w_img = img.shape[:2]

    # ========= 全局调色板（整图主色） =========
    # 下采样，避免计算太慢
    if w_img > 0 and h_img > 0:
        scale = 128.0 / max(w_img, h_img)
        new_w = max(1, int(w_img * scale))
        new_h = max(1, int(h_img * scale))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        Zg = small_hsv.reshape(-1, 3).astype(np.float32)

        Kg = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)
        compact_g, labels_g, centers_g = cv2.kmeans(
            Zg, Kg, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        global_hues = [float(c[0]) for c in centers_g]  # H 通道中心
    else:
        global_hues = []

    print("====== 全局调色板 Hue 中心（0-180）======")
    if global_hues:
        print("Global Hues:", ", ".join(f"{h:.1f}" for h in global_hues))
    else:
        print("Global Hues: (图像尺寸异常，无法计算)")
    print("=====================================")

    # ========= OCR 文本块 =========
    results = ocr.predict(img)
    blocks = _extract_text_blocks_from_ocr_(results)

    if not blocks:
        print("没有检测到文本块，视为正常。")
        return False

    issue_found = False

    for idx, block in enumerate(blocks):
        x1, y1, x2, y2 = block["block_bbox"]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 20 or bh < 10:
            print(f"[块 {idx}] bbox 太小 ({bw}x{bh})，跳过。")
            continue

        roi = img[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Z = roi_hsv.reshape(-1, 3).astype(np.float32)

        # ---- ROI 基本统计 ----
        H = Z[:, 0]
        S = Z[:, 1]
        V = Z[:, 2]
        sat_mean = float(S.mean())
        sat_std  = float(S.std())
        val_mean = float(V.mean())
        val_std  = float(V.std())

        # ---- Step 1: KMeans 分色，主色统计 ----
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)
        compact, labels, centers = cv2.kmeans(
            Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=K)
        total = counts.sum()
        dom_idx = int(np.argmax(counts))
        dom_ratio = counts[dom_idx] / float(total + 1e-6)
        dom_pixels = Z[labels == dom_idx]
        bg_std = float(dom_pixels[:, 2].std()) if len(dom_pixels) > 0 else 999.0
        dom_h = float(centers[dom_idx, 0])
        dom_s = float(centers[dom_idx, 1])
        dom_v = float(centers[dom_idx, 2])

        # ---- Step 2: ROI & context 边缘密度 ----
        pad = int(0.05 * max(w_img, h_img))
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w_img, x2 + pad)
        y2c = min(h_img, y2 + pad)

        context = img[y1c:y2c, x1c:x2c]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges_roi = cv2.Canny(gray_roi, 100, 200)
        roi_edge_density = float(edges_roi.mean())

        gray_ctx = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        edges_ctx = cv2.Canny(gray_ctx, 100, 200)
        ctx_edge_density = float(edges_ctx.mean())

        complex_around = (
            ctx_edge_density > roi_edge_density * context_edge_ratio_thresh
            and ctx_edge_density > 10
        )

        # ---- Step 3: “色块”判定（两路） ----
        # 3.1 KMeans 主色法（原逻辑）
        plate_kmeans = (
            dom_ratio > bg_uniform_ratio_thresh and bg_std < bg_uniform_std_thresh
        )

        # 3.2 平滑高饱和度法（新增）
        plate_smooth = (
            sat_mean > plate_sat_mean_thresh
            and roi_edge_density < plate_edge_simple_thresh
            and val_std < plate_val_std_thresh
        )

        has_color_plate = plate_kmeans or plate_smooth

        # ---- Step 4: 色块与全局调色板的色差 ----
        if global_hues:
            delta_hs = [abs(dom_h - gh) for gh in global_hues]
            min_delta_h_global = float(min(delta_hs))
        else:
            min_delta_h_global = 0.0

        color_clash = (
            has_color_plate and min_delta_h_global > color_clash_delta_h_thresh
        )

        # ---- Step 5: 色块叠加在复杂背景上（子规则 B）----
        plate_on_complex = has_color_plate and complex_around

        # ---- Step 6: 字体多色 / 渐变（子规则 C）----
        if plate_kmeans:
            text_pixels = Z[labels != dom_idx]
        else:
            text_pixels = Z

        multihue_font = False
        delta_h_text = 0.0
        if len(text_pixels) > 50:
            Kt = 2
            compact_t, labels_t, centers_t = cv2.kmeans(
                text_pixels, Kt, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            h_vals = np.sort(centers_t[:, 0])
            delta_h_text = float(abs(h_vals[1] - h_vals[0]))
            if delta_h_text > multihue_delta_h_thresh:
                multihue_font = True

        # ---- Debug 打印当前块的所有中间信息 ----
        print("\n========== 文本块 idx={} ==========".format(idx))
        print(f"bbox=({x1},{y1},{x2},{y2}), size=({bw}x{bh}), area_ratio={bw*bh/(w_img*h_img+1e-6):.4f}")
        print("ROI HSV stats: sat_mean={:.1f}, sat_std={:.1f}, val_mean={:.1f}, val_std={:.1f}".format(
            sat_mean, sat_std, val_mean, val_std
        ))
        print("KMeans 主色: dom_ratio={:.2f}, bg_std(V)={:.2f}, dom_h={:.1f}, dom_s={:.1f}, dom_v={:.1f}".format(
            dom_ratio, bg_std, dom_h, dom_s, dom_v
        ))
        print("Edges: roi_edge={:.2f}, ctx_edge={:.2f}, complex_around={}".format(
            roi_edge_density, ctx_edge_density, complex_around
        ))
        print("Plate flags: plate_kmeans={}, plate_smooth={}, has_color_plate={}".format(
            plate_kmeans, plate_smooth, has_color_plate
        ))
        print("Global color clash: min_delta_h_global={:.1f}, color_clash={}".format(
            min_delta_h_global, color_clash
        ))
        print("Multihue font: delta_h_text={:.1f}, multihue_font={}".format(
            delta_h_text, multihue_font
        ))

        # ---- 子规则触发情况 ----
        rule_A_color_plate_clash = color_clash          # 色块颜色与整体调性差异大
        rule_B_plate_on_complex  = plate_on_complex     # 色块叠加复杂背景
        rule_C_multihue_font     = multihue_font        # 多色字体

        print("Rule A (突兀色块):", rule_A_color_plate_clash)
        print("Rule B (复杂背景上的色块):", rule_B_plate_on_complex)
        print("Rule C (多色/渐变字体):", rule_C_multihue_font)

        # 有任一子规则命中，就认为这个块有问题
        if rule_A_color_plate_clash or rule_B_plate_on_complex or rule_C_multihue_font:
            issue_found = True

    print("\n最终结果（True=存在疑似设计不协调，False=暂未检测到）:", issue_found)
    return issue_found
