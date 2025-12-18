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
