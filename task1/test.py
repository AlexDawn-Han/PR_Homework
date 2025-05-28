import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import hog
import numpy as np
import joblib
from pathlib import Path
import multiprocessing as mp
import xml.etree.ElementTree as ET
from tqdm import tqdm
from datetime import datetime


# ----------------------------
# 配置参数
# ----------------------------
DATASET_PATH = 'D:\\Datasets\\PR-homework\\dataset\\test\\'
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels')

SCALE_PERCENT = 20
TOP_K = 1

LABEL_NAME =  'licence'  # 'car' # 'licence'

if LABEL_NAME == 'car':
    pca_model_path = 'car_pca_model.joblib'
    svm_model_path = 'car_svm_model.pkl'
    target_size = (200, 200)
    min_width, min_height = 400, 400
    # HOG特征提取参数
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (20, 20),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'visualize': False
    }
    result_dir = os.path.join(DATASET_PATH, 'car_results')  # 保存结果的目录
    # 日志文件路径
    log_file = os.path.join('car_evaluation_log.txt')
elif LABEL_NAME == 'licence':
    pca_model_path = 'licence_pca_model.joblib'
    svm_model_path = 'licence_svm_model.pkl'
    target_size = (100, 30)
    max_width, max_height = 300, 120
    min_width, min_height = 60, 30
    # HOG特征提取参数
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (10, 3),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'visualize': False
    }
    result_dir = os.path.join(DATASET_PATH, 'licence_results')  # 保存结果的目录
    # 日志文件路径
    log_file = os.path.join('licence_evaluation_log.txt')
    
# 创建结果目录（如果不存在）
Path(result_dir).mkdir(exist_ok=True)

# ----------------------------
# 图像加载与预处理
# ----------------------------
def load_and_preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 缩放图像用于选择性搜索
    width = int(original_image.shape[1] * SCALE_PERCENT / 100)
    height = int(original_image.shape[0] * SCALE_PERCENT / 100)
    resized_image = cv2.resize(original_image, (width, height), interpolation=cv2.INTER_AREA)

    return original_image, gray_image, resized_image


# ----------------------------
# 获取候选区域并映射回原图尺寸
# ----------------------------
def get_candidate_regions(resized_image, original_image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(resized_image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    # print(f"总共找到了 {len(rects)} 个候选区域")

    # 映射回原始图像坐标并过滤小区域
    scale_x = original_image.shape[1] / resized_image.shape[1]
    scale_y = original_image.shape[0] / resized_image.shape[0]
    
    if LABEL_NAME == 'car':
        scaled_rects = [
            (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            for x, y, w, h in rects
            if int(w * scale_x) >= min_width and int(h * scale_y) >= min_height
        ]
        # print(f"过滤后保留了 {len(scaled_rects)} 个候选区域（宽 ≥ {min_width}, 高 ≥ {min_height}）")
    elif LABEL_NAME == 'licence':

        scaled_rects = [
            (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            for x, y, w, h in rects
            if int(w * scale_x) >= min_width and int(h * scale_y) >= min_height and int(w * scale_x) <= max_width and int(h * scale_y) <= max_height
        ]
        
        # print(f"过滤后保留了 {len(scaled_rects)} 个候选区域（宽 ≥ {min_width}, 高 ≥ {min_height}, 宽 ≤ {max_width}, 高 ≤ {max_height}）")
    return scaled_rects


# ----------------------------
# 提取 HOG 特征并使用 PCA + SVM 进行分类
# ----------------------------
def extract_features_and_predict(gray_image, rects, pca, svm_model):
    hog_features = []
    valid_rects = []

    for x, y, w, h in rects:
        if y + h > gray_image.shape[0] or x + w > gray_image.shape[1]:
            continue

        roi_gray = gray_image[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, target_size, interpolation=cv2.INTER_AREA)

        roi_hog_fd = hog(roi_resized,**hog_params)
        hog_features.append(roi_hog_fd)
        valid_rects.append((x, y, w, h))

    if not hog_features:
        return [], []

    hog_matrix = np.array(hog_features)
    pca_features = pca.transform(hog_matrix)
    predictions = svm_model.predict(pca_features)
    probs = svm_model.predict_proba(pca_features)[:, 1]

    filtered_indices = [i for i, pred in enumerate(predictions) if pred == 1]
    filtered_rects = [valid_rects[i] for i in filtered_indices]
    filtered_probs = probs[filtered_indices]

    return filtered_rects, filtered_probs


# ----------------------------
# Soft-NMS 实现（线性衰减）
# ----------------------------
def soft_nms(boxes, probs, k=1, iou_thresh=0.3, sigma=0.5, thresh=0.01):
    if not boxes:
        return [], []

    converted_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    pick = []
    scores = np.array(probs)
    areas = (converted_boxes[:, 2] - converted_boxes[:, 0]) * (converted_boxes[:, 3] - converted_boxes[:, 1])

    idxs = np.argsort(scores)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(converted_boxes[i, 0], converted_boxes[idxs[:last], 0])
        yy1 = np.maximum(converted_boxes[i, 1], converted_boxes[idxs[:last], 1])
        xx2 = np.minimum(converted_boxes[i, 2], converted_boxes[idxs[:last], 2])
        yy2 = np.minimum(converted_boxes[i, 3], converted_boxes[idxs[:last], 3])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        interArea = w * h

        ious = interArea / (areas[i] + areas[idxs[:last]] - interArea)
        weights = np.exp(- (ious ** 2) / sigma)
        scores[idxs[:last]] *= weights

        idxs = np.delete(idxs, last)
        idxs = idxs[scores[idxs] > thresh]

    pick_scores = scores[pick]
    sorted_indices = np.argsort(pick_scores)[::-1]
    top_k_indices = sorted_indices[:k]

    top_k_boxes = [boxes[pick[i]] for i in top_k_indices]
    top_k_scores = [pick_scores[i] for i in top_k_indices]

    return top_k_boxes, top_k_scores

# ----------------------------
# 核心处理
# ----------------------------
def process_single_image(image_path, pca, svm_model):
    # print(f"\nProcessing: {image_path}")
    original_image, gray_image, resized_image = load_and_preprocess_image(image_path)
    candidate_rects = get_candidate_regions(resized_image, original_image)

    # 提取特征并预测
    filtered_rects, filtered_probs = extract_features_and_predict(gray_image, candidate_rects, pca, svm_model)

    # print(f"共检测到 {len(filtered_rects)} 个预测为正样的候选区域")

    # 应用 Soft-NMS
    topk_rects, topk_scores = soft_nms(filtered_rects, filtered_probs, k=TOP_K)

    # print(f"Soft-NMS 后选取了 Top-{len(topk_rects)} 个候选区域")
    # for idx, (rect, score) in enumerate(zip(topk_rects, topk_scores)):
    #     print(f"Top-{idx+1} 框: {rect}, 置信度: {score:.4f}")

    # 获取真实边框
    annotation_path = get_annotation_path(image_path)
    true_boxes = extract_boxes(annotation_path, label_name=LABEL_NAME)

    # 可视化结果
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # 显示经过 Soft-NMS 后的 Top-K 矩形（黄色）
    for rect in topk_rects:
        x, y, w, h = rect
        rect_patch = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect_patch)

    # 绘制真实边框（绿色）
    for box in true_boxes:
        xmin, ymin, xmax, ymax = box
        w_true = xmax - xmin
        h_true = ymax - ymin
        true_rect_patch = plt.Rectangle((xmin, ymin), w_true, h_true, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(true_rect_patch)

    plt.title(f"Top{TOP_K} Detection Result with Ground Truth")
    plt.axis('off')

    # 保存图像
    image_name = os.path.basename(image_path)
    save_path = os.path.join(result_dir, f'result_{image_name}')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1,dpi = 300)
    plt.close(fig)

    # 返回 topk_rects 和 topk_scores，供日志记录使用
    return topk_rects, topk_scores

def process_and_evaluate(image_path, pca, svm_model, log_file):
    # 提取文件名
    image_filename = os.path.basename(image_path)
    
    annotation_path = get_annotation_path(image_path)
    
    # 提取真实框
    true_boxes = extract_boxes(annotation_path,label_name=LABEL_NAME)

    # 模型预测（返回的是 (x, y, w, h)）
    pred_boxes, pred_scores = process_single_image(image_path, pca, svm_model)

    # 评估
    tp, fp, fn, _, ious = evaluate(true_boxes, pred_boxes)

    # 写入日志
    avg_iou = sum(ious) / len(ious) if len(ious) > 0 else 0.0
    with open(log_file, 'a') as f:
        formatted_scores = [f'{x:.4f}' for x in pred_scores]
        f.write(f"Image: {image_filename}, pred_boxes={pred_boxes}, pred_scores={formatted_scores}, Avg IoU={avg_iou:.4f}\n")

    return tp, fp, fn, ious   

# ----------------------------
# 评估
# ----------------------------
def extract_boxes(annotation_path: str,label_name = 'car') -> list[tuple[int, int, int, int]]:
    """
    从XML标注文件中提取所有 car 的 bounding box。

    :param annotation_path: XML 文件路径
    :return: 所有 car 的 bounding boxes 列表
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    true_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        if name == label_name:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            true_boxes.append((xmin, ymin, xmax, ymax))
    return true_boxes

def xywh_to_xminyminxmaxymax(box):
    x, y, w, h = box
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return (xmin, ymin, xmax, ymax)

def iou(pred_box, gt_box):
    """
    计算两个 bounding box 的 IoU（交并比）
    
    参数：
        pred_box: 预测框，(xmin, ymin, xmax, ymax)
        gt_box: 真实框，(xmin, ymin, xmax, ymax)

    返回：
        IoU 值
    """
    # 获取预测框和真实框的坐标
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box

    # 计算交集区域坐标
    inter_xmin = max(pred_xmin, gt_xmin)
    inter_ymin = max(pred_ymin, gt_ymin)
    inter_xmax = min(pred_xmax, gt_xmax)
    inter_ymax = min(pred_ymax, gt_ymax)

    # 计算交集区域面积
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # 计算预测框与真实框的面积
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)

    # 计算并集面积
    union_area = pred_area + gt_area - inter_area

    # 计算 IoU
    if union_area == 0:
        return 0.0
    iou_score = inter_area / union_area
    return iou_score

def evaluate(true_boxes, pred_boxes, iou_threshold=0.5):
    """
    评估预测框和真实框的匹配情况。
    
    参数：
        true_boxes: list of tuple, 真实框列表 [(xmin, ymin, xmax, ymax), ...]
        pred_boxes: list of tuple, 预测框列表 [(x, y, w, h), ...]
        iou_threshold: float, 判定为正确预测的最小 IoU 值

    返回：
        tp: 正确预测数量
        fp: 错误预测数量
        fn: 漏检数量
        matched_indices: 匹配的真实框索引
        ious: 所有预测框的最大 IoU 值
    """
    pred_boxes_converted = [xywh_to_xminyminxmaxymax(box) for box in pred_boxes]
    
    matched_gt = set()
    tp = 0
    fp = 0
    ious = []

    for pred_box in pred_boxes_converted:
        best_iou = 0
        best_idx = -1
        for idx, gt_box in enumerate(true_boxes):
            if idx in matched_gt:
                continue  # 已经被匹配过
            current_iou = iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        ious.append(best_iou)

        if best_iou >= iou_threshold and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(true_boxes) - tp  # 漏检数 = 总真实框数 - 成功匹配的个数

    return tp, fp, fn, matched_gt, ious

def get_annotation_path(image_path):
    """
    将 .jpg 图像路径转换为对应的 .xml 标注文件路径。
    """
    annotation_path = image_path.replace('images', 'labels')  # 替换目录名
    annotation_path = annotation_path.replace('.jpg', '.xml')  # 替换扩展名
    return annotation_path
   
# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == "__main__":
    print(f"正常执行：{LABEL_NAME}")
    # 清空或创建日志文件，并写入头信息
    with open(log_file, 'w') as f:
        f.write(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 加载模型
    pca = joblib.load(pca_model_path)
    svm_model = joblib.load(svm_model_path)

    # 获取所有 .jpg 文件
    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    # 多进程处理并添加进度条
    pool = mp.Pool(processes=mp.cpu_count())
    results = []

    # 使用 tqdm 包裹进度
    tasks = [(image_path, pca, svm_model, log_file) for image_path in image_files]
    with tqdm(total=len(image_files), desc="Processing Images") as pbar:
        def update_result(_):
            pbar.update()

        for image_path in image_files:
            result = pool.apply_async(process_and_evaluate, args=(image_path, pca, svm_model, log_file), callback=update_result)
            results.append(result)

        pool.close()
        pool.join()

    # 收集结果
    for res in results:
        tp, fp, fn, ious = res.get()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

    # 计算全局指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = sum(all_ious) / len(all_ious) if len(all_ious) > 0 else 0

    # 写入最终统计信息到日志
    with open(log_file, 'a') as f:
        f.write("\nFinal Statistics:\n")
        f.write(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Mean IoU across test set: {mean_iou:.4f}\n")
        f.write(f"Evaluation ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")