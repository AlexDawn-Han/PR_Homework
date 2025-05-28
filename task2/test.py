import cv2
import os
import numpy as np
import joblib
import multiprocessing as mp
import xml.etree.ElementTree as ET
import config
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 应在导入 pyplot 之前设置
import matplotlib.pyplot as plt
from skimage.feature import hog
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# 图像加载与预处理
# ----------------------------
def load_and_preprocess_image(image_path,config):
    """ 加载并预处理图像
    
    :param image_path: 图像文件路径
    :param config: 配置对象
    
    :return: 原始图像、灰度图像和缩放后的图像
    """
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 缩放图像用于选择性搜索
    width = int(original_image.shape[1] * config.SCALE_PERCENT / 100)
    height = int(original_image.shape[0] * config.SCALE_PERCENT / 100)
    resized_image = cv2.resize(original_image, (width, height), interpolation=cv2.INTER_AREA)

    return original_image, gray_image, resized_image


# ----------------------------
# 获取候选区域并映射回原图尺寸
# ----------------------------
def get_candidate_regions(resized_image, original_image ,label_name ,config):
    """
    通过选择性搜索算法+尺寸筛选，获取候选区域
    
    :param resized_image: 缩放后的图像
    :param original_image: 原始图像
    :param label_name: 标签名称
    :param label_name: 待检测的标签名称
    
    :return: 原始图像中的候选区域列表 [(x, y, w, h), ...]
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(resized_image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    # print(f"总共找到了 {len(rects)} 个候选区域")

    # 映射回原始图像坐标并过滤小区域
    scale_x = original_image.shape[1] / resized_image.shape[1]
    scale_y = original_image.shape[0] / resized_image.shape[0]
    
    if label_name == 'car':
        scaled_rects = [
            (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            for x, y, w, h in rects
            if int(w * scale_x) >= config.car_min_width and int(h * scale_y) >= config.car_min_height
        ]
        # print(f"过滤后保留了 {len(scaled_rects)} 个候选区域（宽 ≥ {min_width}, 高 ≥ {min_height}）")
    elif label_name == 'licence':

        scaled_rects = [
            (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            for x, y, w, h in rects
            if int(w * scale_x) >= config.licence_min_width and int(h * scale_y) >= config.licence_min_height and int(w * scale_x) <= config.licence_max_width and int(h * scale_y) <= config.licence_max_height
        ]
        
        # print(f"过滤后保留了 {len(scaled_rects)} 个候选区域（宽 ≥ {min_width}, 高 ≥ {min_height}, 宽 ≤ {max_width}, 高 ≤ {max_height}）")
    return scaled_rects


# ----------------------------
# 提取 HOG 特征并使用 PCA + SVM 进行分类
# ----------------------------
def extract_features_and_predict(gray_image, rects, pca, svm_model,label_name,config):
    """
    使用 HOG 特征和 PCA + SVM 进行车牌检测
    
    :param gray_image: 灰度图像
    :param rects: 候选区域列表 [(x, y, w, h), ...]
    :param pca: PCA 模型
    :param svm_model: SVM 模型
    :param label_name: 待检测的标签名称
    :param config: 配置对象
    
    :return: 预测结果为1的候选区域列表 [(x, y, w, h), ...] 和对应的概率列表 [prob, ...]  
    """
    
    if label_name == 'car':
        target_size = config.car_target_size
        hog_params = config.car_hog_params
    elif  label_name == 'licence':
        target_size = config.licence_target_size
        hog_params = config.licence_hog_params

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
def soft_nms(boxes, probs, k=1, sigma=0.5, thresh=0.01):
    """
    Soft-NMS线性衰减非极大值抑制算法
    
    :param boxes: 候选框坐标列表
    :param probs: 候选框对应的概率列表
    :param k: 截取个数
    :param sigma: 衰减系数
    :param thresh: 最小置信度阈值
    
    :return: 处理后的候选框坐标列表和置信度列表
    """
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


def detect_license_plate_in_candidates(
    original_image,
    gray_image, 
    topk_rects, 
    topk_scores, 
    licence_pca,
    licence_svm_model,
    config
):
    """
    根据 topk_scores 从高到低遍历候选车辆区域，尝试从中检测车牌。
    
    :param original_image: 原始彩色图像
    :param gray_image: 灰度图
    :param topk_rects: 车辆候选框列表 [(x, y, w, h), ...]
    :param topk_scores: 每个候选框的置信度
    :param licence_pca: 车牌分类 PCA 模型
    :param licence_svm_model: 车牌分类 SVM 模型
    :param config: 配置对象或模块
    :return: (vehicle_box, license_box) 或者 None
    """
    # 按得分排序候选区域
    scored_rects = sorted(zip(topk_rects, topk_scores), key=lambda x: x[1], reverse=True)

    for vehicle_box, _ in scored_rects:
        x, y, w, h = vehicle_box

        # 检查 ROI 是否越界
        if y + h > gray_image.shape[0] or x + w > gray_image.shape[1]:
            continue

        # 截取车辆区域 ROI
        roi_color = original_image[y:y+h, x:x+w]
        roi_gray = gray_image[y:y+h, x:x+w]

        # 第二轮 Selective Search 在车辆区域内进行车牌检测
        resized_roi = cv2.resize(roi_color, (int(roi_gray.shape[1] * config.SCALE_PERCENT / 100),
                                       int(roi_gray.shape[0] * config.SCALE_PERCENT / 100)))
        
        # 获取候选车牌区域
        candidate_licence_rects = get_candidate_regions(resized_roi, roi_color, 'licence',config)

        # 映射回 ROI 图像坐标
        # mapped_rects = [(x + rx, y + ry, rw, rh) for (rx, ry, rw, rh) in candidate_licence_rects]

        # 提取特征并预测车牌
        filtered_licence_rects, filtered_licence_probs = extract_features_and_predict(
            roi_gray, candidate_licence_rects, licence_pca, licence_svm_model,'licence',config)

        # 应用 Soft-NMS
        licence_topk_rects, _ = soft_nms(filtered_licence_rects, filtered_licence_probs, k=1)

        if licence_topk_rects:
            global_licence_rects = [(x + rx, y + ry, rw, rh) for (rx, ry, rw, rh) in licence_topk_rects]
            # print(f"在候选车辆 {vehicle_box} 中成功检测到车牌！")
            return [vehicle_box], global_licence_rects

    # print("所有候选车辆区域均未检测到车牌。")
    return topk_rects, None

# ----------------------------
# 核心处理
# ----------------------------
def process_single_image(image_path, car_pca, car_svm_model, licence_pca, licence_svm_model, config):
    """
    处理单个图像文件，返回车牌位置信息。
    
    读取图像,预处理得到 (original_image, gray_image, resized_image)->
    车辆获取候选区域->候选区域结果为1的区域->
    排序取前三个->
        车牌获取候选区域->车牌结果为1的区域->
        返回车牌位置信息。
    
    :param image_path: 图像文件路径
    :param car_pca: 车辆分类 PCA 模型
    :param car_svm_model: 车辆分类 SVM 模型
    :param licence_pca: 车牌分类 PCA 模型
    :param licence_svm_model: 车牌分类 SVM 模型
    :param config: 配置对象或模块
    
    :return: 车辆和车牌位置信息，如 (vehicle_box, license_box) 或者None
    """
    
    
    # print(f"\nProcessing: {image_path}")
    TOP_K = 3
    original_image, gray_image, resized_image = load_and_preprocess_image(image_path,config)
    candidate_rects = get_candidate_regions(resized_image, original_image,'car',config)

    # 提取特征并预测
    filtered_rects, filtered_probs = extract_features_and_predict(gray_image, candidate_rects, car_pca, car_svm_model,'car',config)

    # print(f"共检测到 {len(filtered_rects)} 个预测为正样的候选区域")  

    # 应用 Soft-NMS
    topk_rects, topk_scores = soft_nms(filtered_rects, filtered_probs, k=TOP_K)
    
    # print(f"Soft-NMS 后选取了 Top-{len(topk_rects)} 个车辆候选区域")
    
    # 检测车牌
    car_box, license_box = detect_license_plate_in_candidates(
        original_image, gray_image, topk_rects, topk_scores,
        licence_pca, licence_svm_model, config)
    
    # print(f"检测车牌结果为：{license_box}")

    # 获取真实边框
    annotation_path = get_annotation_path(image_path)
    car_true_boxes = extract_boxes(annotation_path, label_name= 'car')
    licence_true_boxes = extract_boxes(annotation_path, label_name= 'licence')

    # 可视化结果
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # 显示经过 Soft-NMS 后的 Top-K 矩形（黄色）
    if car_box is not None:
        for rect in car_box:
            x, y, w, h = rect
            rect_patch = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='y', facecolor='none')
            ax.add_patch(rect_patch)
        
    if license_box is not None:    
        for rect in license_box:
            x, y, w, h = rect
            rect_patch = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='y', facecolor='none')
            ax.add_patch(rect_patch)

    # 绘制真实边框（绿色）
    for box in car_true_boxes:
        xmin, ymin, xmax, ymax = box
        w_true = xmax - xmin
        h_true = ymax - ymin
        true_rect_patch = plt.Rectangle((xmin, ymin), w_true, h_true, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(true_rect_patch)
    for box in licence_true_boxes:
        xmin, ymin, xmax, ymax = box
        w_true = xmax - xmin
        h_true = ymax - ymin
        true_rect_patch = plt.Rectangle((xmin, ymin), w_true, h_true, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(true_rect_patch)

    # plt.title(f"Top{TOP_K} Detection Result with Ground Truth")
    plt.axis('off')

    # 保存图像
    image_name = os.path.basename(image_path)
    save_path = os.path.join(config.result_dir, f'result_{image_name}')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1,dpi = 300)
    plt.close(fig)

    # 返回车辆和车牌预测位置
    return car_box, license_box

def process_and_evaluate(image_path, car_pca, car_svm_model,
                         licence_pca, licence_svm_model):
    """
    预测并评估结果。
    首先根据文件名读取标注文件->获取真实边框->
    预测边缘框->评估预测效果->返回评估结果。
    
    :param image_path: 图像文件路径
    :param car_pca: 车辆分类 PCA 模型
    :param car_svm_model: 车辆分类 SVM 模型
    :param licence_pca: 车牌分类 PCA 模型
    :param licence_svm_model: 车牌分类 SVM 模型
    :param config: 配置对象或模块
    
    :return: 评估结果
    """
    import config
    # 提取文件名
    image_filename = os.path.basename(image_path)
    
    annotation_path = get_annotation_path(image_path)
    
    # 提取真实框
    car_true_boxes = extract_boxes(annotation_path,label_name='car')
    licence_true_boxes = extract_boxes(annotation_path,label_name='licence')

    # 模型预测（返回的是 (x, y, w, h)）
    car_pred_boxes, license_pred_box= process_single_image(image_path, car_pca, car_svm_model, licence_pca, licence_svm_model, config)

    # 评估车辆检测
    car_tp, car_fp, car_fn, car_ious = evaluate(car_true_boxes, car_pred_boxes)
    car_avg_iou = np.mean(car_ious) if car_ious else 0.0
    num_car_preds = len(car_pred_boxes) if car_pred_boxes else 0

    # 评估车牌检测
        # 初始化默认值
    licence_tp = licence_fp = licence_fn = 0
    licence_ious = []
    num_licence_preds = 0

    if license_pred_box is not None:
        licence_tp, licence_fp, licence_fn, licence_ious = evaluate(licence_true_boxes, license_pred_box)
        licence_avg_iou = np.mean(licence_ious) if licence_ious else 0.0
        num_licence_preds = len(license_pred_box)
    else:
        licence_fn = len(licence_true_boxes)
        licence_avg_iou = 0.0

    # 写入日志
    with open(config.log_file, 'a') as f:
        f.write(f"{image_filename} | "
                f"Vehicle: TP={car_tp}, FP={car_fp}, FN={car_fn}, IoU={car_avg_iou:.4f}, Predictions={num_car_preds} | "
                f"Licence: TP={licence_tp}, FP={licence_fp}, FN={licence_fn}, IoU={licence_avg_iou:.4f}, Predictions={num_licence_preds}\n")

    
    return car_tp, car_fp, car_fn, car_ious, licence_tp, licence_fp, licence_fn, licence_ious

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
    
    :param true_boxes: 真实框列表 [(xmin, ymin, xmax, ymax)]
    :param pred_boxes: 预测框列表 [(x, y, w, h)] 或 None
    :param iou_threshold: IoU 阈值，默认为 0.5
    
    :return: tp, fp, fn, ious
    """
    if pred_boxes is None:
        pred_boxes = []

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
                continue
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

    fn = len(true_boxes) - tp
    return tp, fp, fn, ious
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
    # 创建结果目录（如果不存在）
    Path(config.result_dir).mkdir(exist_ok=True)

    # 清空或创建日志文件，并写入头信息
    with open(config.log_file, 'w') as f:
        f.write(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 加载模型
    car_pca = joblib.load(config.car_pca_model_path)
    car_svm_model = joblib.load(config.car_svm_model_path)
    
    licence_pca = joblib.load(config.licence_pca_model_path)
    licence_svm_model = joblib.load(config.licence_svm_model_path)

    # 获取所有 .jpg 文件
    image_files = [os.path.join(config.IMAGES_DIR, f) for f in os.listdir(config.IMAGES_DIR) if f.endswith('.jpg')]
    
    # 初始化统计变量
    total_car_tp = total_car_fp = total_car_fn = 0
    total_licence_tp = total_licence_fp = total_licence_fn = 0
    all_car_ious = []
    all_licence_ious = []

    # 使用进程池并行处理
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_and_evaluate,
                                   image_path=image_path,
                                   car_pca=car_pca,
                                   car_svm_model=car_svm_model,
                                   licence_pca=licence_pca,
                                   licence_svm_model=licence_svm_model) for image_path in image_files]

        # 添加进度条
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing Images"):
            try:
                car_tp, car_fp, car_fn, car_ious, licence_tp, licence_fp, licence_fn, licence_ious = future.result()

                # 累计车辆检测结果
                total_car_tp += car_tp
                total_car_fp += car_fp
                total_car_fn += car_fn
                all_car_ious.extend(car_ious)

                # 累计车牌检测结果
                total_licence_tp += licence_tp
                total_licence_fp += licence_fp
                total_licence_fn += licence_fn
                all_licence_ious.extend(licence_ious)
            except Exception as e:
                print(f"Generated an exception: {e}")

    # 计算车辆指标
    car_precision = total_car_tp / (total_car_tp + total_car_fp) if (total_car_tp + total_car_fp) > 0 else 0
    car_recall = total_car_tp / (total_car_tp + total_car_fn) if (total_car_tp + total_car_fn) > 0 else 0
    car_f1 = 2 * car_precision * car_recall / (car_precision + car_recall) if (car_precision + car_recall) > 0 else 0
    car_mean_iou = sum(all_car_ious) / len(all_car_ious) if len(all_car_ious) > 0 else 0

    # 计算车牌指标
    licence_precision = total_licence_tp / (total_licence_tp + total_licence_fp) if (total_licence_tp + total_licence_fp) > 0 else 0
    licence_recall = total_licence_tp / (total_licence_tp + total_licence_fn) if (total_licence_tp + total_licence_fn) > 0 else 0
    licence_f1 = 2 * licence_precision * licence_recall / (licence_precision + licence_recall) if (licence_precision + licence_recall) > 0 else 0
    licence_mean_iou = sum(all_licence_ious) / len(all_licence_ious) if len(all_licence_ious) > 0 else 0

    # 写入最终统计信息到日志
    with open(config.log_file, 'a') as f:
        f.write("\nFinal Statistics:\n")
        f.write(f"Vehicle Detection:\n")
        f.write(f"Total TP: {total_car_tp}, Total FP: {total_car_fp}, Total FN: {total_car_fn}\n")
        f.write(f"Precision: {car_precision:.4f}\n")
        f.write(f"Recall: {car_recall:.4f}\n")
        f.write(f"F1 Score: {car_f1:.4f}\n")
        f.write(f"Mean IoU across test set: {car_mean_iou:.4f}\n\n")

        f.write(f"License Plate Detection:\n")
        f.write(f"Total TP: {total_licence_tp}, Total FP: {total_licence_fp}, Total FN: {total_licence_fn}\n")
        f.write(f"Precision: {licence_precision:.4f}\n")
        f.write(f"Recall: {licence_recall:.4f}\n")
        f.write(f"F1 Score: {licence_f1:.4f}\n")
        f.write(f"Mean IoU across test set: {licence_mean_iou:.4f}\n")
        f.write(f"Evaluation ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

