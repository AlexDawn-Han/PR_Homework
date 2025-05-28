import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from skimage.color import rgb2gray
from tqdm import tqdm


# -----------------------------
# 配置参数
# -----------------------------
DATASET_PATH = 'D:\\Datasets\\PR-homework\\dataset\\train\\'
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels')

LABEL_NAME = 'licence'  # car 
if LABEL_NAME == 'car':
    target_size = (200, 200)
    output_pos_dir = os.path.join(DATASET_PATH, 'task2','car_pos_1')
    output_neg_dir = os.path.join(DATASET_PATH, 'task2','car_neg_n')
    neg_sample_multiplier = 5  # 负样本数量 = 正样本数量 × neg_sample_multiplier
elif LABEL_NAME == 'licence':
    target_size = (100, 30)
    output_pos_dir = os.path.join(DATASET_PATH, 'task2','licence_pos_1')
    output_neg_dir = os.path.join(DATASET_PATH, 'task2','licence_neg_n')
    neg_sample_multiplier = 10  # 负样本数量 = 正样本数量 × neg_sample_multiplier



# 创建输出目录（若不存在）
os.makedirs(output_pos_dir, exist_ok=True)
os.makedirs(output_neg_dir, exist_ok=True)


# -----------------------------
# 工具函数
# -----------------------------
def is_overlap(xmin: int, ymin: int, xmax: int, ymax: int, regions: list) -> bool:
    """
    判断一个区域是否与其他已选区域重叠。
    
    :param xmin: 区域左上角x坐标
    :param ymin: 区域左上角y坐标
    :param xmax: 区域右下角x坐标
    :param ymax: 区域右下角y坐标
    :param regions: 已存在的区域列表 [(xmin, ymin, xmax, ymax), ...]
    :return: 是否重叠
    """
    for rxmin, rymin, rxmax, rymax in regions:
        if not (xmax <= rxmin or xmin >= rxmax or ymax <= rymin or ymin >= rymax):
            return True
    return False


def generate_negative_roi(
    image_shape: tuple[int, int, int], 
    occupied_regions: list[tuple[int, int, int, int]], 
    num_samples: int = 1
) -> list[tuple[int, int, int, int]]:
    """
    在非目标区域内生成不重叠的负样本ROI。

    :param image_shape: 图像尺寸 (height, width, channels)
    :param occupied_regions: 已占用的目标区域列表
    :param num_samples: 需要生成的负样本数量
    :return: 生成的负样本区域列表
    """
    height, width = image_shape[:2]
    neg_rois = []

    while len(neg_rois) < num_samples:
        w = np.random.randint(50, width // 2)
        h = np.random.randint(50, height // 2)
        xmin = np.random.randint(0, width - w)
        ymin = np.random.randint(0, height - h)
        xmax = xmin + w
        ymax = ymin + h

        if not is_overlap(xmin, ymin, xmax, ymax, occupied_regions):
            neg_rois.append((xmin, ymin, xmax, ymax))

    return neg_rois

def generate_negative_roi_in_car_not_licence(
    image_shape: tuple[int, int, int],
    target_boxes: list[tuple[int, int, int, int]],
    licence_boxes: list[tuple[int, int, int, int]],
    num_samples: int = 1
) -> list[tuple[int, int, int, int]]:
    """
    在 car 框内、但不在 licence 框内的区域中生成负样本。
    """
    height, width = image_shape[:2]
    neg_rois = []

    while len(neg_rois) < num_samples:
        # 随机选择一个 car 框
        selected_box = target_boxes[np.random.randint(0, len(target_boxes))]
        rxmin, rymin, rxmax, rymax = selected_box

        # 随机生成 ROI 尺寸
        w = np.random.randint(50, min(rxmax - rxmin, width // 2))
        h = np.random.randint(50, min(rymax - rymin, height // 2))

        xmin = rxmin + np.random.randint(0, max(1, (rxmax - rxmin) - w))
        ymin = rymin + np.random.randint(0, max(1, (rymax - rymin) - h))
        xmax = xmin + w
        ymax = ymin + h

        roi_rect = (xmin, ymin, xmax, ymax)

        # 判断是否完全不与任何 licence 框重叠
        if not is_overlap(*roi_rect, licence_boxes):
            # 同时确保该区域在 car 框内
            if (xmin >= rxmin and ymin >= rymin and xmax <= rxmax and ymax <= rymax):
                neg_rois.append(roi_rect)

    return neg_rois

def extract_boxes(annotation_path: str,label_name = 'car') -> list[tuple[int, int, int, int]]:
    """
    从XML标注文件中提取所有 target 的 bounding box。

    :param annotation_path: XML 文件路径
    :return: 所有 target 的 bounding boxes 列表
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    target_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        if name == label_name:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            target_boxes.append((xmin, ymin, xmax, ymax))
    return target_boxes


def save_positive_samples(
    image: np.ndarray,
    target_boxes: list[tuple[int, int, int, int]],
    base_name: str,
    output_dir: str,
    target_size: tuple[int, int]
):
    """
    保存正样本图片。

    :param image: 原始图像数据
    :param target_boxes: 所有 target 的 bounding boxes
    :param base_name: 图片基础名（不含扩展名）
    :param output_dir: 输出目录
    :param target_size: 目标尺寸
    """
    for idx, (xmin, ymin, xmax, ymax) in enumerate(target_boxes):
        roi = image[ymin:ymax, xmin:xmax]
        roi_gray = rgb2gray(roi)
        roi_resized = cv2.resize(roi_gray, target_size)
        roi_8bit = (roi_resized * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"{base_name}_pos_{idx}.jpg")
        cv2.imwrite(output_path, roi_8bit)


def save_negative_samples(
    image: np.ndarray,
    image_shape: tuple[int, int, int],
    target_boxes: list[tuple[int, int, int, int]],
    base_name: str,
    output_dir: str,
    target_size: tuple[int, int],
    multiplier: int = 5
):
    """
    生成并保存负样本图片。

    :param image: 原始图像数据
    :param image_shape: 图像尺寸 (height, width, channels)
    :param target_boxes: 所有 target 的 bounding boxes
    :param base_name: 图片基础名（不含扩展名）
    :param output_dir: 输出目录
    :param target_size: 目标尺寸
    :param multiplier: 每张图负样本数 = 正样本数 × multiplier
    """
    num_samples = len(target_boxes) * multiplier
    neg_regions = generate_negative_roi(image_shape, target_boxes, num_samples=num_samples)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(neg_regions):
        roi = image[ymin:ymax, xmin:xmax]
        roi_gray = rgb2gray(roi)
        roi_resized = cv2.resize(roi_gray, target_size)
        roi_8bit = (roi_resized * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"{base_name}_neg_{idx}.jpg")
        cv2.imwrite(output_path, roi_8bit)


def save_negative_samples_with_licence(
    image: np.ndarray,
    image_shape: tuple[int, int, int],
    car_boxes: list[tuple[int, int, int, int]],
    base_name: str,
    output_dir: str,
    target_size: tuple[int, int],
    multiplier: int = 5,
    licence_boxes: list[tuple[int, int, int, int]] = None  # 可选参数
):
    """
    生成并保存负样本图片。

    :param image: 原始图像数据
    :param image_shape: 图像尺寸 (height, width, channels)
    :param target_boxes: 所有 target 的 bounding boxes
    :param base_name: 图片基础名（不含扩展名）
    :param output_dir: 输出目录
    :param target_size: 目标尺寸
    :param multiplier: 每张图负样本数 = 正样本数 × multiplier
    """
    num_samples = len(target_boxes) * multiplier
    neg_regions = generate_negative_roi_in_car_not_licence(image_shape, car_boxes, licence_boxes,num_samples=num_samples)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(neg_regions):
        roi = image[ymin:ymax, xmin:xmax]
        roi_gray = rgb2gray(roi)
        roi_resized = cv2.resize(roi_gray, target_size)
        roi_8bit = (roi_resized * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"{base_name}_neg_{idx}.jpg")
        cv2.imwrite(output_path, roi_8bit)

# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    print(f"正常执行：{LABEL_NAME}")
    images_list = os.listdir(IMAGES_DIR)

    for img_file in tqdm(images_list, desc="处理图片进度", total=len(images_list)):
        if not img_file.lower().endswith('.jpg'):
            continue

        img_path = os.path.join(IMAGES_DIR, img_file)
        annotation_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.xml'))

        if not os.path.exists(annotation_path):
            print(f"警告：{annotation_path} 不存在，跳过该图片")
            continue

        # 读取图像并转换为 RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = os.path.splitext(img_file)[0]

        # 提取 target 的 bounding box
        target_boxes = extract_boxes(annotation_path,label_name = LABEL_NAME)
        if not target_boxes:
            continue  # 如果没有 target，跳过这张图

        # 保存正样本
        save_positive_samples(image, target_boxes, base_name, output_pos_dir, target_size)

        # 保存负样本
        if LABEL_NAME == 'car':
            save_negative_samples(image, image.shape, target_boxes, base_name, output_neg_dir, target_size, neg_sample_multiplier)
        elif LABEL_NAME == 'licence':
            car_boxes = extract_boxes(annotation_path,label_name = 'car')
            save_negative_samples_with_licence(image, image.shape, car_boxes, base_name, output_neg_dir, target_size, neg_sample_multiplier,target_boxes)
        

    print("预处理完成，正负样本已保存。")