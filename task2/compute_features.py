import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 设置路径
dataset_path = 'D:\\Datasets\\PR-homework\\dataset\\train\\'

LABEL_NAME = 'licence'#'car' # 'licence'

if LABEL_NAME == 'car':
    pos_dataset_path = os.path.join(dataset_path, 'task1','car_pos_1')
    neg_dataset_path = os.path.join(dataset_path, 'task1','car_neg_n')
    output_pos_file_name = os.path.join(dataset_path,'task1','car_pos_features.npy')
    output_neg_file_name = os.path.join(dataset_path,'task1','car_neg_features.npy')
    # HOG特征提取参数
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (20, 20),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'visualize': False
    }
elif LABEL_NAME == 'licence':
    pos_dataset_path = os.path.join(dataset_path, 'task2','licence_pos_1')
    neg_dataset_path = os.path.join(dataset_path, 'task2','licence_neg_n')
    output_pos_file_name = os.path.join(dataset_path,'task2','licence_pos_features.npy')
    output_neg_file_name = os.path.join(dataset_path,'task2','licence_neg_features.npy')
    # HOG特征提取参数
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (10, 3),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'visualize': False
    }



def process_image(img_file, dataset_dir):
    """处理单个图像文件，提取HOG特征"""
    if not img_file.lower().endswith('.jpg'):
        return None
    
    img_path = os.path.join(dataset_dir, img_file)
    try:
        image = cv2.imread(img_path, 0)  # 使用灰度模式读取图片
        if image is None:
            print(f"Error reading image {img_path}")
            return None
        
        feature = hog(image, **hog_params)
        return feature
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def helper_process_image(args):
    """辅助函数，用于并行处理"""
    img_file, dataset_dir = args
    return process_image(img_file, dataset_dir)

def img2feature_parallel(images_dir, dataset_dir):
    """并行提取指定目录下所有图像的HOG特征"""
    images = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    image_args = [(img, dataset_dir) for img in images]  # 创建包含图像文件和数据集目录的元组列表
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(helper_process_image, image_args), total=len(image_args), desc="Processing images", unit="image"))
    
    # 过滤None值
    features = [feature for feature in results if feature is not None]
    return np.array(features)

# 主程序逻辑
if __name__ == "__main__":
    print(f"正常执行：{LABEL_NAME}")
    # 提取正样本特征
    pos_features = img2feature_parallel(pos_dataset_path, pos_dataset_path)
    np.save(output_pos_file_name, pos_features)
    
    # 提取负样本特征
    neg_features = img2feature_parallel(neg_dataset_path, neg_dataset_path)
    np.save(output_neg_file_name, neg_features)
    