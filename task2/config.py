import os

# ----------------------------
# 配置参数
# ----------------------------
DATASET_PATH = 'D:\\Datasets\\PR-homework\\dataset\\test\\'
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels')

SCALE_PERCENT = 20
TOP_K = 1

car_pca_model_path = 'car_pca_model.joblib'
car_svm_model_path = 'car_svm_model.pkl'
car_target_size = (200, 200)
car_min_width, car_min_height = 400, 400
# car_HOG特征提取参数
car_hog_params = {
    'orientations': 9,
    'pixels_per_cell': (20, 20),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False
}


licence_pca_model_path = 'licence_pca_model.joblib'
licence_svm_model_path = 'licence_svm_model.pkl'
licence_target_size = (100, 30)
licence_max_width, licence_max_height = 300, 120
licence_min_width, licence_min_height = 60, 30
# licence_HOG特征提取参数
licence_hog_params = {
    'orientations': 9,
    'pixels_per_cell': (10, 3),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'visualize': False
}

# 保存结果的目录
result_dir = os.path.join(DATASET_PATH, 'task2','results')  
# 日志文件路径
log_file = os.path.join('evaluation_log.txt')
