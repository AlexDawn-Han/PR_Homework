import os
import numpy as np
import joblib
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# -----------------------------
# 配置部分
# -----------------------------
DATASET_PATH = r'D:\Datasets\PR-homework\dataset\train'

LABEL_NAME = 'licence' #'car' #'licence'  

if LABEL_NAME == 'car':
    n_components = 1024
    pos_features_path = os.path.join(DATASET_PATH,'task1', 'car_pos_features.npy')
    neg_features_path = os.path.join(DATASET_PATH,'task1', 'car_neg_features.npy')
    pca_model_save_path = 'car_pca_model.joblib'
    label_save_path = os.path.join(DATASET_PATH, 'task1','car_labels_shuffle.npy')
    pac_features_save_path = os.path.join(DATASET_PATH, 'task1','car_pca_features.npy')
    output_file_name = 'car_svm_model.pkl'
elif LABEL_NAME == 'licence':
    n_components = 512
    pos_features_path = os.path.join(DATASET_PATH,'task2', 'licence_pos_features.npy')
    neg_features_path = os.path.join(DATASET_PATH, 'task2','licence_neg_features.npy')
    pca_model_save_path = 'licence_pca_model.joblib'
    label_save_path = os.path.join(DATASET_PATH, 'task2','licence_labels_shuffle.npy')
    pac_features_save_path = os.path.join(DATASET_PATH, 'task2','licence_pca_features.npy')
    output_file_name = 'licence_svm_model.pkl'

# -----------------------------
# 工具函数：检查文件是否存在
# -----------------------------
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到特征文件: {path}")


# -----------------------------
# 数据加载模块
# -----------------------------
def load_features(pos_path, neg_path):
    print("正在加载特征数据...")
    check_file_exists(pos_path)
    check_file_exists(neg_path)

    pos_features = np.load(pos_path)
    neg_features = np.load(neg_path)

    print(f"正样本特征维度: {pos_features.shape}")
    print(f"负样本特征维度: {neg_features.shape}")
    return pos_features, neg_features


def create_dataset(pos_features, neg_features):
    print("正在构建训练集...")
    all_features = np.vstack((pos_features, neg_features))
    labels = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))
    features_shuffle, labels_shuffle = shuffle(all_features, labels, random_state=42)
    return features_shuffle, labels_shuffle


# -----------------------------
# PCA 处理模块
# -----------------------------
def apply_pca(features, n_components_val, save_path=None):
    print("正在应用PCA降维...")
    pca = PCA(n_components=n_components_val)
    reduced_features = pca.fit_transform(features)

    print(f"原始特征维度: {features.shape}")
    print(f"降维后特征维度: {reduced_features.shape}")
    print(f"解释方差比例: {sum(pca.explained_variance_ratio_):.4f}")

    if save_path:
        joblib.dump(pca, save_path)
        print(f"PCA模型已保存至: {save_path}")

    return reduced_features


# -----------------------------
# SVM 模型训练与评估
# -----------------------------
def train_svm(X_train, y_train, X_valid, y_valid,output_file_name='svm_model.pkl'):
    print("开始训练SVM分类器...")

    # 创建SVM模型实例
    svm_model = SVC(kernel='rbf', C=0.8, probability=True)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = svm_model.predict(X_valid)

    # 输出评价指标
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))

    # 保存模型到文件
    joblib.dump(svm_model, output_file_name)
    print(f"SVM模型已保存至: {output_file_name}")


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    print(f"正常执行：{LABEL_NAME}")
    # 1. 加载特征数据
    try:
        pos_features, neg_features = load_features(pos_features_path, neg_features_path)

        # 2. 构建数据集并打乱顺序
        features_shuffle, labels_shuffle = create_dataset(pos_features, neg_features)

        # 3. 应用PCA降维
        pca_features = apply_pca(features_shuffle, n_components, pca_model_save_path)

        # 4. 保存处理后的特征和标签
        np.save(pac_features_save_path, pca_features)
        np.save(label_save_path, labels_shuffle)
        print("特征处理完成，结果已保存。")

        # 5. 划分训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(
            pca_features, labels_shuffle,
            test_size=0.2,
            random_state=42
        )

        # 6. 训练SVM模型
        train_svm(X_train, y_train, X_valid, y_valid,output_file_name)

    except Exception as e:
        print(f"程序异常中断: {e}")