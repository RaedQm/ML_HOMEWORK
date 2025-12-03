import cv2
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops

# ================= 配置路径 =================
# 请根据您实际的数据存放路径修改这里
TRAIN_DIR = 'dataset-for-task1/train'  # 训练集根目录，下面应有5个子文件夹
TEST_DIR = 'dataset-for-task1/test'  # 测试集图片目录
TEST_CSV = 'test.csv'  # 测试集列表文件
SUBMISSION_TEMPLATE = 'submission-for-task1.csv'  # 提交格式模板
OUTPUT_FILE = 'submission.csv'


# ================= 1. 特征提取函数 =================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None

    # 统一调整大小，减少计算量并标准化
    img = cv2.resize(img, (256, 256))

    # --- A. 颜色特征 (Color Histogram) ---
    # 转换到 HSV 空间，更能反映植物颜色特征
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 计算 H, S, V 三个通道的直方图，使用更多箱以捕捉细节
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    # 归一化
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    color_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    # --- B. 纹理特征 (GLCM & Hu Moments) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GLCM (灰度共生矩阵) 特征
    # 计算距离为1，角度为0, 45, 90, 135度的GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])

    # Hu 矩 (形状特征)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # 对数变换处理 Hu 矩使其数值范围更合理
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # 合并所有特征
    global_features = np.concatenate([color_features, texture_features, hu_moments])
    return global_features


# ================= 2. 加载训练数据 =================
print("开始加载训练数据并提取特征...")
train_features = []
train_labels = []
train_image_paths = []

# 假设 train 目录下有以类别命名的子文件夹
# 获取所有子文件夹名称作为类别
classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
print(f"检测到的类别: {classes}")

for species in classes:
    species_dir = os.path.join(TRAIN_DIR, species)
    # 支持 png 和 jpg
    image_files = glob.glob(os.path.join(species_dir, '*.png')) + glob.glob(os.path.join(species_dir, '*.jpg'))

    print(f"正在处理类别: {species} ({len(image_files)} 张图片)")
    for file_path in image_files:
        feat = extract_features(file_path)
        if feat is not None:
            train_features.append(feat)
            train_labels.append(species)

X = np.array(train_features)
y = np.array(train_labels)

# 标签编码 (String -> Int)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 数据标准化 (StandardScaler) - 对 SVM 非常重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"特征提取完成。训练集形状: {X_scaled.shape}")

# ================= 3. 模型训练 =================
print("开始训练 SVM 分类器...")

# 划分验证集以评估性能（使用分层抽样以保持类别分布）
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 使用网格搜索寻找最佳参数 (C, Gamma)，并考虑类别不平衡
# - 扩展 C 和 gamma 值范围
# - 使用 class_weight='balanced' 以减轻类别不平衡
# - 使用更稳健的评分指标（macro F1）和更高的 CV 折数
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.01, 0.001, 0.0001, 'scale'],
    'kernel': ['rbf']
}
grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),
    param_grid,
    refit=True,
    verbose=2,
    cv=5,
    n_jobs=-1,
    scoring='f1_macro'
)
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")

# 在验证集上评估
val_predictions = grid.predict(X_val)
print("验证集分类报告:")
print(classification_report(y_val, val_predictions, target_names=le.classes_))

# ================= 4. 对测试集进行预测 =================
print("开始处理测试集...")

# 读取提交模板或 test.csv 获取文件列表
try:
    if os.path.exists(TEST_CSV):
        test_df = pd.read_csv(TEST_CSV)
    else:
        print(f"未找到 {TEST_CSV}，尝试使用 {SUBMISSION_TEMPLATE}")
        test_df = pd.read_csv(SUBMISSION_TEMPLATE)
except FileNotFoundError:
    print("错误：找不到测试列表文件 (test.csv 或 submission-for-task1.csv)")
    exit()

test_features = []
valid_ids = []

for idx, row in test_df.iterrows():
    img_name = row['ID']
    img_path = os.path.join(TEST_DIR, img_name)

    feat = extract_features(img_path)
    if feat is not None:
        test_features.append(feat)
        valid_ids.append(img_name)
    else:
        # 如果图片读取失败，填入零向量或处理异常
        print(f"无法读取测试图片: {img_name}")
        # 这里为了保证行数一致，可以添加一个全0特征，或者记录下来
        # 简单起见，假设所有图片都存在。如有缺失需额外处理。
        test_features.append(np.zeros(X.shape[1]))

X_test = np.array(test_features)
X_test_scaled = scaler.transform(X_test)  # 使用同样的 Scaler 转换测试集

# 预测
predictions_encoded = grid.predict(X_test_scaled)
predictions_labels = le.inverse_transform(predictions_encoded)

# ================= 5. 生成提交文件 =================
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Category': predictions_labels
})

submission.to_csv(OUTPUT_FILE, index=False)
print(f"预测完成，结果已保存至 {OUTPUT_FILE}")