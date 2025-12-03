import os
import glob
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import hog
from skimage.morphology import skeletonize


from xgboost import XGBClassifier

# ===================== 配置区域 =====================
TRAIN_DIR = "dataset-for-task1/train"   # 训练集根目录：子文件夹名为类别
TEST_DIR = "dataset-for-task1/test"     # 测试集图片目录
SUBMISSION_TEMPLATE = "submission-for-task1.csv"  # 提交格式模板
OUTPUT_FILE = "submission.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 每张训练图基础增强次数
AUG_PER_IMAGE = 2
# 对这几个难分类别增加更强增强
HARD_CLASSES = ("Black-grass", "Loose Silky-bent", "Scentless Mayweed")
# 对这两个易混淆的类别进行专用二分类判断
PAIR_CLASSES_0_3 = ("Black-grass", "Scentless Mayweed")

# 是否在训练阶段打印每类图片数
VERBOSE_CLASS_COUNTS = True


# ===================== 工具函数 =====================

def load_submission_template(path: str):
    """
    读取提交模板，并自动检测 ID 列（用于拼接文件名）。
    规则：优先寻找列名中包含 'id' 字样（不区分大小写），否则使用第1列。
    """
    sub = pd.read_csv(path)
    id_col = None
    for c in sub.columns:
        if "id" in c.lower():
            id_col = c
            break
    if id_col is None:
        id_col = sub.columns[0]
    return sub, id_col


def segment_leaf(img_bgr: np.ndarray) -> np.ndarray:
    """
    对 BGR 图像进行简单的叶片分割：
    1. HSV 绿色阈值
    2. 形态学开闭去噪
    3. 只保留最大连通域
    返回二值 mask（255 为叶片区域）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 绿色+偏黄叶片，大致范围，可按数据微调
    lower = np.array([25, 30, 20])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # 忽略背景（0），只看 1..n
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + int(np.argmax(areas))
        mask_clean = np.zeros_like(mask)
        mask_clean[labels == max_id] = 255
        mask = mask_clean

    return mask

def compute_skeleton_features(mask: np.ndarray) -> np.ndarray:
    """
    根据二值 mask 计算骨架特征：
      - length_norm: 骨架长度 / 图像面积
      - branch_density: 分叉点个数 / 骨架长度
      - end_density: 端点个数 / 骨架长度
    mask: 0/255 的单通道图像
    返回: (3,) float32 向量
    """
    # 二值化
    bin_mask = mask > 0
    if bin_mask.sum() == 0:
        return np.zeros(3, dtype=np.float32)

    # skeletonize 返回 bool 数组
    skel = skeletonize(bin_mask)
    length = float(skel.sum())
    if length == 0:
        return np.zeros(3, dtype=np.float32)

    # 统计每个骨架像素的 8 邻居数
    skel_uint8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neigh = cv2.filter2D(skel_uint8, -1, kernel)

    # 邻居数 = 3x3 中非零数量 - 自己
    neighbors = neigh - skel_uint8

    # 分叉点: 邻居数 >= 3
    branch_points = np.logical_and(skel, neighbors >= 3).sum()
    # 端点: 邻居数 == 1
    end_points = np.logical_and(skel, neighbors == 1).sum()

    h, w = mask.shape[:2]
    area = float(h * w)

    length_norm = length / (area + 1e-6)
    branch_density = branch_points / (length + 1e-6)
    end_density = end_points / (length + 1e-6)

    return np.array([length_norm, branch_density, end_density], dtype=np.float32)


def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    从 BGR 图像提取特征：
      - 只对叶片区域进行 256x256 resize 后：
        * HSV 颜色直方图 (3*32 维)
        * GLCM 纹理特征 (5 维)
        * Hu 矩 (7 维，经 log 变换)
        * 形状特征：aspect_ratio, extent, solidity (3 维)
    返回一个 1D feature 向量。
    """
    if img_bgr is None:
        return None

    img = cv2.resize(img_bgr, (256, 256))

    # 叶片分割
    mask = segment_leaf(img)
    masked = cv2.bitwise_and(img, img, mask=mask)

    # ---------- 颜色特征（多颜色空间）----------
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [32], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    color_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    
    # LAB 颜色空间直方图（对颜色感知更符合人眼）
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab], [0], mask, [16], [0, 256])
    hist_a = cv2.calcHist([lab], [1], mask, [16], [0, 256])
    hist_b = cv2.calcHist([lab], [2], mask, [16], [0, 256])
    cv2.normalize(hist_l, hist_l)
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    lab_features = np.concatenate([hist_l, hist_a, hist_b]).flatten()
    
    # YCrCb 颜色空间直方图（对纹理边界敏感）
    ycrcb = cv2.cvtColor(masked, cv2.COLOR_BGR2YCrCb)
    hist_y = cv2.calcHist([ycrcb], [0], mask, [16], [0, 256])
    hist_cr = cv2.calcHist([ycrcb], [1], mask, [16], [0, 256])
    hist_cb = cv2.calcHist([ycrcb], [2], mask, [16], [0, 256])
    cv2.normalize(hist_y, hist_y)
    cv2.normalize(hist_cr, hist_cr)
    cv2.normalize(hist_cb, hist_cb)
    ycrcb_features = np.concatenate([hist_y, hist_cr, hist_cb]).flatten()
    
    color_features = np.concatenate([color_features, lab_features, ycrcb_features])

    # ---------- 纹理 & Hu 矩 ----------
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    texture_features = np.array(
        [contrast, dissimilarity, homogeneity, energy, correlation],
        dtype=np.float32
    )

    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # log 变换，使数值尺度更适合建模
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # ---------- SIFT 特征统计 ----------
    sift_desc_mean = np.zeros(128, dtype=np.float32)
    sift_desc_std = np.zeros(128, dtype=np.float32)
    sift_kp_count = 0
    try:
        if hasattr(cv2, 'SIFT_create'):
            sift = cv2.SIFT_create()
        else:
            sift = cv2.xfeatures2d.SIFT_create()
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is not None and len(desc) > 0:
            sift_kp_count = len(kps)
            sift_desc_mean = np.mean(desc, axis=0)
            sift_desc_std = np.std(desc, axis=0)
    except Exception:
        # 若 SIFT 不可用，保持零向量
        sift_desc_mean = np.zeros(128, dtype=np.float32)
        sift_desc_std = np.zeros(128, dtype=np.float32)
        sift_kp_count = 0

    sift_features = np.concatenate([sift_desc_mean, sift_desc_std, np.array([sift_kp_count], dtype=np.float32)])

    # ---------- LBP 特征 ----------
    lbp_P = 8
    lbp_R = 1
    lbp = local_binary_pattern(gray, lbp_P, lbp_R, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_P + 3), range=(0, lbp_P + 2))
    if lbp_hist.sum() != 0:
        lbp_hist = lbp_hist.astype('float32') / lbp_hist.sum()
    lbp_features = lbp_hist.astype(np.float32)

    # ---------- 简单 Haar-like 特征 ----------
    ii = cv2.integral(gray)

    def rect_sum(integ, x1, y1, x2, y2):
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        return integ[y2i, x2i] - integ[y1i, x2i] - integ[y2i, x1i] + integ[y1i, x1i]

    h, w = gray.shape[:2]
    haar_vals = []
    for scale in [0.5, 0.3]:
        rw = int(w * scale)
        rh = int(h * 0.2)
        x = (w - rw) // 2
        y = (h - rh) // 2
        left = rect_sum(ii, x, y, x + rw // 2, y + rh)
        right = rect_sum(ii, x + rw // 2, y, x + rw, y + rh)
        haar_vals.append((left - right) / max(1, rw * rh))
    for scale in [0.5, 0.3]:
        rh = int(h * scale)
        rw = int(w * 0.2)
        x = (w - rw) // 2
        y = (h - rh) // 2
        top = rect_sum(ii, x, y, x + rw, y + rh // 2)
        bottom = rect_sum(ii, x, y + rh // 2, x + rw, y + rh)
        haar_vals.append((top - bottom) / max(1, rw * rh))
    cx, cy = w // 2, h // 2
    sz = int(min(w, h) * 0.25)
    x1, y1 = cx - sz, cy - sz
    x2, y2 = cx + sz, cy + sz
    q1 = rect_sum(ii, x1, y1, cx, cy)
    q2 = rect_sum(ii, cx, y1, x2, cy)
    q3 = rect_sum(ii, x1, cy, cx, y2)
    q4 = rect_sum(ii, cx, cy, x2, y2)
    haar_vals.append(((q1 + q4) - (q2 + q3)) / max(1, (x2 - x1) * (y2 - y1)))

    haar_features = np.array(haar_vals, dtype=np.float32)

    # ---------- 形状特征 ----------
    shape_features = np.zeros(3, dtype=np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / (w + 1e-6)

        cnt_area = cv2.contourArea(cnt)
        rect_area = w * h + 1e-6
        extent = cnt_area / rect_area

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = cnt_area / hull_area

        shape_features = np.array(
            [aspect_ratio, extent, solidity], dtype=np.float32
        )

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    skeleton_feats = compute_skeleton_features(mask)

    # ---------- 增强的纹理特征（多尺度 LBP）----------
    # 除了当前的 LBP，再计算一个不同参数的 LBP
    lbp_P2 = 16
    lbp_R2 = 2
    lbp2 = local_binary_pattern(gray, lbp_P2, lbp_R2, method='uniform')
    (lbp_hist2, _) = np.histogram(lbp2.ravel(), bins=np.arange(0, lbp_P2 + 3), range=(0, lbp_P2 + 2))
    if lbp_hist2.sum() != 0:
        lbp_hist2 = lbp_hist2.astype('float32') / lbp_hist2.sum()
    lbp_features2 = lbp_hist2.astype(np.float32)
    
    # 合并两个 LBP
    lbp_features_multi = np.concatenate([lbp_features, lbp_features2])

    # ---------- 合并所有特征 ----------
    global_features = np.concatenate([
        color_features,
        texture_features,
        hu_moments,
        shape_features,
        hog_feat,
        skeleton_feats,
        sift_features,
        lbp_features_multi,
        haar_features
    ])

    return global_features


def extract_features(image_path: str) -> np.ndarray:
    """从路径读取图像并提取特征的包装函数。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None
    return extract_features_from_image(img)


def augment_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    简单几何增强：
      - 随机水平翻转
      - 随机小角度旋转
    """
    aug = img_bgr.copy()

    # 水平翻转
    if np.random.rand() < 0.5:
        aug = cv2.flip(aug, 1)

    # 小角度旋转
    angle = float(np.random.uniform(-15, 15))
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return aug


# ===================== 主流程 =====================

def main():
    # ---------- 1. 收集所有原始图片路径 + 标签 ----------
    print("收集训练图片路径与标签...")
    class_names = [
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]
    class_names = sorted(class_names)

    if VERBOSE_CLASS_COUNTS:
        print("检测到的类别：", class_names)

    all_paths = []
    all_labels = []

    for cls in class_names:
        species_dir = os.path.join(TRAIN_DIR, cls)
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_files.extend(glob.glob(os.path.join(species_dir, ext)))
        if VERBOSE_CLASS_COUNTS:
            print(f"  类别 {cls}: {len(image_files)} 张图片")
        for p in image_files:
            all_paths.append(p)
            all_labels.append(cls)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # ---------- 2. 按路径划分 train/val（然后只对 train 做增强） ----------
    X_paths_train, X_paths_val, y_train_paths, y_val_paths = train_test_split(
        all_paths,
        all_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    print(f"训练集原始样本数: {len(X_paths_train)}, 验证集样本数: {len(X_paths_val)}")

    # ---------- 3. 提取训练集特征（原图 + 增强） ----------
    train_features = []
    train_labels = []

    # 为二分类器（Black-grass vs Loose Silky-bent）保留一份“未增强特征”
    raw_train_features = []
    raw_train_labels = []

    print("对训练集提取特征（含数据增强）...")
    for path, label in zip(X_paths_train, y_train_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read train image: {path}")
            continue

        # 原始图特征
        feat = extract_features_from_image(img)
        if feat is None:
            continue

        train_features.append(feat)
        train_labels.append(label)

        # 记录一份原始特征，用于后续二分类器
        raw_train_features.append(feat)
        raw_train_labels.append(label)

        # 数据增强次数（难类多增强）
        if label in HARD_CLASSES:
            n_aug = AUG_PER_IMAGE + 2
        else:
            n_aug = AUG_PER_IMAGE

        for _ in range(n_aug):
            aug = augment_image(img)
            feat_aug = extract_features_from_image(aug)
            if feat_aug is not None:
                train_features.append(feat_aug)
                train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    raw_train_features = np.array(raw_train_features)
    raw_train_labels = np.array(raw_train_labels)

    # ---------- 4. 提取验证集特征（不做增强） ----------
    print("对验证集提取特征（不增强）...")
    val_features = []
    val_labels = []

    for path, label in zip(X_paths_val, y_val_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read val image: {path}")
            continue
        feat = extract_features_from_image(img)
        if feat is not None:
            val_features.append(feat)
            val_labels.append(label)

    val_features = np.array(val_features)
    val_labels = np.array(val_labels)

    print(f"训练特征形状: {train_features.shape}, 验证特征形状: {val_features.shape}")

    # ---------- 5. 标签编码 & 标准化 ----------
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train_labels)
    y_val_encoded = le.transform(val_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_val_scaled = scaler.transform(val_features)
    X_train_raw_scaled = scaler.transform(raw_train_features)  # 给二分类器用

    num_classes = len(le.classes_)
    print("类别编码：", dict(zip(le.classes_, range(num_classes))))

    # ---------- 6. 训练主模型：XGBoost 多分类 ----------
    print("训练 XGBoost 主分类器（5 类）...")
    xgb_main = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=2000,        # 上限较大，配合 early stopping
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=50,
    )

    xgb_main.fit(
        X_train_scaled,
        y_train_encoded,
        eval_set=[(X_val_scaled, y_val_encoded)],

        verbose=False,
    )

    # 在验证集上评估
    val_pred_main = xgb_main.predict(X_val_scaled)
    print("\n主模型在验证集上的表现：")
    print(classification_report(y_val_encoded, val_pred_main, target_names=le.classes_))
    print("验证集准确率:", accuracy_score(y_val_encoded, val_pred_main))

    # ---------- 7. 训练二级裁判：只分 Black-grass vs Loose Silky-bent ----------
    print("\n训练 Black-grass vs Loose Silky-bent 二分类器...")
    mask_pair = np.isin(raw_train_labels, HARD_CLASSES)
    X_pair = X_train_raw_scaled[mask_pair]
    y_pair_str = raw_train_labels[mask_pair]

    if X_pair.shape[0] == 0:
        print("[WARN] 没有找到 Black-grass / Loose Silky-bent 训练样本，二级分类器将被跳过。")
        pair_clf = None
    else:
        # 定义：0 = Black-grass, 1 = Loose Silky-bent
        y_pair = (y_pair_str == HARD_CLASSES[1]).astype(int)

        pair_clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        pair_clf.fit(X_pair, y_pair)
        print(f"二分类器训练样本数: {X_pair.shape[0]}")

    from sklearn.metrics import confusion_matrix

    print("\n在验证集上测试两级裁判效果...")

    # 主模型在验证集上的概率
    proba_val_main = xgb_main.predict_proba(X_val_scaled)

    # ---------- 7b. 训练二级裁判：只分 Black-grass vs Scentless Mayweed ----------
    print("\n训练 Black-grass vs Scentless Mayweed 二分类器...")
    mask_pair_0_3 = np.isin(raw_train_labels, PAIR_CLASSES_0_3)
    X_pair_0_3 = X_train_raw_scaled[mask_pair_0_3]
    y_pair_0_3_str = raw_train_labels[mask_pair_0_3]

    pair_clf_0_3 = None
    if X_pair_0_3.shape[0] == 0:
        print("[WARN] 没有找到 Black-grass / Scentless Mayweed 训练样本，0-3 二分类器将被跳过。")
    else:
        # 定义：0 = Black-grass, 1 = Scentless Mayweed
        y_pair_0_3 = (y_pair_0_3_str == PAIR_CLASSES_0_3[1]).astype(int)

        pair_clf_0_3 = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        pair_clf_0_3.fit(X_pair_0_3, y_pair_0_3)
        print(f"0-3 二分类器训练样本数: {X_pair_0_3.shape[0]}")

    print("\n在验证集上测试两级裁判效果...")

    # 主模型在验证集上的概率
    proba_val_main = xgb_main.predict_proba(X_val_scaled)

    final_val_labels = []

    for i in range(len(proba_val_main)):
        p = proba_val_main[i]
        sorted_idx = np.argsort(p)[::-1]
        top1, top2 = sorted_idx[0], sorted_idx[1]
        labels_top2 = le.inverse_transform([top1, top2])

        # 默认是主模型的 top1
        final_label = le.inverse_transform([top1])[0]

        # 先检查是否触发 0-3 二分类器
        if pair_clf_0_3 is not None and set(labels_top2) == set(PAIR_CLASSES_0_3):
            diff_0_3 = abs(p[le.transform([PAIR_CLASSES_0_3[0]])[0]] - p[le.transform([PAIR_CLASSES_0_3[1]])[0]])
            if diff_0_3 < 0.12:  # 阈值更严格，0 和 3 更接近时才用二分类器
                pair_pred_0_3 = pair_clf_0_3.predict(X_val_scaled[i:i + 1])[0]
                final_label = PAIR_CLASSES_0_3[int(pair_pred_0_3)]
        # 再检查是否触发 0-2 二分类器
        elif pair_clf is not None and set(labels_top2) == set(HARD_CLASSES[:2]):
            diff = abs(p[top1] - p[top2])
            if diff < 0.15:
                pair_pred = pair_clf.predict(X_val_scaled[i:i + 1])[0]
                final_label = HARD_CLASSES[:2][int(pair_pred)]

        final_val_labels.append(final_label)

    final_val_labels = np.array(final_val_labels)

    y_val_true = val_labels  # 这是原始字符串标签
    acc_two_stage = accuracy_score(y_val_true, final_val_labels)
    print("两级裁判后的验证集准确率:", acc_two_stage)

    print("\n两级裁判后的分类报告:")
    print(classification_report(
        y_val_true,
        final_val_labels,
        target_names=le.classes_
    ))

    print("\n两级裁判后的混淆矩阵:")
    print(confusion_matrix(y_val_true, final_val_labels, labels=le.classes_))

    # ---------- 8. 处理测试集 ----------
    print("\n加载测试集并提取特征...")
    sub_template, id_col = load_submission_template(SUBMISSION_TEMPLATE)

    test_features = []
    test_ids = []

    for _, row in sub_template.iterrows():
        image_id = str(row[id_col])
        img_path = os.path.join(TEST_DIR, image_id)
        if not os.path.exists(img_path):
            print(f"[WARN] Test image not found: {img_path}")
            # 可以尝试自动补扩展名，这里先简单警告
            continue

        feat = extract_features(img_path)
        if feat is not None:
            test_features.append(feat)
            test_ids.append(image_id)
        else:
            print(f"[WARN] Failed to extract features for test image: {image_id}")

    if len(test_features) == 0:
        raise RuntimeError("没有成功提取任何测试集特征，请检查测试路径和文件名。")

    X_test = np.array(test_features)
    X_test_scaled = scaler.transform(X_test)

    # ---------- 9. 两级判定：主分类器 + 二级裁判 ----------
    print("\n对测试集进行预测（带二级裁判）...")
    proba_main = xgb_main.predict_proba(X_test_scaled)
    final_labels = []

    for i in range(len(proba_main)):
        p = proba_main[i]
        # 找 top2 索引（从大到小排序）
        sorted_idx = np.argsort(p)[::-1]
        top1, top2 = sorted_idx[0], sorted_idx[1]
        labels_top2 = le.inverse_transform([top1, top2])

        # 默认用主模型 top1
        final_label = le.inverse_transform([top1])[0]

        # 先检查是否触发 0-3 二分类器
        if pair_clf_0_3 is not None and set(labels_top2) == set(PAIR_CLASSES_0_3):
            idx_0 = le.transform([PAIR_CLASSES_0_3[0]])[0]
            idx_3 = le.transform([PAIR_CLASSES_0_3[1]])[0]
            diff_0_3 = abs(p[idx_0] - p[idx_3])
            if diff_0_3 < 0.12:  # 阈值更严格
                pair_pred_0_3 = pair_clf_0_3.predict(X_test_scaled[i:i + 1])[0]
                final_label = PAIR_CLASSES_0_3[int(pair_pred_0_3)]
        # 再检查是否触发 0-2 二分类器
        elif pair_clf is not None and set(labels_top2) == set(HARD_CLASSES[:2]):
            diff = abs(p[top1] - p[top2])
            if diff < 0.15:
                pair_pred = pair_clf.predict(X_test_scaled[i:i + 1])[0]
                final_label = HARD_CLASSES[:2][int(pair_pred)]

        final_labels.append(final_label)

    final_labels = np.array(final_labels)

    # ---------- 10. 生成提交文件 ----------
    print("\n生成提交文件...")

    # test_ids 里保存的是所有测试图片的文件名（ID）
    submission = pd.DataFrame({
        "ID": test_ids,
        "Category": final_labels
    })

    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] 格式正确的提交文件已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
