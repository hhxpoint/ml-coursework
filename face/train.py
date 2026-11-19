




"""
PCA + Fisher LDA 人脸识别实验（强化版）
完整优化：
  - 使用 HOG 特征（对光照/姿态鲁棒）
  - Linear SVM 分类器
  - 旋转、亮度调整等数据增强
  - 标准化 + 收缩 LDA
  - 丰富可视化图表保存至 results/figures/
环境：Python 3.8+, opencv, numpy, scikit-learn, scikit-image, matplotlib
"""

import os
import cv2
import numpy as np
import json
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_test1")   # 预处理后的数据
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

IMAGE_SIZE = (64, 64)
TRAIN_RATIO = 0.7              # 每类训练比例
PCA_COMPONENTS = 100           # PCA 保留维度
RANDOM_SEED = 42
USE_AUGMENTATION = True        # 是否使用数据增强
USE_HOG = True                 # 是否使用 HOG 特征（强烈推荐）

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# ==================== 工具函数 ====================
def save_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [Fig] {filename}")

# ==================== 特征提取 ====================
def extract_hog(img_gray):
    """
    提取 HOG 特征，参数优化后维度约 1568
    """
    fd = hog(img_gray,
             orientations=9,
             pixels_per_cell=(8, 8),
             cells_per_block=(2, 2),
             block_norm='L2-Hys',
             visualize=False)
    return fd.astype(np.float32)

# ==================== 数据加载 ====================
def load_dataset():
    """
    加载预处理后的人脸图像，提取 HOG 特征（若开启）或使用原始像素
    """
    X, Y, person_names = [], [], []
    person_folders = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])
    for label, person_name in enumerate(person_folders):
        person_dir = os.path.join(PROCESSED_DIR, person_name)
        files = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        for f in files:
            img = cv2.imread(os.path.join(person_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != IMAGE_SIZE:
                img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            # CLAHE 光照增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            # 提取特征
            if USE_HOG:
                feat = extract_hog(img)
            else:
                # 原始像素，归一化到 [0,1]
                feat = img.astype(np.float32) / 255.0
                feat = feat.flatten()
            X.append(feat)
            Y.append(label)
        person_names.append(person_name)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    print(f"  特征维度: {X.shape[1]}")
    return X, Y, person_names

# ==================== 数据划分与增强 ====================
def split_data(X, Y, train_ratio=TRAIN_RATIO, augment=USE_AUGMENTATION):
    """
    随机比例划分训练/测试集，并对训练集做多种数据增强。
    """
    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []
    for label in np.unique(Y):
        idx = np.where(Y == label)[0]
        n = len(idx)
        n_train = max(1, int(n * train_ratio))
        shuffled = np.random.permutation(idx)
        train_idx = shuffled[:n_train]
        test_idx = shuffled[n_train:]
        X_train_list.append(X[train_idx])
        Y_train_list.append(Y[train_idx])
        X_test_list.append(X[test_idx])
        Y_test_list.append(Y[test_idx])
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)

    if augment:
        X_aug, Y_aug = [], []
        for x_vec, label in zip(X_train, Y_train):
            if USE_HOG:
                # HOG 特征无法逆向几何变换，加入高斯噪声模拟光照/微小变化
                noise = np.random.normal(0, 0.02, x_vec.shape).astype(np.float32)
                X_aug.append(x_vec + noise)
                Y_aug.append(label)
                X_aug.append(x_vec - noise)
                Y_aug.append(label)
            else:
                # 原始像素：翻转、旋转、亮度微调
                img = x_vec.reshape(IMAGE_SIZE)
                # 水平翻转
                flipped = cv2.flip(img, 1)
                X_aug.append(flipped.flatten())
                Y_aug.append(label)
                # 旋转 ±5°
                rows, cols = IMAGE_SIZE
                for angle in [-5, 5]:
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    rotated = cv2.warpAffine(img, M, (cols, rows))
                    X_aug.append(rotated.flatten())
                    Y_aug.append(label)
                # 亮度微调
                for beta in [-15, 15]:
                    bright = np.clip(img * 255.0 + beta, 0, 255).astype(np.uint8) / 255.0
                    X_aug.append(bright.flatten())
                    Y_aug.append(label)
        if X_aug:
            X_aug = np.array(X_aug, dtype=np.float32)
            Y_aug = np.array(Y_aug, dtype=np.int32)
            X_train = np.vstack([X_train, X_aug])
            Y_train = np.concatenate([Y_train, Y_aug])
            print(f"  数据增强后训练集大小: {len(X_train)}")

    return X_train, X_test, Y_train, Y_test

# ==================== PCA+LDA+SVM Pipeline ====================
class PCALDASVMPipeline:
    def __init__(self, n_pca=100, standardize=True):
        self.n_pca = n_pca
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.lda = None
        self.svm = SVC(kernel='linear', C=1.0, probability=False)

    def fit(self, X_train, Y_train):
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(Y_train))

        # 标准化
        if self.standardize:
            self.scaler = StandardScaler().fit(X_train)
            X_proc = self.scaler.transform(X_train)
        else:
            X_proc = X_train

        # PCA
        actual_pca = min(self.n_pca, n_samples - 1, n_features)
        actual_pca = max(actual_pca, n_classes)
        self.actual_pca = actual_pca
        self.pca = PCA(n_components=actual_pca, svd_solver='arpack', random_state=RANDOM_SEED)
        X_pca = self.pca.fit_transform(X_proc)

        # LDA ('eigen' 支持 transform)
        n_lda = min(n_classes - 1, actual_pca)
        self.n_lda = n_lda
        self.lda = LDA(n_components=n_lda, solver='eigen', shrinkage='auto')
        X_lda = self.lda.fit_transform(X_pca, Y_train)

        # 训练 SVM
        self.svm.fit(X_lda, Y_train)

        # 保存中间结果（用于可视化）
        self.X_train_pca = X_pca
        self.X_train_lda = X_lda
        self.Y_train = Y_train
        self.n_classes = n_classes
        self.mean_face = np.mean(X_train, axis=0) if not USE_HOG else None
        return self

    def predict(self, X):
        if self.standardize:
            X = self.scaler.transform(X)
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        return self.svm.predict(X_lda)

    def pca_reconstruct(self, X, k=None):
        if USE_HOG:
            raise NotImplementedError("HOG 特征不支持像素重建")
        if k is None:
            k = self.actual_pca
        X_proc = self.scaler.transform(X) if self.standardize else X
        X_pca = self.pca.transform(X_proc)
        X_recon_pca = X_pca[:, :k] @ self.pca.components_[:k, :]
        if self.standardize:
            X_recon_pca = self.scaler.inverse_transform(X_recon_pca)
        return X_recon_pca

    def pca_lda_reconstruct(self, X):
        if USE_HOG:
            raise NotImplementedError
        X_proc = self.scaler.transform(X) if self.standardize else X
        X_pca = self.pca.transform(X_proc)
        X_lda = self.lda.transform(X_pca)
        W_lda = self.lda.scalings_[:, :self.n_lda]
        W_lda_pinv = np.linalg.pinv(W_lda)
        X_pca_recon = X_lda @ W_lda_pinv
        X_recon_proc = X_pca_recon @ self.pca.components_
        if self.standardize:
            X_recon = self.scaler.inverse_transform(X_recon_proc)
        else:
            X_recon = X_recon_proc
        return X_recon

# ==================== 评估指标 ====================
def compute_mse_psnr(orig, recon):
    orig_255 = (orig * 255.0).clip(0, 255)
    recon_255 = (recon * 255.0).clip(0, 255)
    mse = np.mean((orig_255 - recon_255) ** 2)
    if mse == 0:
        return 0.0, float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return float(mse), float(psnr)

# ==================== 可视化函数 ====================
def plot_sample_faces(X, Y, names, title, filename, n_persons=6, n_per=3):
    if USE_HOG:
        print("  [跳过] 使用 HOG 特征，无法显示原始人脸。")
        return
    fig = plt.figure(figsize=(3*n_per, 2*n_persons))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(n_persons, n_per, figure=fig, hspace=0.3, wspace=0.1)
    for p in range(min(n_persons, len(names))):
        mask = Y == p
        imgs = X[mask][:n_per]
        for i, img_vec in enumerate(imgs):
            ax = fig.add_subplot(gs[p, i])
            ax.imshow(img_vec.reshape(IMAGE_SIZE), cmap='gray')
            ax.set_title(f"{names[p]} #{i+1}", fontsize=8)
            ax.axis('off')
    save_fig(filename)

def plot_mean_and_eigenfaces(mean_face, pca, filename):
    if USE_HOG or mean_face is None:
        print("  [跳过] 无法可视化特征脸（HOG特征）。")
        return
    n_show = min(20, len(pca.components_))
    rows, cols = 4, 6
    fig = plt.figure(figsize=(cols*2.5, rows*2.5))
    fig.suptitle("Mean Face & Eigenfaces", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mean_face.reshape(IMAGE_SIZE), cmap='gray')
    ax0.set_title("Mean Face", fontsize=9)
    ax0.axis('off')
    for i in range(n_show):
        row = (i + 1) // cols
        col = (i + 1) % cols
        if row >= rows:
            break
        ef = pca.components_[i].reshape(IMAGE_SIZE)
        ef_norm = cv2.normalize(ef, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(ef_norm, cmap='gray')
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=8)
        ax.axis('off')
    save_fig(filename)

def plot_pca_variance(pca, filename):
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    n_show = min(40, len(pca.explained_variance_ratio_))
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(cum_var)+1), cum_var, 'b-', linewidth=2)
    ax1.axhline(y=90, color='r', linestyle='--')
    ax1.axhline(y=95, color='orange', linestyle='--')
    ax1.set_xlabel('Number of PCs')
    ax1.set_ylabel('Cumulative Variance (%)')
    ax1.set_title('Cumulative Explained Variance')
    ax1.grid(True)
    ax2 = fig.add_subplot(gs[0, 1])
    individual = pca.explained_variance_ratio_[:n_show] * 100
    ax2.bar(range(1, n_show+1), individual, color='steelblue')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance (%)')
    ax2.set_title('Scree Plot')
    ax2.grid(True, axis='y')
    save_fig(filename)

def plot_fisherfaces(pca, lda, filename):
    if USE_HOG:
        print("  [跳过] Fisherfaces 无法在 HOG 空间可视化。")
        return
    n_lda = lda.n_components
    W_lda = lda.scalings_[:, :n_lda]
    W_pca = pca.components_
    W_fisher = W_lda.T @ W_pca
    cols = min(5, n_lda)
    rows = int(np.ceil(n_lda / cols))
    fig = plt.figure(figsize=(cols*2.5, rows*2.5))
    fig.suptitle("Fisherfaces", fontsize=14)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    for i in range(n_lda):
        ff = W_fisher[i].reshape(IMAGE_SIZE)
        ff_norm = cv2.normalize(ff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(ff_norm, cmap='gray')
        ax.axis('off')
    save_fig(filename)

def plot_lda_projection(X_lda, Y, names, filename_prefix):
    cmap = plt.cm.tab20
    n_classes = len(names)
    # 2D
    if X_lda.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl in range(n_classes):
            mask = Y == lbl
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                       c=[cmap(lbl % 20)], s=30, alpha=0.8,
                       label=names[lbl], edgecolors='k', linewidth=0.2)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_title('LDA 2D Projection (training data)')
        ax.legend(fontsize=6, loc='best', ncol=2)
        ax.grid(True)
        save_fig(f"{filename_prefix}_2d.png")
    # 3D
    if X_lda.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for lbl in range(n_classes):
            mask = Y == lbl
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], X_lda[mask, 2],
                       c=[cmap(lbl % 20)], s=20, alpha=0.8,
                       label=names[lbl])
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        ax.set_title('LDA 3D Projection')
        ax.legend(fontsize=6, loc='best', ncol=2)
        save_fig(f"{filename_prefix}_3d.png")

def plot_confusion_matrix(Y_test, Y_pred, names, filename):
    cm = confusion_matrix(Y_test, Y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Confusion Matrix", fontsize=14)
    short = [f"P{i+1}" for i in range(len(names))]
    for ax, data, title in zip(axes, [cm, cm_norm], ['Counts', 'Normalized']):
        im = ax.imshow(data, cmap='Blues', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                val_str = f"{data[i,j]:.2f}" if title == 'Normalized' else str(data[i,j])
                color = 'white' if data[i,j] > data.max()/2 else 'black'
                ax.text(j, i, val_str, ha='center', va='center', fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save_fig(filename)

def plot_per_person_accuracy(Y_test, Y_pred, names, filename):
    accs = []
    for lbl in range(len(names)):
        mask = Y_test == lbl
        if np.sum(mask) > 0:
            accs.append(accuracy_score(Y_test[mask], Y_pred[mask]))
        else:
            accs.append(0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2E86AB' if a>=0.8 else '#F6AE2D' if a>=0.5 else '#E94F37' for a in accs]
    ax.bar(range(len(names)), accs, color=colors, edgecolor='k', linewidth=0.3)
    ax.axhline(np.mean(accs), color='k', linestyle='--', label=f'Mean: {np.mean(accs):.2%}')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Person Identification Accuracy')
    ax.legend()
    ax.grid(True, axis='y')
    save_fig(filename)

def plot_reconstruction_comparison(X_test, model, Y_test, names, filename, n_samples=6):
    if USE_HOG:
        print("  [跳过] HOG 特征不支持像素重建。")
        return
    indices = []
    for lbl in range(min(len(names), n_samples)):
        idx = np.where(Y_test == lbl)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    indices = indices[:n_samples]
    fig = plt.figure(figsize=(12, 3*len(indices)))
    gs = gridspec.GridSpec(len(indices), 4, figure=fig, hspace=0.25, wspace=0.15)
    for row, idx in enumerate(indices):
        orig = X_test[idx]
        pca_full = model.pca_reconstruct(orig.reshape(1, -1), k=model.actual_pca)[0]
        pca_20 = model.pca_reconstruct(orig.reshape(1, -1), k=20)[0]
        pca_lda = model.pca_lda_reconstruct(orig.reshape(1, -1))[0]
        imgs = [orig, pca_20, pca_full, pca_lda]
        titles = ['Original', f'PCA k=20', f'PCA k={model.actual_pca}', 'PCA+LDA']
        for col, (img, title) in enumerate(zip(imgs, titles)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img.reshape(IMAGE_SIZE), cmap='gray')
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(names[Y_test[idx]], fontsize=9)
            ax.axis('off')
            if col > 0:
                mse, psnr = compute_mse_psnr(orig, img)
                ax.text(0.5, -0.08, f"MSE:{mse:.0f} PSNR:{psnr:.1f}",
                        transform=ax.transAxes, ha='center', fontsize=7)
    save_fig(filename)

def plot_parameter_analysis(X_train, Y_train, X_test, Y_test, names, filename_prefix):
    # PCA 维度影响
    max_dim = min(X_train.shape[0] - 1, X_train.shape[1])
    dims = [10, 20, 40, 60, 80, 100, 120]
    dims = [d for d in dims if d <= max_dim]
    acc_list = []
    for d in dims:
        pipe = PCALDASVMPipeline(n_pca=d, standardize=True)
        pipe.fit(X_train, Y_train)
        pred = pipe.predict(X_test)
        acc_list.append(accuracy_score(Y_test, pred))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, acc_list, 'o-', color='#2E86AB', linewidth=2)
    ax.set_xlabel('PCA dimension')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs PCA Dimension')
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    best_idx = np.argmax(acc_list)
    ax.scatter(dims[best_idx], acc_list[best_idx], c='red', s=80, zorder=5)
    ax.text(dims[best_idx], acc_list[best_idx] + 0.02, f"{acc_list[best_idx]:.2%}", ha='center')
    save_fig(f"{filename_prefix}_pca_dim.png")

    # 训练比例影响
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    ratio_accs = []
    for r in ratios:
        Xtr, Xte, Ytr, Yte = split_data(X_train, Y_train, train_ratio=r, augment=False)
        pipe = PCALDASVMPipeline(n_pca=60, standardize=True)
        pipe.fit(Xtr, Ytr)
        pred = pipe.predict(Xte)
        ratio_accs.append(accuracy_score(Yte, pred))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, ratio_accs, 'D-', color='#F6AE2D', linewidth=2, markersize=8)
    ax.set_xlabel('Training ratio per person')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Training Set Size')
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    for x, y in zip(ratios, ratio_accs):
        ax.text(x, y + 0.02, f"{y:.2%}", ha='center')
    save_fig(f"{filename_prefix}_train_ratio.png")

def plot_dimension_pipeline(filename):
    if USE_HOG:
        dim0 = "HOG\n(1568D)"
    else:
        dim0 = "Original\n(4096D)"
    stages = [dim0, "Standardized\n+ PCA", "LDA\nProjection", "Linear SVM\nClassifier"]
    colors = ['#E94F37', '#F6AE2D', '#2E86AB', '#3B1F2B']
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    xpos = np.linspace(1, 12, len(stages))
    for i, (name, color) in enumerate(zip(stages, colors)):
        rect = plt.Rectangle((xpos[i] - 0.8, 0.3), 1.6, 0.4,
                             facecolor=color, alpha=0.85, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(xpos[i], 0.5, name, ha='center', va='center', fontsize=8, color='white', weight='bold', zorder=4)
        if i < len(stages) - 1:
            ax.annotate('', xy=(xpos[i+1] - 0.8, 0.5), xytext=(xpos[i] + 0.8, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1)
    save_fig(filename)

def plot_tsne_comparison(X_train, Y_train, model, names, filename):
    n_classes = len(names)
    cmap = plt.cm.tab20
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    spaces = [
        (X_train, "Original Feature Space"),
        (model.X_train_pca, f"PCA Space ({model.actual_pca}D)"),
        (model.X_train_lda, f"LDA Space ({model.n_lda}D)")
    ]
    for col, (data, title) in enumerate(spaces):
        ax = fig.add_subplot(gs[0, col])
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, len(data)-1))
        X_tsne = tsne.fit_transform(data)
        for lbl in range(n_classes):
            mask = Y_train == lbl
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[cmap(lbl % 20)], s=30, alpha=0.8,
                       label=names[lbl] if col == 2 else "",
                       edgecolors='k', linewidth=0.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if col == 2:
            ax.legend(fontsize=6, loc='best', ncol=2)
    save_fig(filename)

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("PCA + Fisher LDA Face Recognition (Enhanced with HOG + SVM)")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/7] Loading dataset...")
    X, Y, person_names = load_dataset()
    print(f"   Classes: {len(person_names)}")
    print(f"   Total samples: {len(X)}")

    # 2. 划分训练/测试集
    print("\n[2/7] Splitting train/test with augmentation...")
    X_train, X_test, Y_train, Y_test = split_data(X, Y, train_ratio=TRAIN_RATIO, augment=USE_AUGMENTATION)
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 3. 训练模型
    print("\n[3/7] Training PCA+LDA+SVM pipeline...")
    start = time.time()
    model = PCALDASVMPipeline(n_pca=PCA_COMPONENTS, standardize=True)
    model.fit(X_train, Y_train)
    print(f"   PCA actual dim: {model.actual_pca}")
    print(f"   LDA dim: {model.n_lda}")
    print(f"   Training time: {time.time() - start:.2f}s")

    # 4. 测试与评估
    print("\n[4/7] Evaluating on test set...")
    Y_pred = model.predict(X_test)
    overall_acc = accuracy_score(Y_test, Y_pred)
    print(f"   Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    per_class_acc = {}
    for label, name in enumerate(person_names):
        mask = Y_test == label
        if np.sum(mask) > 0:
            acc = accuracy_score(Y_test[mask], Y_pred[mask])
            per_class_acc[name] = float(acc)
            print(f"   {name}: {acc:.2f} (n={np.sum(mask)})")

    # 5. 重建评估（仅适用于原始像素）
    avg_mse_pca = avg_psnr_pca = avg_mse_lda = avg_psnr_lda = 0.0
    recon_metrics = []
    if not USE_HOG:
        print("\n[5/7] Reconstruction evaluation...")
        for i in range(min(20, len(X_test))):
            orig = X_test[i]
            pca_recon = model.pca_reconstruct(orig.reshape(1, -1), k=model.actual_pca)[0]
            pca_lda_recon = model.pca_lda_reconstruct(orig.reshape(1, -1))[0]
            mse_pca, psnr_pca = compute_mse_psnr(orig, pca_recon)
            mse_lda, psnr_lda = compute_mse_psnr(orig, pca_lda_recon)
            recon_metrics.append({
                "index": i,
                "pca_full": {"mse": mse_pca, "psnr": psnr_pca},
                "pca_lda": {"mse": mse_lda, "psnr": psnr_lda}
            })
        avg_mse_pca = np.mean([r['pca_full']['mse'] for r in recon_metrics])
        avg_psnr_pca = np.mean([r['pca_full']['psnr'] for r in recon_metrics])
        avg_mse_lda = np.mean([r['pca_lda']['mse'] for r in recon_metrics])
        avg_psnr_lda = np.mean([r['pca_lda']['psnr'] for r in recon_metrics])
        print(f"   PCA full recon: MSE={avg_mse_pca:.2f}, PSNR={avg_psnr_pca:.2f}dB")
        print(f"   PCA+LDA recon : MSE={avg_mse_lda:.2f}, PSNR={avg_psnr_lda:.2f}dB")

    # 6. 可视化
    print("\n[6/7] Generating figures...")
    plot_sample_faces(X_train, Y_train, person_names, "Training Samples", "01_training_samples.png")
    plot_mean_and_eigenfaces(model.mean_face, model.pca, "02_mean_eigenfaces.png")
    plot_pca_variance(model.pca, "03_pca_variance.png")
    plot_fisherfaces(model.pca, model.lda, "04_fisherfaces.png")
    plot_lda_projection(model.X_train_lda, Y_train, person_names, "05_lda_projection")
    plot_confusion_matrix(Y_test, Y_pred, person_names, "06_confusion_matrix.png")
    plot_per_person_accuracy(Y_test, Y_pred, person_names, "07_per_person_acc.png")
    plot_reconstruction_comparison(X_test, model, Y_test, person_names, "08_reconstruction.png", n_samples=6)
    plot_parameter_analysis(X_train, Y_train, X_test, Y_test, person_names, "09_parameter")
    plot_dimension_pipeline("10_dimension_flow.png")
    plot_tsne_comparison(X_train, Y_train, model, person_names, "11_tsne_comparison.png")

    # 7. 保存指标
    print("\n[7/7] Saving metrics...")
    metrics = {
        "dataset": {
            "n_classes": len(person_names),
            "n_total": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "person_names": person_names
        },
        "model": {
            "pca_dim": int(model.actual_pca),
            "lda_dim": int(model.n_lda),
            "classifier": "LinearSVM",
            "features": "HOG" if USE_HOG else "pixels"
        },
        "classification": {
            "overall_accuracy": float(overall_acc),
            "per_class_accuracy": per_class_acc
        },
        "reconstruction": {
            "pca_full": {"avg_mse": float(avg_mse_pca), "avg_psnr": float(avg_psnr_pca)},
            "pca_lda": {"avg_mse": float(avg_mse_lda), "avg_psnr": float(avg_psnr_lda)},
            "sample_details": recon_metrics
        }
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Experiment completed! Accuracy={overall_acc*100:.2f}%")
    print(f"Results saved to {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()