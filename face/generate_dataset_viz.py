import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_test1")
RAW_DIR = os.path.join(BASE_DIR, "raw_photos")
EXTRA_VIZ_DIR = os.path.join(BASE_DIR, "results", "extra_viz")
os.makedirs(EXTRA_VIZ_DIR, exist_ok=True)

IMAGE_SIZE = (64, 64)

def get_dataset_info():
    person_folders = sorted([d for d in os.listdir(PROCESSED_DIR)
                              if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    info = {}
    for person in person_folders:
        person_dir = os.path.join(PROCESSED_DIR, person)
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        info[person] = len(files)
    return info

def plot_dataset_sample_grid():
    """生成20类×5张的样本展示图"""
    person_folders = sorted([d for d in os.listdir(PROCESSED_DIR)
                              if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    n_persons = len(person_folders)
    n_per = 5
    fig = plt.figure(figsize=(n_per * 2.2, n_persons * 2.4))
    fig.suptitle('Dataset Sample Gallery (processed_test1)\n20 Classes × 5 Samples',
                 fontsize=18, fontweight='bold', y=0.995)
    gs = gridspec.GridSpec(n_persons, n_per, figure=fig, hspace=0.35, wspace=0.15)
    for row, person in enumerate(person_folders):
        person_dir = os.path.join(PROCESSED_DIR, person)
        files = sorted([f for f in os.listdir(person_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        for col in range(n_per):
            ax = fig.add_subplot(gs[row, col])
            if col < len(files):
                img = cv2.imread(os.path.join(person_dir, files[col]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap='gray')
            ax.set_title(f"{person}" if col == 0 else "", fontsize=9, fontweight='bold')
            ax.axis('off')
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'dataset_sample_grid.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] dataset_sample_grid.png")

def plot_class_distribution():
    """每类样本数量分布"""
    info = get_dataset_info()
    names = list(info.keys())
    counts = list(info.values())
    colors = plt.cm.Spectral(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xlabel('Person Class', fontsize=12)
    ax.set_title('Sample Distribution per Class (processed_test1)', fontsize=14, fontweight='bold')
    ax.axhline(np.mean(counts), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(counts):.1f}')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'class_distribution.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] class_distribution.png")

def plot_preprocessing_comparison():
    """原始图像 vs 预处理图像对比"""
    person_folders = sorted([d for d in os.listdir(RAW_DIR)
                              if os.path.isdir(os.path.join(RAW_DIR, d))])
    if not person_folders:
        print("  [SKIP] raw_photos not available for comparison")
        return
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Preprocessing Comparison: Raw vs Processed', fontsize=16, fontweight='bold')
    n_show = min(5, len(person_folders))
    gs = gridspec.GridSpec(n_show, 4, figure=fig, hspace=0.35, wspace=0.2)
    for i, person in enumerate(person_folders[:n_show]):
        raw_dir = os.path.join(RAW_DIR, person)
        proc_dir = os.path.join(PROCESSED_DIR, person)
        raw_files = sorted([f for f in os.listdir(raw_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.heic'))])
        proc_files = sorted([f for f in os.listdir(proc_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if not raw_files or not proc_files:
            continue
        raw_img = cv2.imread(os.path.join(raw_dir, raw_files[0]))
        proc_img = cv2.imread(os.path.join(proc_dir, proc_files[0]), cv2.IMREAD_GRAYSCALE)
        if raw_img is not None:
            raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            ax0 = fig.add_subplot(gs[i, 0])
            ax0.imshow(raw_rgb)
            ax0.set_title(f'{person} Raw', fontsize=9)
            ax0.axis('off')
            # raw gray
            raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            ax1 = fig.add_subplot(gs[i, 1])
            ax1.imshow(raw_gray, cmap='gray')
            ax1.set_title('Raw Gray', fontsize=9)
            ax1.axis('off')
        if proc_img is not None:
            ax2 = fig.add_subplot(gs[i, 2])
            ax2.imshow(proc_img, cmap='gray')
            ax2.set_title('Processed (64×64)', fontsize=9)
            ax2.axis('off')
            # CLAHE效果展示（模拟）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(proc_img)
            ax3 = fig.add_subplot(gs[i, 3])
            ax3.imshow(enhanced, cmap='gray')
            ax3.set_title('CLAHE Enhanced', fontsize=9)
            ax3.axis('off')
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'preprocessing_comparison.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] preprocessing_comparison.png")

def plot_pixel_intensity_distribution():
    """像素强度分布直方图"""
    all_pixels = []
    person_folders = sorted([d for d in os.listdir(PROCESSED_DIR)
                              if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    for person in person_folders:
        person_dir = os.path.join(PROCESSED_DIR, person)
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for f in files[:3]:
            img = cv2.imread(os.path.join(person_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_pixels.extend(img.flatten())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(all_pixels, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Pixel Intensity Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    # 每类的平均像素强度
    means = []
    names = []
    for person in person_folders:
        person_dir = os.path.join(PROCESSED_DIR, person)
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        vals = []
        for f in files:
            img = cv2.imread(os.path.join(person_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                vals.append(np.mean(img))
        if vals:
            means.append(np.mean(vals))
            names.append(person)
    axes[1].bar(range(len(names)), means, color='coral', edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean Pixel Intensity', fontsize=12)
    axes[1].set_xlabel('Person Class', fontsize=12)
    axes[1].set_title('Mean Pixel Intensity per Class', fontsize=13, fontweight='bold')
    axes[1].axhline(np.mean(means), color='red', linestyle='--', label=f'Global Mean: {np.mean(means):.1f}')
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'pixel_intensity_distribution.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] pixel_intensity_distribution.png")

def plot_data_augmentation_visual():
    """数据增强可视化"""
    person_folders = sorted([d for d in os.listdir(PROCESSED_DIR)
                              if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Data Augmentation Visualization (Gaussian Noise on HOG Features)',
                 fontsize=16, fontweight='bold')
    # 展示一个人物的原始图像和增强后的效果（通过像素级模拟）
    person = person_folders[0]
    person_dir = os.path.join(PROCESSED_DIR, person)
    files = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if len(files) < 1:
        plt.close()
        return
    img = cv2.imread(os.path.join(person_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        plt.close()
        return
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
    titles = ['Original', 'Noise +0.02', 'Noise -0.02', 'Brightness +15',
              'Brightness -15', 'Flip Horizontal', 'Rotate +5°', 'Rotate -5°']
    imgs = [img]
    # 模拟HOG增强在像素上的表现
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    imgs.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    imgs.append(np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8))
    imgs.append(np.clip(img.astype(np.int16) + 15, 0, 255).astype(np.uint8))
    imgs.append(np.clip(img.astype(np.int16) - 15, 0, 255).astype(np.uint8))
    imgs.append(cv2.flip(img, 1))
    rows, cols = img.shape
    for angle in [5, -5]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        imgs.append(cv2.warpAffine(img, M, (cols, rows)))
    for i, (im, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.imshow(im, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'data_augmentation.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] data_augmentation.png")

def plot_hog_visualization():
    """HOG特征可视化"""
    from skimage.feature import hog
    person_folders = sorted([d for d in os.listdir(PROCESSED_DIR)
                              if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    person = person_folders[0]
    person_dir = os.path.join(PROCESSED_DIR, person)
    files = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if not files:
        return
    img = cv2.imread(os.path.join(person_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    fd, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title('Original Image (64×64)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title(f'HOG Visualization\n({len(fd)} dimensions)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    # HOG特征向量展示
    axes[2].plot(fd, linewidth=0.8, color='steelblue')
    axes[2].set_title('HOG Feature Vector', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Magnitude')
    axes[2].grid(True, alpha=0.3)
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'hog_visualization.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] hog_visualization.png")

def plot_train_test_split_visual():
    """训练/测试划分可视化"""
    info = get_dataset_info()
    names = list(info.keys())
    counts = list(info.values())
    train_counts = [int(c * 0.7) for c in counts]
    test_counts = [c - t for c, t in zip(counts, train_counts)]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train (70%)', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test (30%)', color='#F6AE2D', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xlabel('Person Class', fontsize=12)
    ax.set_title('Train/Test Split Distribution per Class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    # 标注总数
    for i, (t, te) in enumerate(zip(train_counts, test_counts)):
        ax.text(i, t + te + 0.3, str(t+te), ha='center', va='bottom', fontsize=8, color='gray')
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'train_test_split.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] train_test_split.png")

def plot_feature_space_comparison():
    """不同特征表示的对比概念图"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    stages = [
        ('Raw Pixels', '4,096D', '#E94F37'),
        ('HOG Features', '~1,568D', '#F6AE2D'),
        ('Standardized', '~1,568D', '#F6AE2D'),
        ('PCA Space', '100D', '#2E86AB'),
        ('LDA Space', '19D', '#3B1F2B'),
    ]
    xpos = np.linspace(1, 14, len(stages))
    for i, (name, dim, color) in enumerate(zip(stages, [s[1] for s in stages], [s[2] for s in stages])):
        rect = plt.Rectangle((xpos[i] - 0.8, 0.35), 1.6, 0.3,
                             facecolor=color, alpha=0.85, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(xpos[i], 0.5, dim, ha='center', va='center', fontsize=10, color='white', weight='bold', zorder=4)
        ax.text(xpos[i], 0.3, name, ha='center', va='top', fontsize=9, color=color, weight='bold')
        if i < len(stages) - 1:
            ax.annotate('', xy=(xpos[i+1] - 0.8, 0.5), xytext=(xpos[i] + 0.8, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_title('Feature Space Dimensionality Evolution', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(EXTRA_VIZ_DIR, 'feature_space_evolution.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [OK] feature_space_evolution.png")

if __name__ == '__main__':
    print("=" * 60)
    print("Generating extra dataset visualizations...")
    print("=" * 60)
    plot_dataset_sample_grid()
    plot_class_distribution()
    plot_preprocessing_comparison()
    plot_pixel_intensity_distribution()
    plot_data_augmentation_visual()
    plot_hog_visualization()
    plot_train_test_split_visual()
    plot_feature_space_comparison()
    print("\nAll extra visualizations saved to:", EXTRA_VIZ_DIR)
