# ============================================================
# BirdCLEF 2026 — Training Notebook
# 
# 目标: 训练一个高效的鸟类识别模型，在CPU推理环境下稳定运行
# 数据: ~35,549个音频文件，234个物种，多标签分类
# 评估: Macro-averaged ROC-AUC
#
# 这个notebook包含完整的训练流程:
# 1. 数据探索与预处理
# 2. 数据集和数据加载器
# 3. 模型定义 (EfficientNet-B0 backbone)
# 4. 训练循环
# 5. 验证与评估
# 6. ONNX导出(为CPU推理准备)
# 7. 快速推理验证
# ============================================================

# %% [markdown]
# # BirdCLEF 2026 — Model Training
#
# **架构**: EfficientNet-B0 + GEM Pooling → 234类多标签分类
# **特征**: 3通道Mel频谱图 (不同n_fft分辨率)
# **损失**: BCEWithLogitsLoss + 类别权重
# **优化**: AdamW + CosineAnnealingLR

# %% [code]
# ============================================================
# Cell 1: 安装依赖
# ============================================================
# 如果在Kaggle上运行，大部分依赖已预装
# !pip install -q librosa soundfile timm torch torchvision
# !pip install -q onnx onnxruntime scikit-learn tqdm pandas matplotlib

# %% [code]
# ============================================================
# Cell 2: 导入库
# ============================================================
import os
import gc
import json
import random
import warnings
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 确定运行环境
# ============================================================
ON_KAGGLE = os.path.exists("/kaggle")
print(f"Running on Kaggle: {ON_KAGGLE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [code]
# ============================================================
# Cell 3: 配置
# ============================================================
class Config:
    # ============= 路径配置 =============
    if ON_KAGGLE:
        # Kaggle环境路径 (根据实际数据集名称调整)
        BASE_DIR = Path("/kaggle/input/birdclef-2026")
        AUDIO_DIR = BASE_DIR / "train_audio"
        TEST_SOUNDSCAPE_DIR = BASE_DIR / "test_soundscapes"
        TRAIN_CSV = BASE_DIR / "train.csv"
        TAXONOMY_CSV = BASE_DIR / "taxonomy.csv"
        SOUNDSCAPE_LABELS_CSV = BASE_DIR / "train_soundscape_labels.csv"  # 如果存在
        OUTPUT_DIR = Path("/kaggle/working/birdclef2026_output")
    else:
        # 本地环境路径 - 根据你的实际路径修改
        BASE_DIR = Path("./birdclef-2026")
        AUDIO_DIR = BASE_DIR / "train_audio"
        TEST_SOUNDSCAPE_DIR = BASE_DIR / "test_soundscapes"
        TRAIN_CSV = BASE_DIR / "train.csv"
        TAXONOMY_CSV = BASE_DIR / "taxonomy.csv"
        SOUNDSCAPE_LABELS_CSV = BASE_DIR / "train_soundscape_labels.csv"
        OUTPUT_DIR = Path("./output")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============= 音频参数 =============
    SAMPLE_RATE = 32000
    DURATION = 5.0           # 秒
    N_FFT = 1024
    HOP_LENGTH = 320         # 10ms hop
    N_MELS = 128
    FMIN = 50
    FMAX = 16000
    NUM_CLASSES = 234
    
    # ============= 训练参数 =============
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 32           # 根据GPU显存调整: A100可用64, V100用32, 本地用16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    GRAD_ACCUM_STEPS = 1
    MIXUP_ALPHA = 0.3         # MixUp augmentation
    LABEL_SMOOTHING = 0.01
    
    # ============= 模型参数 =============
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    GEM_P = 3.0              # GEM pooling power
    
    # ============= 早停 =============
    PATIENCE = 8
    
    # ============= ONNX导出 =============
    ONNX_OPSET = 18    # PyTorch 2.x 推荐18+

cfg = Config()
print(f"Output dir: {cfg.OUTPUT_DIR}")
print(f"Audio dir exists: {cfg.AUDIO_DIR.exists()}")
print(f"Train CSV exists: {cfg.TRAIN_CSV.exists()}")

# %% [code]
# ============================================================
# Cell 4: 工具函数
# ============================================================
def seed_everything(seed=42):
    """设置所有随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# %% [code]
# ============================================================
# Cell 5: 数据探索
# ============================================================
print("=" * 60)
print("数据探索")
print("=" * 60)

# 读取训练数据
train_df = pd.read_csv(cfg.TRAIN_CSV)
print(f"\n训练数据形状: {train_df.shape}")
print(f"列名: {list(train_df.columns)}")
print(f"\n前5行:")
print(train_df.head())

# 读取分类学信息
taxonomy_df = pd.read_csv(cfg.TAXONOMY_CSV)
print(f"\n分类学数据形状: {taxonomy_df.shape}")
print(f"列名: {list(taxonomy_df.columns)}")

# 物种统计
if "primary_label" in train_df.columns:
    label_col = "primary_label"
elif "species" in train_df.columns:
    label_col = "species"
else:
    # 尝试找到标签列
    label_col = [c for c in train_df.columns if "label" in c.lower() or "species" in c.lower()][0]

print(f"\n标签列: {label_col}")
species_counts = train_df[label_col].value_counts()
print(f"物种数量: {len(species_counts)}")
print(f"每个物种的样本数统计:")
print(f"  最少: {species_counts.min()}")
print(f"  最多: {species_counts.max()}")
print(f"  中位数: {species_counts.median():.0f}")
print(f"  平均: {species_counts.mean():.1f}")

# 少样本物种
few_shot = species_counts[species_counts < 10]
print(f"\n少于10个样本的物种: {len(few_shot)} ({len(few_shot)/len(species_counts)*100:.1f}%)")
very_few = species_counts[species_counts <= 5]
print(f"少于等于5个样本的物种: {len(very_few)}")

# 检查音频文件
if cfg.AUDIO_DIR.exists():
    audio_files = list(cfg.AUDIO_DIR.rglob("*.ogg")) + list(cfg.AUDIO_DIR.rglob("*.wav"))
    print(f"\n音频文件总数: {len(audio_files)}")
else:
    print(f"\n警告: 音频目录 {cfg.AUDIO_DIR} 不存在!")
    print("请检查数据路径配置。")

# %% [code]
# ============================================================
# Cell 6: 数据预处理 - 构建完整数据集
# ============================================================
def build_data_splits(train_df, label_col, n_folds=5, seed=42):
    """
    构建训练/验证数据划分
    
    策略:
    - 使用分层K折交叉验证
    - 按物种分层，确保每个fold都有各类别的样本
    - 少样本物种(<5)全部放入训练集
    """
    df = train_df.copy()
    
    # 确定音频文件路径
    if "filename" in df.columns:
        file_col = "filename"
    elif "file_path" in df.columns:
        file_col = "file_path"
    elif "path" in df.columns:
        file_col = "path"
    else:
        # 假设文件名在某一列中
        file_col = [c for c in df.columns if "file" in c.lower() or "path" in c.lower() or "name" in c.lower()][0]
    
    print(f"文件名列: {file_col}")
    
    # 构建完整路径
    def get_full_path(row_path):
        path = str(row_path)
        full = cfg.AUDIO_DIR / path
        if not full.exists():
            # 尝试去掉前导目录
            full = cfg.AUDIO_DIR / Path(path).name
        if not full.exists():
            # 尝试其他可能的组合
            full = cfg.AUDIO_DIR / path.replace("train_audio/", "")
        return str(full) if full.exists() else str(cfg.AUDIO_DIR / path)
    
    df["file_path"] = df[file_col].apply(get_full_path)
    
    # 构建标签编码
    all_species = sorted(df[label_col].unique().tolist())
    species_to_idx = {sp: i for i, sp in enumerate(all_species)}
    num_classes = len(all_species)
    df["label_idx"] = df[label_col].map(species_to_idx)
    
    print(f"物种数量: {num_classes}")
    print(f"总样本数: {len(df)}")
    
    # 分层K折
    # 对于少样本物种，将它们的fold设为0(全部训练)
    df["fold"] = -1
    
    species_count = df[label_col].value_counts()
    few_shot_species = set(species_count[species_count < 5].index)
    
    # 少样本物种全部放入fold 0 (训练)
    mask_few = df[label_col].isin(few_shot_species)
    df.loc[mask_few, "fold"] = 0
    
    # 其余物种使用分层K折
    df_remaining = df[~mask_few].copy()
    if len(df_remaining) > 0:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(df_remaining, df_remaining[label_col])):
            df.loc[df_remaining.iloc[val_idx].index, "fold"] = fold_idx
    
    # 统计
    for f in range(n_folds):
        fold_data = df[df["fold"] == f]
        print(f"  Fold {f}: {len(fold_data)} 样本, {fold_data[label_col].nunique()} 物种")
    
    return df, species_to_idx, all_species, num_classes

df, species_to_idx, all_species, NUM_CLASSES_ACTUAL = build_data_splits(train_df, label_col, cfg.N_FOLDS)

# 更新实际类别数
if NUM_CLASSES_ACTUAL != cfg.NUM_CLASSES:
    print(f"注意: 实际物种数 {NUM_CLASSES_ACTUAL} != 配置的 {cfg.NUM_CLASSES}")
    cfg.NUM_CLASSES = NUM_CLASSES_ACTUAL

# %% [code]
# ============================================================
# Cell 7: 音频特征提取
# ============================================================
def load_audio_safe(path, sr=32000, duration=5.0):
    """
    安全地加载音频文件
    
    处理各种边界情况:
    - 文件不存在
    - 文件损坏
    - 文件太短
    - 采样率不匹配
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(path):
            print(f"警告: 文件不存在 {path}")
            return np.zeros(int(sr * duration), dtype=np.float32)
        
        # 检查文件大小
        file_size = os.path.getsize(path)
        if file_size < 100:  # 太小肯定有问题
            print(f"警告: 文件太小 ({file_size} bytes): {path}")
            return np.zeros(int(sr * duration), dtype=np.float32)
        
        # 尝试加载
        audio, orig_sr = librosa.load(path, sr=sr, duration=duration, mono=True)
        
        # 确保长度正确
        target_len = int(sr * duration)
        if len(audio) < target_len:
            # 补零
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        elif len(audio) > target_len:
            # 截断
            audio = audio[:target_len]
        
        # 检查是否全零或NaN
        if np.all(audio == 0) or np.any(np.isnan(audio)):
            return np.zeros(target_len, dtype=np.float32)
        
        return audio.astype(np.float32)
        
    except Exception as e:
        print(f"加载音频失败 {path}: {e}")
        return np.zeros(int(sr * duration), dtype=np.float32)


def audio_to_mel_spectrogram(audio, sr=32000, n_fft=1024, hop_length=320, 
                              n_mels=128, fmin=50, fmax=16000):
    """
    音频转Mel频谱图，使用PCEN归一化
    
    PCEN (Per-Channel Energy Normalization) 比log-Mel更适合生物声学:
    - 自适应增益控制
    - 对噪声更鲁棒
    - 保留更多动态范围
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    
    # PCEN归一化 (关键!)
    mel = librosa.pcen(
        mel * (2**31),  # 缩放到合适范围
        sr=sr, 
        hop_length=hop_length,
        gain=0.8,       # 增益控制
        bias=10,        # 偏置
        power=0.25,     # 幂次
        time_constant=0.06,  # 时间常数
        eps=1e-6
    )
    
    return mel


def create_three_channel_mel(audio, sr=32000):
    """
    创建3通道Mel频谱图
    
    不同的n_fft提供不同频率分辨率:
    - Channel 0: n_fft=1024 (标准)
    - Channel 1: n_fft=2048 (高频率分辨率)
    - Channel 2: n_fft=512 (高时间分辨率)
    
    这种多尺度特征对鸟类识别非常有效
    """
    mel_0 = audio_to_mel_spectrogram(audio, sr, n_fft=cfg.N_FFT)
    mel_1 = audio_to_mel_spectrogram(audio, sr, n_fft=cfg.N_FFT * 2)
    mel_2 = audio_to_mel_spectrogram(audio, sr, n_fft=cfg.N_FFT // 2)
    
    # 对齐时间维度
    min_time = min(mel_0.shape[1], mel_1.shape[1], mel_2.shape[1])
    mel_0 = mel_0[:, :min_time]
    mel_1 = mel_1[:, :min_time]
    mel_2 = mel_2[:, :min_time]
    
    return np.stack([mel_0, mel_1, mel_2], axis=0)  # (3, n_mels, time)


def normalize_mel(mel):
    """标准化Mel频谱图"""
    # 逐通道标准化
    for c in range(mel.shape[0]):
        channel = mel[c]
        mean = channel.mean()
        std = channel.std() + 1e-6
        mel[c] = (channel - mean) / std
    return mel


# 测试特征提取
print("测试特征提取...")
if cfg.AUDIO_DIR.exists():
    test_files = list(cfg.AUDIO_DIR.rglob("*.ogg"))[:3]
    if test_files:
        test_audio = load_audio_safe(str(test_files[0]))
        test_mel = create_three_channel_mel(test_audio)
        print(f"音频长度: {len(test_audio)} samples ({len(test_audio)/cfg.SAMPLE_RATE:.1f}s)")
        print(f"Mel频谱图形状: {test_mel.shape} (channels, n_mels, time)")
        print(f"Mel值范围: [{test_mel.min():.3f}, {test_mel.max():.3f}]")
    else:
        print("未找到测试音频文件")
else:
    print(f"音频目录不存在: {cfg.AUDIO_DIR}")

# %% [code]
# ============================================================
# Cell 8: 数据增强
# ============================================================
class AudioAugmentation:
    """
    音频/频谱图数据增强
    
    训练时使用:
    1. 随机时间偏移 (模拟不同起始点)
    2. 频谱图时间遮蔽
    3. 频谱图频率遮蔽
    4. 随机增益
    5. 噪声注入
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def time_shift(self, audio, max_shift=16000):
        """随机时间偏移"""
        if random.random() < self.p:
            shift = random.randint(-max_shift, max_shift)
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0
            else:
                audio[shift:] = 0
        return audio
    
    def add_noise(self, audio, noise_level=0.005):
        """添加高斯噪声"""
        if random.random() < self.p:
            noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
            audio = audio + noise
        return audio
    
    def random_gain(self, audio, min_gain=0.8, max_gain=1.2):
        """随机增益"""
        if random.random() < self.p:
            gain = random.uniform(min_gain, max_gain)
            audio = audio * gain
        return audio
    
    def spectrogram_augment(self, mel, freq_mask_param=15, time_mask_param=35):
        """
        频谱图增强 (SpecAugment)
        
        - 频率遮蔽: 随机遮蔽连续的频率通道
        - 时间遮蔽: 随机遮蔽连续的时间帧
        """
        mel = mel.copy()
        _, n_mels, n_time = mel.shape
        
        for c in range(3):  # 对每个通道
            # 频率遮蔽
            if random.random() < self.p:
                f = random.randint(0, freq_mask_param)
                f0 = random.randint(0, n_mels - f)
                mel[c, f0:f0+f, :] = 0
            
            # 时间遮蔽
            if random.random() < self.p:
                t = random.randint(0, min(time_mask_param, n_time - 1))
                t0 = random.randint(0, n_time - t)
                mel[c, :, t0:t0+t] = 0
        
        return mel
    
    def mixup(self, mel1, label1, mel2, label2, alpha=0.3):
        """
        MixUp数据增强
        
        将两个样本混合:
        - 混合频谱图: mel = λ * mel1 + (1-λ) * mel2
        - 混合标签: label = λ * label1 + (1-λ) * label2
        """
        lam = np.random.beta(alpha, alpha)
        mixed_mel = lam * mel1 + (1 - lam) * mel2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_mel, mixed_label

augmentation = AudioAugmentation(p=0.5)

# %% [code]
# ============================================================
# Cell 9: Dataset 类
# ============================================================
class BirdCLEFDataset(Dataset):
    """
    BirdCLEF 2026 数据集
    
    支持:
    - 训练模式: 数据增强、随机裁剪
    - 验证模式: 无增强、中心裁剪
    - 标签平滑
    """
    
    def __init__(self, df, species_to_idx, num_classes, mode="train", 
                 duration=5.0, sample_rate=32000, label_smoothing=0.0):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.mode = mode
        self.duration = duration
        self.sample_rate = sample_rate
        self.label_smoothing = label_smoothing
        self.augmentation = AudioAugmentation(p=0.5) if mode == "train" else None
        
        # 确定标签列
        if "primary_label" in self.df.columns:
            self.label_col = "primary_label"
        elif "species" in self.df.columns:
            self.label_col = "species"
        else:
            self.label_col = [c for c in self.df.columns if "label" in c.lower()][0]
        
        # 过滤掉文件不存在的样本
        self.valid_indices = []
        for i in range(len(self.df)):
            path = self.df.iloc[i]["file_path"]
            if os.path.exists(path):
                self.valid_indices.append(i)
        
        print(f"[{mode}] 总样本: {len(self.df)}, 有效样本: {len(self.valid_indices)}")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"[{mode}] 没有找到有效的音频文件! 请检查路径配置。")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        # 加载音频
        audio = load_audio_safe(
            row["file_path"], 
            sr=self.sample_rate, 
            duration=self.duration
        )
        
        # 音频增强 (训练时)
        if self.augmentation:
            audio = self.augmentation.time_shift(audio)
            audio = self.augmentation.add_noise(audio)
            audio = self.augmentation.random_gain(audio)
        
        # 转换为Mel频谱图
        mel = create_three_channel_mel(audio, self.sample_rate)
        
        # 固定时间步数
        target_steps = int(self.sample_rate * self.duration / cfg.HOP_LENGTH) + 1
        mel = self._crop_or_pad(mel, target_steps)
        
        # 频谱图增强 (训练时)
        if self.augmentation:
            mel = self.augmentation.spectrogram_augment(mel)
        
        # 标准化
        mel = normalize_mel(mel)
        
        # 生成标签 (多标签)
        label = self._encode_label(row[self.label_col])
        
        # 标签平滑
        if self.label_smoothing > 0:
            label = label * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        return torch.FloatTensor(mel), torch.FloatTensor(label)
    
    def _crop_or_pad(self, mel, target_steps):
        """裁剪或填充到固定时间步"""
        _, current = mel.shape[1], mel.shape[2] if len(mel.shape) == 3 else mel.shape[1]
        if len(mel.shape) == 3:
            C, M, T = mel.shape
            if T > target_steps:
                start = (T - target_steps) // 2
                mel = mel[:, :, start:start + target_steps]
            elif T < target_steps:
                pad_left = (target_steps - T) // 2
                pad_right = target_steps - T - pad_left
                mel = np.pad(mel, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")
        else:
            M, T = mel.shape
            if T > target_steps:
                start = (T - target_steps) // 2
                mel = mel[:, start:start + target_steps]
            elif T < target_steps:
                pad_left = (target_steps - T) // 2
                pad_right = target_steps - T - pad_left
                mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant")
        return mel
    
    def _encode_label(self, species_name):
        """编码标签为one-hot"""
        label = np.zeros(self.num_classes, dtype=np.float32)
        if species_name in self.species_to_idx:
            label[self.species_to_idx[species_name]] = 1.0
        return label


# %% [code]
# ============================================================
# Cell 10: 计算类别权重
# ============================================================
def compute_class_weights(df, species_to_idx, num_classes, label_col="primary_label"):
    """
    计算类别权重 (逆频率)
    
    用于处理极端类别不平衡:
    - 常见物种权重低
    - 稀有物种权重高
    """
    counts = np.zeros(num_classes)
    for sp in df[label_col]:
        if sp in species_to_idx:
            counts[species_to_idx[sp]] += 1
    
    # 避免除零
    counts = np.maximum(counts, 1)
    
    # 逆频率权重
    total = counts.sum()
    weights = total / (num_classes * counts)
    
    # 限制权重范围，避免极端值
    weights = np.clip(weights, 0.1, 10.0)
    
    # 归一化
    weights = weights / weights.mean()
    
    print(f"类别权重统计:")
    print(f"  最小: {weights.min():.3f}")
    print(f"  最大: {weights.max():.3f}")
    print(f"  中位数: {np.median(weights):.3f}")
    
    return torch.FloatTensor(weights)


label_col_for_weights = "primary_label" if "primary_label" in df.columns else label_col
class_weights = compute_class_weights(df, species_to_idx, cfg.NUM_CLASSES, label_col_for_weights)

# %% [code]
# ============================================================
# Cell 11: 模型定义
# ============================================================
class GEMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    
    比普通的Global Average Pooling更灵活:
    - p=1: 等价于Average Pooling
    - p→∞: 等价于Max Pooling
    - p=3: 一个好的折中值
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=-1).pow(1.0 / self.p)


class BirdCLEFModel(nn.Module):
    """
    BirdCLEF 2026 主模型
    
    架构:
    EfficientNet-B0 backbone → GEM Pooling → FC → 234类输出
    
    特点:
    - 使用预训练的ImageNet权重
    - 3通道输入 (对应3种分辨率的Mel频谱图)
    - GEM Pooling自适应聚合特征
    - 支持MixUp训练
    """
    
    def __init__(self, model_name="efficientnet_b0", num_classes=234, 
                 pretrained=True, gem_p=3.0, dropout=0.3):
        super().__init__()
        
        # 加载backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # 去掉原始分类头
            global_pool=""  # 去掉原始池化
        )
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 501)  # (B, C, n_mels, time_steps)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
            self.spatial_size = features.shape[2:]
        
        print(f"Backbone: {model_name}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Spatial size: {self.spatial_size}")
        
        # GEM Pooling
        self.gem_pool = GEMPooling(p=gem_p)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # 初始化分类头权重
        self._init_classifier()
    
    def _init_classifier(self):
        """初始化分类头"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 3, n_mels, time_steps) 3通道Mel频谱图
        
        Returns:
            logits: (B, num_classes) 未经过sigmoid的输出
        """
        # Backbone特征提取
        features = self.backbone(x)  # (B, C, H, W)
        
        # GEM Pooling: (B, C, H, W) → (B, C)
        B, C, H, W = features.shape
        features = features.view(B, C, -1)  # (B, C, H*W)
        features = self.gem_pool(features)  # (B, C)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


# 创建模型
print("创建模型...")
model = BirdCLEFModel(
    model_name=cfg.MODEL_NAME,
    num_classes=cfg.NUM_CLASSES,
    pretrained=cfg.PRETRAINED,
    gem_p=cfg.GEM_P,
    dropout=0.3
)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

# %% [code]
# ============================================================
# Cell 12: 损失函数
# ============================================================
class BCEWithLogitsLossWeighted(nn.Module):
    """
    带类别权重的BCE损失函数
    
    对稀有物种给予更高的权重，缓解类别不平衡
    """
    
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / targets.shape[-1]
        
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            reduction="mean"
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 对难分类样本给予更高权重
    
    公式: FL = -α(1-p)^γ * log(p)
    
    当模型对某个样本很确信时(高p)，损失降低;
    对难样本(低p)，损失保持较高。
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device).unsqueeze(0)
        
        return loss.mean()


# 使用BCE + 类别权重
criterion = BCEWithLogitsLossWeighted(
    class_weights=class_weights,
    label_smoothing=cfg.LABEL_SMOOTHING
)
print("损失函数: BCEWithLogitsLoss + 类别权重 + 标签平滑")

# %% [code]
# ============================================================
# Cell 13: 优化器和调度器
# ============================================================
def create_optimizer_and_scheduler(model, epochs, lr, weight_decay, steps_per_epoch):
    """
    创建优化器和学习率调度器
    
    优化器: AdamW (权重衰减的Adam)
    调度器: CosineAnnealingWarmRestarts (余弦退火)
    """
    # AdamW优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch * 5,  # 每5个epoch重启一次
        T_mult=2,                  # 每次重启后周期翻倍
        eta_min=1e-6
    )
    
    return optimizer, scheduler

print("优化器: AdamW")
print("调度器: CosineAnnealingWarmRestarts")

# %% [code]
# ============================================================
# Cell 14: 评估指标
# ============================================================
def calculate_auc_roc(predictions, targets, num_classes):
    """
    计算 Macro-averaged ROC-AUC (比赛评估指标)
    
    Args:
        predictions: (N, num_classes) 概率值
        targets: (N, num_classes) 真实标签
    
    Returns:
        macro_auc: 宏平均AUC
        per_class_auc: 每类AUC
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    per_class_auc = []
    for c in range(num_classes):
        try:
            if len(np.unique(targets[:, c])) > 1:
                auc = roc_auc_score(targets[:, c], predictions[:, c])
                per_class_auc.append(auc)
            else:
                # 只有一个类别，无法计算AUC
                per_class_auc.append(0.5)  # 随机猜测
        except Exception:
            per_class_auc.append(0.5)
    
    macro_auc = np.mean(per_class_auc)
    return macro_auc, per_class_auc


# %% [code]
# ============================================================
# Cell 15: MixUp 工具
# ============================================================
def mixup_data(x, y, alpha=0.3):
    """
    MixUp数据增强
    
    将两个样本线性混合:
    x_mix = λ * x_i + (1-λ) * x_j
    y_mix = λ * y_i + (1-λ) * y_j
    
    其中 λ ~ Beta(α, α)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


# %% [code]
# ============================================================
# Cell 16: 训练一个Epoch
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, 
                     epoch, mixup_alpha=0.3, grad_accum_steps=1, scaler=None):
    """
    训练一个epoch
    
    包含:
    - MixUp数据增强
    - 梯度累积
    - 混合精度训练
    - 学习率调度
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, (mels, labels) in enumerate(pbar):
        mels = mels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # MixUp增强
        if mixup_alpha > 0 and random.random() < 0.5:
            mels, labels = mixup_data(mels, labels, alpha=mixup_alpha)
        
        # 前向传播
        if scaler is not None:
            with autocast():
                logits = model(mels)
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(mels)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps
            loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
        
        running_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "lr": f"{current_lr:.2e}"
        })
    
    epoch_loss = running_loss / max(num_batches, 1)
    return epoch_loss


# %% [code]
# ============================================================
# Cell 17: 验证一个Epoch
# ============================================================
@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    """
    验证一个epoch
    
    计算:
    - 验证损失
    - Macro-averaged ROC-AUC
    - 每类AUC
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    for mels, labels in pbar:
        mels = mels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        logits = model(mels)
        loss = criterion(logits, labels)
        
        # 收集预测
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        
        running_loss += loss.item()
        num_batches += 1
    
    # 计算指标
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    val_loss = running_loss / max(num_batches, 1)
    macro_auc, per_class_auc = calculate_auc_roc(all_preds, all_targets, num_classes)
    
    return val_loss, macro_auc, per_class_auc


# %% [code]
# ============================================================
# Cell 18: ONNX导出
# ============================================================
def export_to_onnx(model, save_path, num_classes, sample_rate=32000, 
                    duration=5.0, hop_length=320, n_mels=128):
    """
    将PyTorch模型导出为ONNX格式
    
    ONNX格式的优势:
    - 跨平台部署
    - CPU推理优化
    - 推理速度快
    """
    model.eval()
    
    # 计算输入尺寸
    time_steps = int(sample_rate * duration / hop_length) + 1
    dummy_input = torch.randn(1, 3, n_mels, time_steps).to(next(model.parameters()).device)
    
    # 导出 (兼容 PyTorch 2.x 的 dynamo 导出器)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=cfg.ONNX_OPSET,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
    except Exception as e:
        # PyTorch 2.x+ 使用 dynamo 导出器时可能不支持 dynamic_axes
        print(f"Standard export failed ({e}), trying without dynamic_axes...")
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=cfg.ONNX_OPSET,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
    
    # 验证
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX模型已导出: {save_path}")
    print(f"文件大小: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
    
    return save_path


# %% [code]
# ============================================================
# Cell 19: ONNX推理验证
# ============================================================
def verify_onnx_inference(onnx_path, model, device, sample_rate=32000,
                           duration=5.0, hop_length=320, n_mels=128):
    """
    验证ONNX模型推理结果与PyTorch一致
    """
    import onnxruntime as ort
    
    # PyTorch推理
    model.eval()
    time_steps = int(sample_rate * duration / hop_length) + 1
    dummy = torch.randn(1, 3, n_mels, time_steps).to(device)
    
    with torch.no_grad():
        pt_output = torch.sigmoid(model(dummy)).cpu().numpy()
    
    # ONNX推理
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    
    onnx_output = sess.run(None, {"input": dummy.cpu().numpy()})
    onnx_probs = 1.0 / (1.0 + np.exp(-onnx_output[0]))
    
    # 对比
    diff = np.abs(pt_output - onnx_probs).max()
    print(f"PyTorch vs ONNX 最大差异: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ ONNX导出验证通过!")
    else:
        print("⚠️ ONNX导出可能有问题，差异较大")
    
    return diff < 1e-4


# %% [code]
# ============================================================
# Cell 20: 主训练循环
# ============================================================
def train_fold(fold_idx, df, species_to_idx, num_classes, class_weights, device):
    """
    训练单个fold
    
    包含完整的训练流程:
    1. 创建数据集和数据加载器
    2. 初始化模型、优化器、损失函数
    3. 训练循环 (含早停)
    4. 保存最佳模型
    5. 导出ONNX
    """
    print("\n" + "=" * 60)
    print(f"FOLD {fold_idx}")
    print("=" * 60)
    
    # 划分训练/验证集
    train_data = df[df["fold"] != fold_idx].copy()
    val_data = df[df["fold"] == fold_idx].copy()
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 创建数据集
    label_col_use = "primary_label" if "primary_label" in df.columns else label_col
    train_dataset = BirdCLEFDataset(
        train_data, species_to_idx, num_classes, 
        mode="train", label_smoothing=cfg.LABEL_SMOOTHING
    )
    val_dataset = BirdCLEFDataset(
        val_data, species_to_idx, num_classes, 
        mode="val", label_smoothing=0.0
    )
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print(f"❌ 训练集为空! 跳过 Fold {fold_idx}")
        return None
    
    if len(val_dataset) == 0:
        print(f"❌ 验证集为空! 跳过 Fold {fold_idx}")
        return None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    # 初始化模型
    model = BirdCLEFModel(
        model_name=cfg.MODEL_NAME,
        num_classes=num_classes,
        pretrained=cfg.PRETRAINED,
        gem_p=cfg.GEM_P,
        dropout=0.3
    ).to(device)
    
    # 损失函数
    criterion = BCEWithLogitsLossWeighted(
        class_weights=class_weights,
        label_smoothing=cfg.LABEL_SMOOTHING
    )
    
    # 优化器和调度器
    steps_per_epoch = len(train_loader) // cfg.GRAD_ACCUM_STEPS
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, cfg.EPOCHS, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, steps_per_epoch
    )
    
    # 混合精度
    scaler = GradScaler() if device.type == "cuda" else None
    
    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "lr": []
    }
    
    # 早停
    best_auc = 0.0
    patience_counter = 0
    best_model_path = cfg.OUTPUT_DIR / f"best_model_fold{fold_idx}.pth"
    
    # 训练循环
    for epoch in range(cfg.EPOCHS):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, cfg.MIXUP_ALPHA, cfg.GRAD_ACCUM_STEPS, scaler
        )
        
        # 验证
        val_loss, val_auc, per_class_auc = validate_one_epoch(
            model, val_loader, criterion, device, num_classes
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)
        
        # 打印
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": {
                    "model_name": cfg.MODEL_NAME,
                    "num_classes": num_classes,
                    "sample_rate": cfg.SAMPLE_RATE,
                    "duration": cfg.DURATION,
                    "n_fft": cfg.N_FFT,
                    "hop_length": cfg.HOP_LENGTH,
                    "n_mels": cfg.N_MELS,
                    "fmin": cfg.FMIN,
                    "fmax": cfg.FMAX,
                }
            }, best_model_path)
            
            print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"  ⏹️ 早停触发 (连续{cfg.PATIENCE}个epoch无提升)")
                break
        
        # 清理显存
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # 导出ONNX
    print(f"\n加载最佳模型并导出ONNX...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    onnx_path = cfg.OUTPUT_DIR / f"model_fold{fold_idx}.onnx"
    export_to_onnx(model, str(onnx_path), num_classes)
    
    # 验证ONNX
    verify_onnx_inference(str(onnx_path), model, device)
    
    # 保存训练历史
    history_path = cfg.OUTPUT_DIR / f"history_fold{fold_idx}.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    
    print(f"\nFold {fold_idx} 完成!")
    print(f"最佳验证AUC: {best_auc:.4f}")
    print(f"模型保存: {best_model_path}")
    print(f"ONNX导出: {onnx_path}")
    
    return {
        "fold": fold_idx,
        "best_auc": best_auc,
        "model_path": str(best_model_path),
        "onnx_path": str(onnx_path),
        "history": history
    }


# %% [code]
# ============================================================
# Cell 21: 运行训练
# ============================================================
def run_training():
    """
    运行完整训练流程
    
    根据配置选择:
    - 单fold快速验证
    - 多fold交叉验证
    """
    print("=" * 60)
    print("BirdCLEF 2026 训练开始")
    print("=" * 60)
    
    results = []
    
    # 选择训练模式
    # 快速模式: 只训练1个fold (用于验证流程)
    # 完整模式: 训练所有fold
    
    # 默认: 训练前2个fold (平衡速度和效果)
    FOLDS_TO_TRAIN = [0, 1]  # 修改这里来训练更多fold
    
    for fold_idx in FOLDS_TO_TRAIN:
        result = train_fold(
            fold_idx, df, species_to_idx, 
            cfg.NUM_CLASSES, class_weights, DEVICE
        )
        if result is not None:
            results.append(result)
        
        # 每个fold后清理显存
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("训练完成! 汇总结果:")
    print("=" * 60)
    
    for r in results:
        print(f"Fold {r['fold']}: Best AUC = {r['best_auc']:.4f}")
    
    if results:
        avg_auc = np.mean([r["best_auc"] for r in results])
        print(f"\n平均 AUC: {avg_auc:.4f}")
    
    # 保存结果
    results_path = cfg.OUTPUT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果保存到: {results_path}")
    
    return results

# 开始训练!
results = run_training()

# %% [code]
# ============================================================
# Cell 22: 可视化训练过程
# ============================================================
def plot_training_history(results):
    """绘制训练历史"""
    n_folds = len(results)
    fig, axes = plt.subplots(2, n_folds, figsize=(6 * n_folds, 10))
    if n_folds == 1:
        axes = axes.reshape(2, 1)
    
    for i, r in enumerate(results):
        history = r["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss曲线
        axes[0, i].plot(epochs, history["train_loss"], "b-", label="Train Loss")
        axes[0, i].plot(epochs, history["val_loss"], "r-", label="Val Loss")
        axes[0, i].set_xlabel("Epoch")
        axes[0, i].set_ylabel("Loss")
        axes[0, i].set_title(f"Fold {r['fold']} - Loss")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[1, i].plot(epochs, history["val_auc"], "g-", label="Val AUC")
        axes[1, i].axhline(y=r["best_auc"], color="r", linestyle="--", 
                           label=f"Best: {r['best_auc']:.4f}")
        axes[1, i].set_xlabel("Epoch")
        axes[1, i].set_ylabel("ROC-AUC")
        axes[1, i].set_title(f"Fold {r['fold']} - AUC")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = cfg.OUTPUT_DIR / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"训练历史图保存到: {save_path}")

if results:
    plot_training_history(results)

# %% [code]
# ============================================================
# Cell 23: 快速推理验证
# ============================================================
def quick_inference_test(results):
    """
    快速推理验证
    
    使用ONNX Runtime在CPU上推理，验证:
    1. 模型可以正常加载
    2. 推理输出形状正确
    3. 推理时间在预算内
    """
    import onnxruntime as ort
    
    if not results:
        print("没有可用的模型")
        return
    
    onnx_path = results[0]["onnx_path"]
    
    print(f"测试ONNX推理: {onnx_path}")
    
    # 加载ONNX模型
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_options, 
                                 providers=["CPUExecutionProvider"])
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    output_shape = sess.get_outputs()[0].shape
    
    print(f"Input: {input_name}, shape: {input_shape}")
    print(f"Output: {output_name}, shape: {output_shape}")
    
    # 生成测试输入
    time_steps = int(cfg.SAMPLE_RATE * cfg.DURATION / cfg.HOP_LENGTH) + 1
    test_input = np.random.randn(1, 3, cfg.N_MELS, time_steps).astype(np.float32)
    
    # 测速
    num_runs = 10
    times = []
    for _ in range(num_runs):
        start = time.time()
        output = sess.run([output_name], {input_name: test_input})
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    probs = 1.0 / (1.0 + np.exp(-output[0]))
    
    print(f"\n推理结果:")
    print(f"  输出形状: {probs.shape}")
    print(f"  概率范围: [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"  平均推理时间: {avg_time*1000:.1f} ms")
    print(f"  预计总推理时间 (600个音频): {avg_time * 600 * 12 / 60:.1f} 分钟")
    
    # 检查是否在时间预算内
    total_estimated = avg_time * 600 * 12  # 600文件 × 12个5秒窗口
    if total_estimated < 90 * 60:
        print(f"  ✅ 预计时间在90分钟预算内!")
    else:
        print(f"  ⚠️ 预计时间超出90分钟预算，需要优化!")
    
    return probs

if results:
    quick_inference_test(results)

# %% [markdown]
# # 训练完成!
# 
# ## 输出文件
# 
# - `best_model_fold{N}.pth` - PyTorch模型权重
# - `model_fold{N}.onnx` - ONNX模型 (用于CPU推理)
# - `history_fold{N}.json` - 训练历史
# - `training_history.png` - 训练曲线图
# - `training_results.json` - 训练结果汇总
# 
# ## 下一步
# 
# 1. 使用 `model_fold{N}.onnx` 进行推理
# 2. 训练更多fold以获得更好的集成效果
# 3. 根据验证AUC调整超参数
# 4. 使用推理notebook生成submission.csv

print("\n" + "=" * 60)
print("训练notebook执行完成!")
print(f"输出目录: {cfg.OUTPUT_DIR}")
print("=" * 60)
