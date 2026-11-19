
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import math
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置 
device = torch.device("cpu")


# 1. 数据准备
print("正在加载和预处理数据...")
iris = datasets.load_iris()
X = iris.data.astype(np.float32)  # (150, 4)
y = iris.target.astype(np.int64)  # (150,)

feature_names = iris.feature_names
target_names = iris.target_names

# 标准化 (极其重要，加速深度学习收敛)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled)
y_tensor = torch.tensor(y)

# 划分数据集 (80%训练，20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)


# 2. 模型定义

# 2.1 深度神经网络 (DNN)
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x), None # None是为了和自编码器接口统一

# 2.2 卷积神经网络 (CNN - 使用1D卷积处理4个连续特征)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入 shape: (batch, channels=1, sequence_length=4)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(16 * 1, 3) 
        
    def forward(self, x):
        x = x.unsqueeze(1) # (B, 4) -> (B, 1, 4)
        x = F.relu(self.conv1(x)) # -> (B, 16, 3)
        x = self.pool(x)          # -> (B, 16, 1)
        x = x.view(x.size(0), -1) # 展平 -> (B, 16)
        return self.fc(x), None

# 2.3 长短期记忆网络 (LSTM)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        # 把4个特征看作时间序列长度为4，每个时间点输入维度为1
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 3)
        
    def forward(self, x):
        x = x.unsqueeze(-1) # (B, 4) -> (B, 4, 1)
        out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = out[:, -1, :] 
        return self.fc(last_out), None

# 2.4 自编码器分类器 (AutoEncoder)
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码器：将4维降维到2维（特征提取）
        self.encoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
        # 解码器：从2维重构回4维
        self.decoder = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        # 分类器：利用降维后的2维特征进行分类
        self.classifier = nn.Linear(2, 3)
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        logits = self.classifier(encoded)
        return logits, reconstructed # 返回分类结果和重构结果

# 2.5 Transformer (严格拆解版，包含Encoder和Decoder)
# 拆解1：多头注意力机制的基础模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义Q, K, V的线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # 输出投影
        
    def forward(self, q, k, v):
        B = q.size(0)
        # 线性变换后拆分成多头: (B, Seq, d_model) -> (B, num_heads, Seq, d_k)
        Q = self.W_q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算: Softmax(Q * K^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # 拼接多头并做最后的线性变换
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)

# 拆解2：编码器层 (Self-Attention)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 残差连接 + 归一化
        attn_out = self.self_attn(x, x, x) # 自注意力：Q=K=V=x
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# 拆解3：解码器层 (Cross-Attention)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 交叉注意力：Q来自Decoder自身，K和V来自Encoder的输出
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out):
        # 1. 解码器自注意力
        attn1 = self.self_attn(x, x, x)
        x = self.norm1(x + attn1)
        # 2. 编码器-解码器交叉注意力 (Q=x, K=enc_out, V=enc_out)
        attn2 = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + attn2)
        # 3. 前馈神经网络
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)

# 拆解4：完整的Transformer组装
class TransformerArchitecture(nn.Module):
    def __init__(self):
        super(TransformerArchitecture, self).__init__()
        d_model = 16
        # 将每个特征(标量)映射到d_model维度
        self.feature_embedding = nn.Linear(1, d_model)
        # 位置编码 (告知模型特征的排列顺序)
        self.pos_emb = nn.Parameter(torch.randn(1, 4, d_model))
        
        self.encoder = EncoderLayer(d_model, num_heads=2)
        
        # 为分类任务设计的一个专用Query Token (替代机器翻译中的目标句子)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.decoder = DecoderLayer(d_model, num_heads=2)
        
        self.classifier = nn.Linear(d_model, 3)
        
    def forward(self, x):
        B = x.size(0)
        # 输入处理：(B, 4) -> (B, 4, 1) -> (B, 4, d_model)
        x = x.unsqueeze(-1)
        x_emb = self.feature_embedding(x) + self.pos_emb
        
        # 通过Encoder提取全局特征
        enc_out = self.encoder(x_emb)
        
        # 构造Decoder输入 (B, 1, d_model)
        dec_in = self.cls_token.expand(B, -1, -1)
        
        # 通过Decoder融合信息
        dec_out = self.decoder(dec_in, enc_out)
        
        # 降维输出分类结果
        logits = self.classifier(dec_out.squeeze(1)) # (B, 3)
        return logits, None


# 3. 训练与评估统一引擎
def train_and_eval(model, name, epochs=80):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss() # 仅用于自编码器
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, recon = model(batch_x)
            
            # 计算分类损失
            loss = criterion_cls(logits, batch_y)
            # 如果是自编码器，加入重构损失实现联合训练
            if recon is not None:
                loss += 0.5 * criterion_mse(recon, batch_x)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        history['train_loss'].append(epoch_loss / len(train_loader))
        
        # --- 评估阶段 ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits, _ = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        acc = correct / total
        history['test_acc'].append(acc)
        
    print(f"[{name}] 训练完成 | 最终测试集精度: {history['test_acc'][-1]*100:.2f}%")
    return history


# 4. 运行所有模型
models = {
    'DNN': DNN(),
    'CNN': CNN(),
    'LSTM': LSTMModel(),
    'AutoEncoder': AutoEncoder(),
    'Transformer': TransformerArchitecture()
}

results_history = {}
final_accuracies = {}

print("开始训练各深度学习模型...\n")
for name, model in models.items():
    history = train_and_eval(model, name)
    results_history[name] = history
    final_accuracies[name] = history['test_acc'][-1]

# 找出表现最好的模型获取混淆矩阵
best_model_name = max(final_accuracies, key=final_accuracies.get)
best_model = models[best_model_name]
best_model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits, _ = best_model(x)
        all_preds.extend(torch.argmax(logits, dim=1).numpy())
        all_trues.extend(y.numpy())


# 5. 可视化图表呈现
print("\n正在生成可视化图表...")
fig = plt.figure(figsize=(16, 10))

# 图1：模型精度对比柱状图
ax1 = plt.subplot(2, 2, 1)
colors = sns.color_palette("husl", len(models))
bars = ax1.bar(final_accuracies.keys(), final_accuracies.values(), color=colors)
ax1.set_title('各模型在测试集上的最终精度对比', fontsize=14)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Accuracy')
# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

# 图2：训练Loss下降曲线
ax2 = plt.subplot(2, 2, 2)
for name, hist in results_history.items():
    ax2.plot(hist['train_loss'], label=name, linewidth=2)
ax2.set_title('训练集 Loss 下降曲线', fontsize=14)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# 图3：测试集精度上升曲线
ax3 = plt.subplot(2, 2, 3)
for name, hist in results_history.items():
    ax3.plot(hist['test_acc'], label=name, linewidth=2)
ax3.set_title('测试集 Accuracy 变化曲线', fontsize=14)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

# 图4：最佳模型的混淆矩阵
ax4 = plt.subplot(2, 2, 4)
cm = confusion_matrix(all_trues, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=target_names, yticklabels=target_names)
ax4.set_title(f'混淆矩阵 (最佳模型: {best_model_name})', fontsize=14)
ax4.set_xlabel('预测标签')
ax4.set_ylabel('真实标签')

plt.tight_layout()
plt.show()


# 6. SHAP 模型解释性分析
print("\n=== SHAP 模型可解释性分析 ===")
print("正在计算SHAP值 (这可能会花费几秒钟)...")

# SHAP对纯PyTorch模型比较敏感，最好写一个包装函数转换为Numpy计算
def predict_fn_for_shap(x_numpy):
    # 将输入的numpy数组转换为tensor喂给最佳模型
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = best_model(x_tensor)
        # 返回概率分布
        probs = torch.softmax(logits, dim=1).numpy()
    return probs

# 使用 KernelExplainer (最通用、最不会报错的SHAP解释器)
# 背景数据集（使用部分训练集作为基准）
background_data = X_train[:40].numpy()
explainer = shap.KernelExplainer(predict_fn_for_shap, background_data)

# 取测试集前20个样本进行解释计算
test_samples = X_test[:20].numpy()
shap_values = explainer.shap_values(test_samples, silent=True)

# 1. 特征重要性摘要图 (多分类的整体影响)
plt.figure()
plt.title(f"SHAP 特征重要性汇总 ({best_model_name})", fontsize=14)
# shap_values 返回的是个列表，里面有3个类别对应的SHAP矩阵
shap.summary_plot(shap_values, test_samples, feature_names=feature_names, 
                  class_names=list(target_names), show=False)
plt.show()

# 2. 针对第0个样本，预测为某一类别(比如类别0 Setosa)的瀑布图/受力图
plt.figure()
plt.title(f"样本0被判定为 {target_names[0]} 的特征推导图", fontsize=14)
# 在SHAP 0.42.1中，KernelExplainer生成力导向图最稳定
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], test_samples[0,:], 
                feature_names=feature_names, matplotlib=True, show=False)
plt.show()

print("\n程序全部运行完毕！")