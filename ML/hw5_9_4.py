"""
机器学习第五次作业 - 9.4
K-均值聚类算法实现，DB指标与Dunn指标计算
西瓜数据集4.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from itertools import combinations

# 设置输出目录
OUT_DIR = os.path.join(os.path.dirname(__file__), "hw5_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


X = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]
])


# ========================================
# K-Means算法实现
# ========================================
class KMeans:
    def __init__(self, k, init_centers=None, max_iter=100, tol=1e-4):
        self.k = k
        self.init_centers = init_centers
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.labels = None
        self.inertia_ = None  # SSE
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化聚类中心
        if self.init_centers is not None:
            self.centers = np.array(self.init_centers)
        else:
            # 随机选择k个样本作为初始中心
            idx = np.random.choice(n_samples, self.k, replace=False)
            self.centers = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # 分配样本到最近的中心
            distances = pairwise_distances(X, self.centers, metric='euclidean')
            self.labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centers = np.zeros_like(self.centers)
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centers[i] = cluster_points.mean(axis=0)
                else:
                    new_centers[i] = self.centers[i]  # 保持原中心
            
            # 检查收敛
            center_shift = np.linalg.norm(new_centers - self.centers)
            self.centers = new_centers
            
            if center_shift < self.tol:
                break
        
        # 计算SSE (Sum of Squared Errors)
        self.inertia_ = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centers[i]) ** 2)
        
        return self


# ========================================
# DB指标 (Davies-Bouldin Index) 计算
# ========================================
def davies_bouldin_index(X, labels, centers):
    """
    DB指标: 簇内距离平均 / 簇间距离，越小越好
    """
    n_clusters = len(centers)
    if n_clusters < 2:
        return 0
    
    # 计算每个簇的平均距离（簇内紧密度）
    avg_distances = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            avg_distances[i] = np.mean(np.linalg.norm(cluster_points - centers[i], axis=1))
    
    # 计算DB指标
    db_sum = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # 簇间距离
                inter_dist = np.linalg.norm(centers[i] - centers[j])
                if inter_dist > 0:
                    # R_ij = (s_i + s_j) / d_ij
                    ratio = (avg_distances[i] + avg_distances[j]) / inter_dist
                    max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio
    
    return db_sum / n_clusters


# ========================================
# Dunn指标计算
# ========================================
def dunn_index(X, labels, centers):
    """
    Dunn指标: min(簇间最小距离) / max(簇内最大距离)，越大越好
    """
    n_clusters = len(centers)
    if n_clusters < 2:
        return 0
    
    # 计算簇内最大距离（簇直径）
    max_intra_dists = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            intra_dists = pairwise_distances(cluster_points, metric='euclidean')
            max_intra_dists.append(np.max(intra_dists))
        else:
            max_intra_dists.append(0)
    
    max_intra_dist = max(max_intra_dists) if max_intra_dists else 1e-10
    
    # 计算簇间最小距离
    min_inter_dist = float('inf')
    for i, j in combinations(range(n_clusters), 2):
        points_i = X[labels == i]
        points_j = X[labels == j]
        if len(points_i) > 0 and len(points_j) > 0:
            inter_dists = pairwise_distances(points_i, points_j, metric='euclidean')
            min_inter_dist = min(min_inter_dist, np.min(inter_dists))
    
    if max_intra_dist == 0 or min_inter_dist == float('inf'):
        return 0
    
    return min_inter_dist / max_intra_dist


# ========================================
# 实验: 不同k值和初始中心
# ========================================
def run_experiments():
    # 三组不同的k值
    k_values = [2, 3, 4]
    
    # 三组不同的初始中心设置策略
    init_strategies = [
        {
            "name": "随机选择",
            "description": "随机选择k个样本点作为初始中心",
            "centers": None  # 在fit时随机初始化
        },
        {
            "name": "边缘点选择",
            "description": "选择数据集中相距较远的点作为初始中心",
            "centers": {
                2: [[0.243, 0.267], [0.774, 0.376]],
                3: [[0.243, 0.267], [0.774, 0.376], [0.656, 0.091]],
                4: [[0.243, 0.267], [0.774, 0.376], [0.656, 0.091], [0.360, 0.370]]
            }
        },
        {
            "name": "密集区选择",
            "description": "选择数据密集区域的点作为初始中心",
            "centers": {
                2: [[0.556, 0.215], [0.639, 0.161]],
                3: [[0.556, 0.215], [0.639, 0.161], [0.608, 0.318]],
                4: [[0.556, 0.215], [0.639, 0.161], [0.608, 0.318], [0.697, 0.460]]
            }
        }
    ]
    
    results = []
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(len(k_values), len(init_strategies), figsize=(15, 12), dpi=160)
    
    for i, k in enumerate(k_values):
        for j, strategy in enumerate(init_strategies):
            # 初始化并训练
            if strategy["centers"] is None:
                km = KMeans(k=k, init_centers=None)
            else:
                km = KMeans(k=k, init_centers=strategy["centers"][k])
            
            km.fit(X)
            
            # 计算指标
            db = davies_bouldin_index(X, km.labels, km.centers)
            dunn = dunn_index(X, km.labels, km.centers)
            sse = km.inertia_
            
            results.append({
                'k': k,
                'strategy': strategy["name"],
                'description': strategy["description"],
                'DB': db,
                'Dunn': dunn,
                'SSE': sse,
                'labels': km.labels.copy(),
                'centers': km.centers.copy()
            })
            
            # 可视化
            ax = axes[i, j]
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            for cluster_id in range(k):
                cluster_points = X[km.labels == cluster_id]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[cluster_id], marker='o', s=50, alpha=0.6,
                          label=f'簇{cluster_id+1}')
            
            ax.scatter(km.centers[:, 0], km.centers[:, 1], 
                      c='black', marker='*', s=200, edgecolors='white', linewidths=1.5,
                      label='中心点')
            
            ax.set_title(f'k={k}, {strategy["name"]}\nDB={db:.4f}, Dunn={dunn:.4f}')
            ax.set_xlabel('密度')
            ax.set_ylabel('含糖率')
            ax.legend(loc='best', fontsize=7)
    
    plt.suptitle('9.4 K-Means聚类实验结果（不同k值与初始中心）', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, '9_4_kmeans_comparison.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    return results, fig_path


# ========================================
# 生成结果报告
# ========================================
def generate_report(results):
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("9.4 K-均值聚类实验结果")
    report_lines.append("西瓜数据集 4.0 (30个样本)")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # 按k值分组
    for k in [2, 3, 4]:
        report_lines.append(f"\n【k = {k}】")
        report_lines.append("-" * 40)
        k_results = [r for r in results if r['k'] == k]
        
        for r in k_results:
            report_lines.append(f"\n初始中心策略: {r['strategy']}")
            report_lines.append(f"  描述: {r['description']}")
            report_lines.append(f"  SSE (误差平方和): {r['SSE']:.6f}")
            report_lines.append(f"  DB 指标 (Davies-Bouldin): {r['DB']:.6f}  (越小越好)")
            report_lines.append(f"  Dunn 指标: {r['Dunn']:.6f}  (越大越好)")
            report_lines.append(f"  聚类中心:")
            for idx, center in enumerate(r['centers']):
                report_lines.append(f"    中心{idx+1}: ({center[0]:.3f}, {center[1]:.3f})")
    
    report_lines.append("\n" + "=" * 60)
    report_lines.append("【分析与结论】")
    report_lines.append("=" * 60)
    report_lines.append("""
1. k值选择:
   - k=2: 将西瓜分为好瓜/坏瓜两类，聚类边界清晰
   - k=3: 能发现数据中的中间状态样本
   - k=4: 可能过度划分，某些簇样本过少

2. 初始中心影响:
   - 随机选择: 结果可能不稳定，容易陷入局部最优
   - 边缘点选择: 有利于发现明显的类别分界
   - 密集区选择: 收敛速度快，对凸形分布效果好

3. 指标解释:
   - DB指标衡量簇内紧密度与簇间分离度的比值，越小越好
   - Dunn指标衡量最小簇间距离与最大簇内直径的比值，越大越好
   
4. 结论:
   合适的初始中心应该分散在数据的不同区域，避免过于集中。
   对于西瓜数据集，k=3配合"边缘点选择"策略通常能取得较好的聚类效果。
""")
    
    return "\n".join(report_lines)


# ========================================
# 主程序
# ========================================
if __name__ == "__main__":
    print("开始运行9.4 K-均值聚类实验...")
    
    # 运行实验
    results, fig_path = run_experiments()
    
    # 生成报告
    report = generate_report(results)
    print(report)
    
    # 保存报告
    report_path = os.path.join(OUT_DIR, '9_4_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n\n结果已保存:")
    print(f"  可视化图片: {fig_path}")
    print(f"  实验报告: {report_path}")
