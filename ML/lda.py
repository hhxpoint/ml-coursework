import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103]
])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

X0 = X[y == 0] 
X1 = X[y == 1] 

# 类内均值向量 (mu0, mu1)
mu0 = np.mean(X0, axis=0).reshape(-1, 1)
mu1 = np.mean(X1, axis=0).reshape(-1, 1)

# 计算类内散度矩阵 Sw
Sw = np.zeros((X.shape[1], X.shape[1]))
for i in range(X0.shape[0]):
    x_i = X0[i].reshape(-1, 1)
    Sw += np.dot((x_i - mu0), (x_i - mu0).T)
for i in range(X1.shape[0]):
    x_i = X1[i].reshape(-1, 1)
    Sw += np.dot((x_i - mu1), (x_i - mu1).T)

# 最佳投影方向 w
# w = Sw^(-1) * (mu0 - mu1) 
Sw_inv = np.linalg.inv(Sw)
w = np.dot(Sw_inv, (mu0 - mu1))

print("--- 3.5 线性判别分析(LDA) 结果 ---")
print(f"求得的投影方向 w: \n{w}")


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.scatter(X1[:,0], X1[:,1], color='g', marker='o', label='好瓜 (1)')
plt.scatter(X0[:,0], X0[:,1], color='r', marker='x', label='坏瓜 (0)')

# 画出投影线（过原点，方向为 w）
# w[1]/w[0] 是直线的斜率
slope = w[1,0] / w[0,0]
x_line = np.linspace(0.1, 0.9, 100)
y_line = slope * x_line
plt.plot(x_line, y_line, color='blue', label='LDA投影方向线', linestyle='--')


for i in range(X.shape[0]):
    x_pt = X[i].reshape(-1,1)
    # 投影点计算
    proj_pt = (np.dot(w.T, x_pt) / np.dot(w.T, w)) * w
    plt.plot([x_pt[0,0], proj_pt[0,0]], [x_pt[1,0], proj_pt[1,0]], 'k:', alpha=0.3)

plt.title('西瓜数据集 3.0a - 线性判别分析(LDA)')
plt.xlabel('密度')
plt.ylabel('含糖率')
plt.axis('equal') 
plt.legend()
plt.show()