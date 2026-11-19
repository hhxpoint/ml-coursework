import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score


# 西瓜数据集 3.0a
X = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103]
])
y_cls = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 6.8 使用“密度”作为输入，“含糖率”作为输出
X_reg = X[:, 0].reshape(-1, 1)
y_reg = X[:, 1]


OUT_DIR = os.path.join(os.path.dirname(__file__), "hw3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def save_text_as_image(text, save_path, title):
    fig = plt.figure(figsize=(10, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.title(title, fontsize=13)
    plt.text(
        0.01,
        0.95,
        text,
        va="top",
        ha="left",
        fontfamily="sans-serif",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def solve_6_2():
    # 使用接近硬间隔的 C，便于展示支持向量差异
    svm_linear = SVC(kernel="linear", C=1000)
    svm_rbf = SVC(kernel="rbf", C=1000, gamma="scale")

    svm_linear.fit(X, y_cls)
    svm_rbf.fit(X, y_cls)

    linear_sv_count = svm_linear.n_support_
    rbf_sv_count = svm_rbf.n_support_

    result_text = []
    result_text.append("--- 6.2 SVM 训练结果（西瓜数据集 3.0a）---")
    result_text.append("线性核 SVM:")
    result_text.append(f"  每类支持向量个数: {linear_sv_count}")
    result_text.append(f"  支持向量总数: {svm_linear.support_vectors_.shape[0]}")
    result_text.append(f"  支持向量下标: {svm_linear.support_}")
    result_text.append("")
    result_text.append("高斯核(RBF) SVM:")
    result_text.append(f"  每类支持向量个数: {rbf_sv_count}")
    result_text.append(f"  支持向量总数: {svm_rbf.support_vectors_.shape[0]}")
    result_text.append(f"  支持向量下标: {svm_rbf.support_}")

    text = "\n".join(result_text)

    # 可视化
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    x_min, x_max = X[:, 0].min() - 0.06, X[:, 0].max() + 0.06
    y_min, y_max = X[:, 1].min() - 0.06, X[:, 1].max() + 0.06
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)

    for ax, clf, title in [
        (axes[0], svm_linear, "线性核 SVM"),
        (axes[1], svm_rbf, "高斯核(RBF) SVM"),
    ]:
        zz = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.2, cmap="coolwarm")
        ax.scatter(X[y_cls == 1, 0], X[y_cls == 1, 1], c="green", marker="o", label="好瓜(1)")
        ax.scatter(X[y_cls == 0, 0], X[y_cls == 0, 1], c="red", marker="x", label="坏瓜(0)")

        sv = clf.support_vectors_
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=160,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="支持向量",
        )

        ax.set_title(title)
        ax.set_xlabel("密度")
        ax.set_ylabel("含糖率")
        ax.legend(loc="best")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.suptitle("6.2 线性核与高斯核 SVM 对比")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, "6_2_svm_compare.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    console_img_path = os.path.join(OUT_DIR, "6_2_console.png")
    save_text_as_image(text, console_img_path, "6.2 控制台输出结果")

    return text, fig_path, console_img_path


def solve_6_8():
    # sklearn 的 SVR 底层就是 LIBSVM 实现
    svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.02)
    svr.fit(X_reg, y_reg)

    y_pred = svr.predict(X_reg)
    mse = mean_squared_error(y_reg, y_pred)
    r2 = r2_score(y_reg, y_pred)

    result_text = []
    result_text.append("--- 6.8 SVR 训练结果（西瓜数据集 3.0a）---")
    result_text.append("输入特征: 密度")
    result_text.append("输出目标: 含糖率")
    result_text.append("核函数: RBF")
    result_text.append(f"支持向量个数: {svr.support_.shape[0]}")
    result_text.append(f"支持向量下标: {svr.support_}")
    result_text.append(f"训练集 MSE: {mse:.6f}")
    result_text.append(f"训练集 R^2: {r2:.6f}")

    text = "\n".join(result_text)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    x_plot = np.linspace(X_reg.min() - 0.03, X_reg.max() + 0.03, 400).reshape(-1, 1)
    y_plot = svr.predict(x_plot)

    fig = plt.figure(figsize=(9, 5), dpi=160)
    plt.scatter(X_reg[:, 0], y_reg, c="dodgerblue", s=45, label="原始样本")
    plt.plot(x_plot[:, 0], y_plot, c="darkorange", lw=2, label="SVR拟合曲线")

    # 标记支持向量
    plt.scatter(
        X_reg[svr.support_, 0],
        y_reg[svr.support_],
        s=150,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.5,
        label="支持向量",
    )

    # epsilon 管道可视化
    plt.plot(x_plot[:, 0], y_plot + svr.epsilon, "k--", alpha=0.5, label="epsilon管道")
    plt.plot(x_plot[:, 0], y_plot - svr.epsilon, "k--", alpha=0.5)

    plt.title("6.8 西瓜数据集 3.0a 的 SVR 回归结果")
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, "6_8_svr.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    console_img_path = os.path.join(OUT_DIR, "6_8_console.png")
    save_text_as_image(text, console_img_path, "6.8 控制台输出结果")

    return text, fig_path, console_img_path


if __name__ == "__main__":
    text_62, fig_62, console_62 = solve_6_2()
    text_68, fig_68, console_68 = solve_6_8()

    all_text = text_62 + "\n\n" + text_68
    print(all_text)

    txt_path = os.path.join(OUT_DIR, "run_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print("\n已保存文件:")
    print(fig_62)
    print(console_62)
    print(fig_68)
    print(console_68)
    print(txt_path)
