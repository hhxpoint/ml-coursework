import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def find_chinese_font():
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
        "C:/Windows/Fonts/simfang.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    for font in fm.fontManager.ttflist:
        name = font.name.lower()
        if any(k in name for k in ["yahei", "microsoft yahei", "simhei", "heiti", "songti", "kaiti", "fangsong"]):
            return font.fname
    return None
FONT_PATH = find_chinese_font()
if FONT_PATH:
    CHINESE_FONT = fm.FontProperties(fname=FONT_PATH)
else:
    CHINESE_FONT = None
plt.rcParams["axes.unicode_minus"] = False
if FONT_PATH and "msyh" in FONT_PATH.lower():
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
elif FONT_PATH and "simhei" in FONT_PATH.lower():
    plt.rcParams["font.family"] = ["SimHei"]


# 西瓜数据集 3.0
DATA = [
    {"编号": 1, "色泽": "青绿", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.697, "含糖率": 0.460, "好瓜": "是"},
    {"编号": 2, "色泽": "乌黑", "根蒂": "蜷缩", "敲声": "沉闷", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.774, "含糖率": 0.376, "好瓜": "是"},
    {"编号": 3, "色泽": "乌黑", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.634, "含糖率": 0.264, "好瓜": "是"},
    {"编号": 4, "色泽": "青绿", "根蒂": "蜷缩", "敲声": "沉闷", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.608, "含糖率": 0.318, "好瓜": "是"},
    {"编号": 5, "色泽": "浅白", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑", "密度": 0.556, "含糖率": 0.215, "好瓜": "是"},
    {"编号": 6, "色泽": "青绿", "根蒂": "稍蜷", "敲声": "浊响", "纹理": "清晰", "脐部": "稍凹", "触感": "软粘", "密度": 0.403, "含糖率": 0.237, "好瓜": "是"},
    {"编号": 7, "色泽": "乌黑", "根蒂": "稍蜷", "敲声": "浊响", "纹理": "稍糊", "脐部": "稍凹", "触感": "软粘", "密度": 0.481, "含糖率": 0.149, "好瓜": "是"},
    {"编号": 8, "色泽": "乌黑", "根蒂": "稍蜷", "敲声": "浊响", "纹理": "清晰", "脐部": "稍凹", "触感": "硬滑", "密度": 0.437, "含糖率": 0.211, "好瓜": "是"},
    {"编号": 9, "色泽": "乌黑", "根蒂": "稍蜷", "敲声": "沉闷", "纹理": "稍糊", "脐部": "稍凹", "触感": "硬滑", "密度": 0.666, "含糖率": 0.091, "好瓜": "否"},
    {"编号": 10, "色泽": "青绿", "根蒂": "硬挺", "敲声": "清脆", "纹理": "清晰", "脐部": "平坦", "触感": "软粘", "密度": 0.243, "含糖率": 0.267, "好瓜": "否"},
    {"编号": 11, "色泽": "浅白", "根蒂": "硬挺", "敲声": "清脆", "纹理": "模糊", "脐部": "平坦", "触感": "硬滑", "密度": 0.245, "含糖率": 0.057, "好瓜": "否"},
    {"编号": 12, "色泽": "浅白", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "模糊", "脐部": "平坦", "触感": "软粘", "密度": 0.343, "含糖率": 0.099, "好瓜": "否"},
    {"编号": 13, "色泽": "青绿", "根蒂": "稍蜷", "敲声": "浊响", "纹理": "稍糊", "脐部": "凹陷", "触感": "硬滑", "密度": 0.639, "含糖率": 0.161, "好瓜": "否"},
    {"编号": 14, "色泽": "浅白", "根蒂": "稍蜷", "敲声": "沉闷", "纹理": "稍糊", "脐部": "凹陷", "触感": "硬滑", "密度": 0.657, "含糖率": 0.198, "好瓜": "否"},
    {"编号": 15, "色泽": "乌黑", "根蒂": "稍蜷", "敲声": "浊响", "纹理": "清晰", "脐部": "稍凹", "触感": "软粘", "密度": 0.360, "含糖率": 0.370, "好瓜": "否"},
    {"编号": 16, "色泽": "浅白", "根蒂": "蜷缩", "敲声": "浊响", "纹理": "模糊", "脐部": "平坦", "触感": "硬滑", "密度": 0.593, "含糖率": 0.042, "好瓜": "否"},
    {"编号": 17, "色泽": "青绿", "根蒂": "蜷缩", "敲声": "沉闷", "纹理": "稍糊", "脐部": "稍凹", "触感": "硬滑", "密度": 0.719, "含糖率": 0.103, "好瓜": "否"},
]

DISCRETE_ATTRS = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]

ATTR_VALUES = {
    "色泽": ["青绿", "乌黑", "浅白"],
    "根蒂": ["蜷缩", "稍蜷", "硬挺"],
    "敲声": ["浊响", "沉闷", "清脆"],
    "纹理": ["清晰", "稍糊", "模糊"],
    "脐部": ["凹陷", "稍凹", "平坦"],
    "触感": ["硬滑", "软粘"],
}

LABEL = "好瓜"
CLASSES = ["是", "否"]
CLASS_NAMES = {"是": "好瓜", "否": "坏瓜"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "hw4_outputs")
os.makedirs(OUT_DIR, exist_ok=True)



# 朴素贝叶斯分类器
class NaiveBayes:
    def __init__(self, laplace=False):
        self.laplace = laplace
        self.priors = {}
        self.conditionals = {}
        self.class_counts = {}
        self.total_count = 0

    def fit(self, data, attrs):
        self.total_count = len(data)
        class_data = {c: [] for c in CLASSES}
        for sample in data:
            class_data[sample[LABEL]].append(sample)

        for c in CLASSES:
            self.class_counts[c] = len(class_data[c])
            if self.laplace:
                self.priors[c] = (self.class_counts[c] + 1) / (self.total_count + len(CLASSES))
            else:
                self.priors[c] = self.class_counts[c] / self.total_count

        self.conditionals = {c: {} for c in CLASSES}
        for c in CLASSES:
            samples = class_data[c]
            n_c = len(samples)
            for attr in attrs:
                self.conditionals[c][attr] = {}
                values = ATTR_VALUES[attr]
                for v in values:
                    count = sum(1 for s in samples if s[attr] == v)
                    if self.laplace:
                        n_i = len(values)
                        prob = (count + 1) / (n_c + n_i)
                    else:
                        prob = count / n_c if n_c > 0 else 0
                    self.conditionals[c][attr][v] = prob

    def predict(self, sample, attrs):
        scores = {}
        for c in CLASSES:
            log_prob = math.log(self.priors[c])
            for attr in attrs:
                v = sample[attr]
                p = self.conditionals[c][attr].get(v, 1e-10)
                log_prob += math.log(p) if p > 0 else math.log(1e-10)
            scores[c] = log_prob
        best_class = max(scores, key=scores.get)
        return best_class, scores


def _title(ax_or_fig, text, **kwargs):
    if CHINESE_FONT:
        return ax_or_fig.set_title(text, fontproperties=CHINESE_FONT, **kwargs)
    return ax_or_fig.set_title(text, **kwargs)


def _suptitle(fig, text, **kwargs):
    if CHINESE_FONT:
        return fig.suptitle(text, fontproperties=CHINESE_FONT, **kwargs)
    return fig.suptitle(text, **kwargs)


def _label(ax, xlabel=None, ylabel=None):
    if xlabel:
        if CHINESE_FONT:
            ax.set_xlabel(xlabel, fontproperties=CHINESE_FONT)
        else:
            ax.set_xlabel(xlabel)
    if ylabel:
        if CHINESE_FONT:
            ax.set_ylabel(ylabel, fontproperties=CHINESE_FONT)
        else:
            ax.set_ylabel(ylabel)


def _xticks(ax, labels):
    if CHINESE_FONT:
        ax.set_xticklabels(labels, fontproperties=CHINESE_FONT)
    else:
        ax.set_xticklabels(labels)


def _annotate(ax, text, *args, **kwargs):
    if CHINESE_FONT:
        kwargs.setdefault("fontproperties", CHINESE_FONT)
    return ax.annotate(text, *args, **kwargs)


def _legend(ax, *args, **kwargs):
    if CHINESE_FONT:
        kwargs.setdefault("prop", CHINESE_FONT)
    return ax.legend(*args, **kwargs)


def plot_7_1_mle(mle_model):
    top3_attrs = DISCRETE_ATTRS[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)

    for idx, attr in enumerate(top3_attrs):
        ax = axes[idx]
        values = ATTR_VALUES[attr]
        x = np.arange(len(values))
        width = 0.35

        probs_yes = [mle_model.conditionals["是"][attr][v] for v in values]
        probs_no = [mle_model.conditionals["否"][attr][v] for v in values]

        bars1 = ax.bar(x - width / 2, probs_yes, width, label="好瓜", color="#2ecc71", edgecolor="black")
        bars2 = ax.bar(x + width / 2, probs_no, width, label="坏瓜", color="#e74c3c", edgecolor="black")

        _label(ax, ylabel="概率")
        _title(ax, attr, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        _xticks(ax, values)
        _legend(ax, fontsize=10)
        ax.set_ylim(0, 1.0)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    _suptitle(fig, "7.1 极大似然估计 - 前3个属性的类条件概率", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(OUT_DIR, "7_1_mle_probs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def save_text_image(lines, title, filename):
    fig = plt.figure(figsize=(12, 10), dpi=160)
    fig.patch.set_facecolor("white")
    plt.axis("off")

    if CHINESE_FONT:
        plt.title(title, fontproperties=CHINESE_FONT, fontsize=14, fontweight="bold", pad=20)
    else:
        plt.title(title, fontsize=14, fontweight="bold", pad=20)

    full_text = "\n".join(lines)
    if CHINESE_FONT:
        plt.text(0.02, 0.98, full_text, va="top", ha="left",
                 fontproperties=CHINESE_FONT, fontsize=9, linespacing=1.5)
    else:
        plt.text(0.02, 0.98, full_text, va="top", ha="left",
                 fontsize=9, linespacing=1.5)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def build_mle_text(model):
    lines = []
    lines.append("训练集: 西瓜数据集3.0 (好瓜8个, 坏瓜9个)")
    lines.append("")
    lines.append("【先验概率】")
    for c in CLASSES:
        lines.append(f"  P({CLASS_NAMES[c]}) = {model.priors[c]:.6f}")
    lines.append("")
    for attr in DISCRETE_ATTRS:
        lines.append(f"【属性: {attr}】")
        for c in CLASSES:
            probs = model.conditionals[c][attr]
            for v, p in probs.items():
                lines.append(f"  P({attr}={v} | {CLASS_NAMES[c]}) = {p:.6f}")
        lines.append("")
    return lines


def build_laplace_text(model):
    lines = []
    lines.append("训练集: 西瓜数据集3.0 (好瓜8个, 坏瓜9个)")
    lines.append("")
    lines.append("【先验概率】")
    for c in CLASSES:
        lines.append(f"  P({CLASS_NAMES[c]}) = {model.priors[c]:.6f}")
    lines.append("")
    for attr in DISCRETE_ATTRS:
        lines.append(f"【属性: {attr}】")
        for c in CLASSES:
            probs = model.conditionals[c][attr]
            for v, p in probs.items():
                lines.append(f"  P({attr}={v} | {CLASS_NAMES[c]}) = {p:.6f}")
        lines.append("")
    return lines



def plot_test1_comparison(mle_scores, laplace_scores, test1):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)

    categories = ["好瓜", "坏瓜"]
    colors = ["#2ecc71", "#e74c3c"]

    # MLE
    ax = axes[0]
    mle_vals = [math.exp(mle_scores["是"]), math.exp(mle_scores["否"])]
    bars = ax.bar(categories, mle_vals, color=colors, edgecolor="black", width=0.5)
    _label(ax, ylabel="后验概率比例值")
    _title(ax, "极大似然估计 (MLE)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(mle_vals) * 1.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.6f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    _annotate(ax, "判别结果: 好瓜", xy=(0.5, 0.85), xycoords="axes fraction",
              fontsize=12, ha="center", color="green", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

    # 拉普拉斯
    ax = axes[1]
    lap_vals = [math.exp(laplace_scores["是"]), math.exp(laplace_scores["否"])]
    bars = ax.bar(categories, lap_vals, color=colors, edgecolor="black", width=0.5)
    _label(ax, ylabel="后验概率比例值")
    _title(ax, "拉普拉斯修正", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(lap_vals) * 1.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.6f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    _annotate(ax, "判别结果: 好瓜", xy=(0.5, 0.85), xycoords="axes fraction",
              fontsize=12, ha="center", color="green", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

    sample_str = ", ".join([f"{k}={v}" for k, v in test1.items()])
    _suptitle(fig, f'7.3 "测1"样本判别结果对比\n({sample_str})', fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(OUT_DIR, "7_3_test1_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_zero_prob_demo(mle_model, laplace_model):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    attr = "根蒂"
    values = ATTR_VALUES[attr]
    x = np.arange(len(values))
    width = 0.35

    # MLE
    ax = axes[0]
    probs_yes = [mle_model.conditionals["是"][attr][v] for v in values]
    probs_no = [mle_model.conditionals["否"][attr][v] for v in values]

    bars1 = ax.bar(x - width / 2, probs_yes, width, label="好瓜", color="#2ecc71", edgecolor="black")
    bars2 = ax.bar(x + width / 2, probs_no, width, label="坏瓜", color="#e74c3c", edgecolor="black")

    _label(ax, ylabel="概率")
    _title(ax, f"极大似然估计 - {attr}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    _xticks(ax, values)
    _legend(ax, fontsize=10)
    ax.set_ylim(0, 0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    _annotate(ax, "零概率问题!", xy=(2 - width / 2, 0.01), xytext=(2 - width / 2, 0.25),
              fontsize=9, ha="center", color="red", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color="red"))

    # 拉普拉斯
    ax = axes[1]
    probs_yes_l = [laplace_model.conditionals["是"][attr][v] for v in values]
    probs_no_l = [laplace_model.conditionals["否"][attr][v] for v in values]

    bars1 = ax.bar(x - width / 2, probs_yes_l, width, label="好瓜", color="#2ecc71", edgecolor="black")
    bars2 = ax.bar(x + width / 2, probs_no_l, width, label="坏瓜", color="#e74c3c", edgecolor="black")

    _label(ax, ylabel="概率")
    _title(ax, f"拉普拉斯修正 - {attr}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    _xticks(ax, values)
    _legend(ax, fontsize=10)
    ax.set_ylim(0, 0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    _annotate(ax, "修正后非零", xy=(2 - width / 2, 0.09), xytext=(2 - width / 2, 0.3),
              fontsize=9, ha="center", color="green", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color="green"))

    _suptitle(fig, "拉普拉斯修正解决零概率问题示意", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(OUT_DIR, "7_3_laplace_demo.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    test1 = {
        "色泽": "青绿", "根蒂": "蜷缩", "敲声": "浊响",
        "纹理": "清晰", "脐部": "凹陷", "触感": "硬滑"
    }


    nb_mle = NaiveBayes(laplace=False)
    nb_mle.fit(DATA, DISCRETE_ATTRS)
    pred_mle, scores_mle = nb_mle.predict(test1, DISCRETE_ATTRS)

    nb_laplace = NaiveBayes(laplace=True)
    nb_laplace.fit(DATA, DISCRETE_ATTRS)
    pred_laplace, scores_laplace = nb_laplace.predict(test1, DISCRETE_ATTRS)

    paths = []

    p1 = plot_7_1_mle(nb_mle)
    paths.append(p1)
    print(f"[1/5] 7.1 极大似然估计概率柱状图 -> {p1}")

    p2 = save_text_image(build_mle_text(nb_mle), "7.3 极大似然估计 (MLE) 类条件概率表", "7_3_mle_table.png")
    paths.append(p2)
    print(f"[2/5] MLE完整概率表 -> {p2}")

    p3 = save_text_image(build_laplace_text(nb_laplace), "7.3 拉普拉斯修正 类条件概率表", "7_3_laplace_table.png")
    paths.append(p3)
    print(f"[3/5] 拉普拉斯修正完整概率表 -> {p3}")

    p4 = plot_test1_comparison(scores_mle, scores_laplace, test1)
    paths.append(p4)
    print(f"[4/5] 测1判别结果对比图 -> {p4}")

    p5 = plot_zero_prob_demo(nb_mle, nb_laplace)
    paths.append(p5)
    print(f"[5/5] 零概率问题解决示意图 -> {p5}")

    print("\n" + "=" * 60)
    print("所有图片已生成，保存在:", OUT_DIR)
    print("=" * 60)
    for p in paths:
        print(f"  {os.path.basename(p)}")
