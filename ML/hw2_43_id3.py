import os
import math
import numpy as np
import matplotlib.pyplot as plt


# 西瓜数据集 3.0（表4.3）
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

ATTRS = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率"]
CONT_ATTRS = {"密度", "含糖率"}
LABEL = "好瓜"

OUT_DIR = os.path.join(os.path.dirname(__file__), "hw2_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def entropy(samples):
    n = len(samples)
    if n == 0:
        return 0.0
    counts = {}
    for s in samples:
        y = s[LABEL]
        counts[y] = counts.get(y, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def majority_label(samples):
    counts = {}
    for s in samples:
        y = s[LABEL]
        counts[y] = counts.get(y, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def split_discrete(samples, attr):
    groups = {}
    for s in samples:
        v = s[attr]
        groups.setdefault(v, []).append(s)
    return groups


def split_continuous(samples, attr, threshold):
    left = [s for s in samples if s[attr] <= threshold]
    right = [s for s in samples if s[attr] > threshold]
    return left, right


def best_threshold_and_gain(samples, attr, base_ent):
    values = sorted(set(s[attr] for s in samples))
    if len(values) <= 1:
        return None, 0.0

    best_thr = None
    best_gain = -1.0
    n = len(samples)

    for i in range(len(values) - 1):
        thr = (values[i] + values[i + 1]) / 2.0
        left, right = split_continuous(samples, attr, thr)
        if len(left) == 0 or len(right) == 0:
            continue

        cond_ent = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
        gain = base_ent - cond_ent
        if gain > best_gain:
            best_gain = gain
            best_thr = thr

    if best_thr is None:
        return None, 0.0
    return best_thr, best_gain


def choose_best_attr(samples, candidate_attrs):
    base_ent = entropy(samples)
    gain_table = []

    best_attr = None
    best_gain = -1.0
    best_thr = None
    best_is_cont = False

    for attr in candidate_attrs:
        if attr in CONT_ATTRS:
            thr, gain = best_threshold_and_gain(samples, attr, base_ent)
            gain_table.append((attr, gain, thr))
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_thr = thr
                best_is_cont = True
        else:
            groups = split_discrete(samples, attr)
            cond_ent = 0.0
            n = len(samples)
            for g in groups.values():
                cond_ent += (len(g) / n) * entropy(g)
            gain = base_ent - cond_ent
            gain_table.append((attr, gain, None))
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_thr = None
                best_is_cont = False

    gain_table.sort(key=lambda x: x[1], reverse=True)
    return best_attr, best_gain, best_thr, best_is_cont, gain_table


def build_tree(samples, candidate_attrs, depth=0, max_depth=8):
    labels = [s[LABEL] for s in samples]

    # 纯节点
    if len(set(labels)) == 1:
        return {"type": "leaf", "label": labels[0], "samples": len(samples)}

    # 属性耗尽或达到深度上限
    if len(candidate_attrs) == 0 or depth >= max_depth:
        return {"type": "leaf", "label": majority_label(samples), "samples": len(samples)}

    best_attr, best_gain, best_thr, is_cont, _ = choose_best_attr(samples, candidate_attrs)

    # 信息增益过小时停止
    if best_attr is None or best_gain <= 1e-12:
        return {"type": "leaf", "label": majority_label(samples), "samples": len(samples)}

    node = {
        "type": "node",
        "attr": best_attr,
        "gain": best_gain,
        "is_cont": is_cont,
        "threshold": best_thr,
        "samples": len(samples),
        "children": {},
        "majority": majority_label(samples),
    }

    if is_cont:
        left, right = split_continuous(samples, best_attr, best_thr)
        if len(left) == 0 or len(right) == 0:
            return {"type": "leaf", "label": majority_label(samples), "samples": len(samples)}

        # 连续属性允许后续再次使用
        node["children"]["<="] = build_tree(left, candidate_attrs, depth + 1, max_depth)
        node["children"][">"] = build_tree(right, candidate_attrs, depth + 1, max_depth)
    else:
        groups = split_discrete(samples, best_attr)
        next_attrs = [a for a in candidate_attrs if a != best_attr]
        for v, subset in groups.items():
            node["children"][str(v)] = build_tree(subset, next_attrs, depth + 1, max_depth)

    return node


def predict_one(tree, sample):
    if tree["type"] == "leaf":
        return tree["label"]

    attr = tree["attr"]
    if tree["is_cont"]:
        if sample[attr] <= tree["threshold"]:
            return predict_one(tree["children"]["<="], sample)
        return predict_one(tree["children"][">"], sample)

    value = str(sample[attr])
    if value not in tree["children"]:
        return tree["majority"]
    return predict_one(tree["children"][value], sample)


def tree_to_lines(tree, indent=""):
    lines = []
    if tree["type"] == "leaf":
        lines.append(f"{indent}叶节点 -> 类别: {tree['label']} (样本数={tree['samples']})")
        return lines

    if tree["is_cont"]:
        lines.append(
            f"{indent}[划分属性: {tree['attr']} <= {tree['threshold']:.3f}, 信息增益={tree['gain']:.4f}, 样本数={tree['samples']}]"
        )
        lines.append(f"{indent}  分支: <=")
        lines.extend(tree_to_lines(tree["children"]["<="], indent + "    "))
        lines.append(f"{indent}  分支: >")
        lines.extend(tree_to_lines(tree["children"][">"] , indent + "    "))
    else:
        lines.append(
            f"{indent}[划分属性: {tree['attr']}, 信息增益={tree['gain']:.4f}, 样本数={tree['samples']}]"
        )
        for v, child in tree["children"].items():
            lines.append(f"{indent}  分支: {tree['attr']} = {v}")
            lines.extend(tree_to_lines(child, indent + "    "))

    return lines


def save_text_image(text, title, path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.text(0.01, 0.97, text, va="top", ha="left", fontsize=10, fontfamily="sans-serif")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_root_gain_bar(gain_table, path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    attrs = [x[0] for x in gain_table]
    gains = [x[1] for x in gain_table]

    fig = plt.figure(figsize=(9, 5), dpi=160)
    bars = plt.bar(attrs, gains, color="steelblue", alpha=0.85)
    plt.title("4.3 根节点各属性信息增益对比")
    plt.xlabel("属性")
    plt.ylabel("信息增益")
    plt.grid(axis="y", alpha=0.25)

    for b, g in zip(bars, gains):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{g:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    # 先计算根节点层面的信息增益，便于展示“按信息增益划分”的依据
    _, _, _, _, root_gain_table = choose_best_attr(DATA, ATTRS)

    tree = build_tree(DATA, ATTRS, max_depth=8)
    preds = [predict_one(tree, s) for s in DATA]
    labels = [s[LABEL] for s in DATA]
    acc = np.mean([p == y for p, y in zip(preds, labels)])

    tree_lines = tree_to_lines(tree)

    out_lines = []
    out_lines.append("--- 4.3 决策树(ID3, 信息增益) 训练结果 ---")
    out_lines.append("根节点候选属性信息增益（降序）:")
    for attr, gain, thr in root_gain_table:
        if thr is None:
            out_lines.append(f"  {attr}: Gain = {gain:.6f}")
        else:
            out_lines.append(f"  {attr}: Gain = {gain:.6f}, 最优阈值 = {thr:.3f}")

    out_lines.append("")
    out_lines.append(f"训练集准确率: {acc:.4f}")
    out_lines.append("")
    out_lines.append("生成的决策树结构:")
    out_lines.extend(tree_lines)

    output_text = "\n".join(out_lines)
    print(output_text)

    txt_path = os.path.join(OUT_DIR, "run_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    console_path = os.path.join(OUT_DIR, "4_3_console.png")
    tree_path = os.path.join(OUT_DIR, "4_3_tree.png")
    gain_path = os.path.join(OUT_DIR, "4_3_root_gain.png")

    save_text_image(output_text, "4.3 控制台输出结果", console_path)
    save_text_image("\n".join(tree_lines), "4.3 决策树结构可视化", tree_path)
    save_root_gain_bar(root_gain_table, gain_path)

    print("\n已保存文件:")
    print(console_path)
    print(tree_path)
    print(gain_path)
    print(txt_path)


if __name__ == "__main__":
    main()
