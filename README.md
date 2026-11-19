# 机器学习课程项目合集

> 本科机器学习 / 模式识别课程项目的源代码与实验结果归档：**PCA/LDA 人脸识别**、**猫狗分类**、**BirdCLEF 2026 鸟类声音识别**、**IRIS 分类**与 **ML 作业代码**。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2F6B5F)

> 仓库只保留**源代码、配置与小体积结果图**；原始照片、数据集、模型权重、报告 docx 不纳入，避免隐私与仓库膨胀。

---

## 子项目

### 1. `face/` — PCA + LDA 人脸识别

- 预处理、数据集可视化、PCA 降维、LDA 判别、MLP 分类
- 结果图：`results/`、`visualization/`
- 运行：

```bash
cd face
pip install numpy opencv-python scikit-learn matplotlib
python preprocess.py      # 预处理（需要 raw_photos，仓库未包含）
python train.py           # PCA/LDA + 分类实验
python generate_dataset_viz.py
```

> 原始人脸照片因隐私未开源；请使用自己的数据集放入 `raw_photos/`。

### 2. `catdog/` — 猫狗分类

- 基础实验与进阶实验脚本
- 结果图：`experiment_results.png`、`report_curves.png`、`report_confusion_matrix.png`、`report_predictions.png`、`system_flowchart.png`
- 运行：

```bash
cd catdog
pip install tensorflow numpy matplotlib scikit-learn
python catdog_experiment.py
python advanced_catdog_experiment.py
```

> CIFAR-10 数据与训练报告未入库；脚本会自动下载/读取数据集。

### 3. `birdclef2026/` — BirdCLEF 2026 鸟类声音识别

- Kaggle 训练/推理代码、配置、Notebook、方案说明
- 入口：`kaggle_train.py`、`train_notebook.py`
- 方案：`BIRDCLEF2026_SOLUTION.md`
- 运行：

```bash
cd birdclef2026
pip install -r requirements.txt
python kaggle_train.py --config configs/xxx.yaml
```

> 音频数据、checkpoints、submissions 未入库；需从 Kaggle 竞赛页面下载。

### 4. `DL/` — 深度学习基础

- IRIS 数据集分类脚本、实验代码
- 运行：`python DL/iris.py`

### 5. `ML/` — 机器学习作业

- 课程作业代码与小型输出（`hw*_outputs/`）
- 按作业说明运行对应脚本

---

## 环境建议

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install numpy pandas matplotlib scikit-learn
# 深度学习项目额外安装
pip install tensorflow opencv-python
```

---

## 测试与验收

仓库以课程实验代码为主，未附带统一单元测试；建议按子项目做冒烟检查：

```bash
python -m py_compile face/*.py catdog/*.py birdclef2026/*.py DL/*.py
```

手工验收：

- [ ] `face/train.py` 能读取自己的数据集并输出 PCA/LDA 结果
- [ ] `catdog/catdog_experiment.py` 能加载数据并训练
- [ ] `birdclef2026/kaggle_train.py` 能解析配置并启动
- [ ] `DL/iris.py` 能输出分类结果

---

## 数据与隐私

- 人脸原始照片、含姓名/学号的报告 docx、数据集、模型权重 **未纳入仓库**。
- `face/` 需要自备人脸数据；请遵守数据来源与肖像权要求。
- `catdog/`、`birdclef2026/` 需自行下载公开数据集。
- 代码中的路径按原始实验环境编写，换机器后请调整为本地路径。

---

## 致谢

- 课程实验基于 scikit-learn、TensorFlow/Keras、OpenCV、NumPy、Matplotlib 等开源库。
- BirdCLEF 数据与竞赛规则归 Kaggle/主办方所有。

---

## License

代码 MIT License，详见 [LICENSE](LICENSE)。数据集、预训练模型、课程报告遵循各自来源许可。
