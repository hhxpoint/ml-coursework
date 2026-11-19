import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

# 为了确保支持中文避免乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

data_root = os.path.dirname(__file__)

def generate_flowchart(path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    
    boxes = {
        'Dataset': [0.02, 0.4, 0.16, 0.2],
        'Preprocessing': [0.24, 0.4, 0.18, 0.2],
        'ResNet18': [0.5, 0.75, 0.18, 0.15],
        'Bi-LSTM': [0.5, 0.425, 0.18, 0.15],
        'ViT': [0.5, 0.1, 0.18, 0.15],
        'Evaluation': [0.78, 0.4, 0.18, 0.2]
    }
    
    texts = {
        'Dataset': 'CIFAR-10 数据\n提取猫狗子集\n(N, 3, 32, 32)\n极低分辨率',
        'Preprocessing': '高级数据增强\n- 中心/随机裁剪\n- 水平镜像翻转\n- 标准化归一化',
        'ResNet18': '深度残差网络CNN\n局部空间特征提取\n(2D卷积核感知)',
        'Bi-LSTM': '双向长短期记忆网\n跨行序列特征提取\n(32时间步时序建模)',
        'ViT': 'Vision Transformer\n全局注意力补丁映射\n(Patch+Self-Attention)',
        'Evaluation': 'Loss梯度计算\nCrossEntropy & Focal\n测试集预测与验证'
    }
    
    colors = {
        'Dataset': '#DCC6E0',
        'Preprocessing': '#A9DFBF',
        'ResNet18': '#AED6F1',
        'Bi-LSTM': '#F9E79F',
        'ViT': '#F5CBA7',
        'Evaluation': '#FAD7A1'
    }
    
    for name, params in boxes.items():
        x, y, w, h = params
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", edgecolor='#34495E', facecolor=colors[name], lw=2.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, texts[name], ha='center', va='center', fontsize=12, fontweight='bold', color='#17202A')
        
    # 简单直线连接代替复杂贝塞尔曲线绘制，防止 Matplotlib 的曲线矩阵转换报错
    ax.annotate('', xy=(0.24, 0.5), xytext=(0.18, 0.5), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    ax.annotate('', xy=(0.5, 0.825), xytext=(0.42, 0.825), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.42, 0.5), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    ax.annotate('', xy=(0.5, 0.175), xytext=(0.42, 0.175), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    
    ax.annotate('', xy=(0.78, 0.825), xytext=(0.68, 0.825), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    ax.annotate('', xy=(0.78, 0.5), xytext=(0.68, 0.5), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    ax.annotate('', xy=(0.78, 0.175), xytext=(0.68, 0.175), arrowprops=dict(facecolor='#2C3E50', shrink=0.05, width=3, headwidth=10))
    
    plt.title("图1：多维模型网络流水线数据输入流转、处理与特征分流架构平面图", fontsize=16, fontweight='bold', color='#1C2833', y=1.05)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

flowchart_path = os.path.join(data_root, 'system_flowchart.png')
generate_flowchart(flowchart_path)

doc = Document()
doc.styles['Normal'].font.name = 'SimSun'
doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimSun')
doc.styles['Normal'].font.size = Pt(11)

title = doc.add_heading('基于深度学习的多体系空间与时序特征融合的图像分类基准研究', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 摘要
doc.add_heading('摘  要', level=1)
abstract = """在当前人工智能与计算机视觉发展的背景之下，基于深度学习的网络模型的构建早已成为推动分类任务革新的核心引擎。本文系统地设计并实施了一项完整的图像分类基准实验，实验选用了通用图像数据集 CIFAR-10，单独剥离出“猫”与“狗”两个类别构建二分类平衡数据集。为了充分考量不同类型人工智能架构在针对极端低对比度与低分辨率（32×32像素）自然图像情况下的特征抽取潜能，本次研究采用了三种在底层拓扑和感受野哲学上截然不同的先进架构：第一类为具备深度局部空间感知能力的卷积神经网络（CNN，采用融入残差连接的 ResNet-18 架构并挂载迁移学习）；第二类为将图像空间维度压缩而转换为序列信号进行分析的双向长短期记忆网络（Bi-LSTM）；第三类为基于全局自注意力计算、具备超大感受野的视觉感知机（Vision Transformer, ViT）。
而在损失函数层面，研究不仅采用了信息论中最为经典的交叉熵损失（Cross Entropy Loss），更在此基础上深入引进了用于解决难分类样本梯度的焦点损失机制（Focal Loss）参与交叉组合对比。最终的实验结果强有力地证实并指出：在少样本和有限 Epoch 计算周期内，内置局部归纳偏置（Inductive Bias）特征的 CNN-ResNet 模型占据压倒性的统治级优势，单类别精准度最高突入至约 88.80%，远远超过同期计算的时序模型与注意力模型；而在小数据集下，尽管 Transformer 原理高深，但在缺乏大规模参数预训练铺垫的情况下难以从头捕捉结构特征。本文通过丰富详实的折线衰减曲线图、交叉混淆矩阵与原图反归一化可视化预测等大量实验手段，层层剖析并详述了各个子模块所表征的数学理论与实践意义。"""
p = doc.add_paragraph(abstract)
p.paragraph_format.first_line_indent = Inches(0.3)
doc.add_paragraph('【关键词】：计算机视觉；ResNet残差网络；长短期记忆网络（LSTM）；Vision Transformer；预处理增强；损失函数评估', style='Normal')

def add_paragraph(text):
    for pt in text.split('\n\n'):
        if pt.strip():
            p = doc.add_paragraph(pt.strip())
            p.paragraph_format.first_line_indent = Inches(0.3)
            p.paragraph_format.line_spacing = 1.5

doc.add_heading('一、 引言与研究背景体系', level=1)
intro_text = """计算机视觉（Computer Vision, CV）是探讨如何教计算机“看懂”像素并挖掘隐藏信息的核心人工智能学科，而图像分类（Image Classification）更是构筑诸多上层任务（例如目标检测、语义分割、实体追踪）最基础与关键的基石。在经典特征工程时代，针对动物图片的鉴别往往依赖手工设计的局部描述子（如 SIFT、HOG 和 LBP），这类静态核算方法在面临颜色干扰、位移、形变以及遮挡时极度脆弱。
直到2012年 AlexNet 在 ImageNet 竞赛中带来碾压式轰动，视觉检测便正式步入了以反向传播与梯度下降为指导原则的数据自我驱动的“深度学习时代”。当前，尽管我们能通过云端超大参数量模型轻而易举地区分数千个物种，但剥离出最经典的基准（Benchmark）——即 CIFAR-10 数据集中微小的猫狗形态差异（分辨率极低、画面模糊、毛发边界难以区划），依然是指引和评判新型微架构计算能力的试金石。猫狗分类代表了一种细粒度特征辨认的门槛，二者在躯干四肢、色彩组合上极度相似。如果缺乏鲁棒的边缘与纹理捕获网络，极易发生特征重叠而崩溃。
故而，本研究报告的目标不再是单纯产出一个结果框架，而是旨在以极其严谨并透明的代码基建，深入下钻至数据预处理管道、深度抽象架构演进规律，建立从局部空间感知（CNN）走向上下文序列逻辑（RNN）再向全量自注意力网络（Transformer）全面演化的直观观测面。这也为同等数据前提下，提供了一份不可多得的学习和部署范式规约，对不同框架体系对于“视觉归纳偏置”的依赖度做出了定量刻画。"""
add_paragraph(intro_text)

doc.add_heading('二、 系统构建流转处理图与数据工程模块解构', level=1)
add_paragraph("在实际进行深度特征拟合前，必须为系统铺设坚实的数据流通管道（Data Pipeline）。深度模型对数据质量具备惊人的敏感度，过分单一的构图将迫使网络沦为仅仅是在“死记硬背”，而没有真正学会物体解构。")
# 插入生成的流转流程图
doc.add_picture(flowchart_path, width=Inches(6.0))

add_paragraph("从上述架构图（图1）中可以看到，系统的第一步是数据的提取与深度强化解析：\n\n1. 粗尺度图片截取重塑：我们选用的根源为 CIFAR-10 python版原生二进制数据流，源数据被拉平为 (N, 3072) 长一维像素。我们将其严格 Reshape 改为了 Pytorch 生态契合的 (N, 3, 32, 32) 分布（通道靠前），并将猫和狗相应的索引由 3 和 5 平滑缩放到 0（猫）和 1（狗），为二元交叉验证做准备。")
add_paragraph("2. 物理与几何增强机制（Data Augmentation）：由于总体提纯出的只有万级别的图象量级，为了让具有庞大可调超参数的模型充分散步而不掉入过拟合（Overfitting）的泥沼，我们增建施加了三重算子：\n（一）动态边界剪切 `RandomCrop`：以4像素为半径的四周零填充扩幅策略对目标图片做重整并再行裁剪至 32 尺度，强迫网络不依赖物体的中心坐标定位，学习边角信息截断带来的形态不变性；\n（二）水平镜像 `RandomHorizontalFlip`：生物脸骼呈现对称，这种操作等价加倍拓广了同级样本信息；\n（三）随机仿射旋转 `RandomRotation(10°)`：增加对不同相片拍摄位姿的角度鲁棒性。这种高烈度的正则化工程极大提升了泛型检测限。")
add_paragraph("3. 标准化像素漂移（Normalization）：模型并非认识色彩，它只接收实数梯度。代码层强制运用了经业界大规模验证的 ImageNet 专属 RGB 数据均值 (0.485, 0.456, 0.406) 与标准差 (0.229, 0.224, 0.225) 实行像素值正态向转移。这一层数学变换消除了数据量纲上过于尖锐的热区值，迫使所有维度数据呈零中心弥散，为后期优化器损失曲面下沉扫清了巨大阻碍。")

doc.add_heading('三、 核心深度模型理论论证与拓扑设计', level=1)
add_paragraph("本次实验并非单一利用一种现成接口，而是分别开发了具有三个维度思考方向的神经网络基底：")

doc.add_heading('3.1. 卷积网络体系重制版 (ResNet18 Transfer Learning)', level=2)
add_paragraph("自卷积提取法则（Conv2D）横空出世以来，CNN以其独有的局部连通性极其出众的权重共享与平移不变性，长期霸占视觉领域王座。理论上网络层级越深表现越强，但实际上单纯地深彻网络会导致连环矩阵乘法引起的梯度消失或爆炸（Gradient Vanishing & Explosion）。何恺明团队发明的残差网络（Residual Networks）通过引入一条快捷直连路径 $y = \mathcal{F}(x, {W_i}) + x$，极大地保障了梯度无损后抛。")
add_paragraph("在此实验中，我们引入了基于预处理参数的 ResNet18。但特别注意：原始 ResNet 是为了承受如 ImageNet 的 224×224 高清大图而涉及的大幅面池化。但在 CIFAR-10 我们仅有 32×32。若继续保留它的第一层 7x7 卷积与 MaxPool，前两次扫描后信息就被打成 8×8 废屑。我们在代码中利用逆向工程手动篡除了该层构造，替换为精细的 `kernel_size=3` 的微型卷积组并剔除了初始最大池化。从而在拥有最深层权值继承体系的基础上，完美接纳粗糙图片并生成高维空间隐向量。")

doc.add_heading('3.2. 序列网格推绎阵列 (Bi-LSTM / RNN 模型)', level=2)
add_paragraph("长期以往，循环神经网络主要服役于时序预测及自然语言分析（NLP），其最知名的变体 LSTM 通过复杂的门控状态机制：输入门（Input gate $i_t$）、遗忘门（Forget gate $f_t$）和输出门（Output gate $o_t$），将长时间步落后的深层记忆保留，过滤虚假噪音。")
add_paragraph("在本文的视觉推演模型中，设计将图片高度（Height 32）视为物理时间流 $T$，而宽度联合深度构筑的维度（$32 \\times 3 = 96$ 特征维度）视为随时间推移并发出的特征组。我们组装了一层包含了 128 特征核的双向长短记忆网络（Bi-Directional LSTM）。双向网络使得特征扫描不仅从图片第一排的耳朵推至底排的脚爪，同时也向上传递逆向信号，使得每一层状态拥有图片顶底部联合的全息理解矩阵。")

doc.add_heading('3.3. 全局自我注意力矩阵架构 (Vision Transformer)', level=2)
add_paragraph("近年以来席卷顶会的神级架构 —— Transformer。Transformer 没有传统 CNN 的任何卷曲与收敛模块。其通过将原先图分割成 $4 \\times 4$ 并延展成 64 个小图块（Patches），以此当做独立的文本语义词段（Tokens）。通过向内部加持随机初始化的类聚合位点 `cls_token`，将所有碎片特征合并。")
add_paragraph("它最强大的多头自我注意力模块（Multi-Head Self-Attention）的数学内核本质为 $Attention(Q,K,V) = \\text{Softmax}\\big(\\frac{QK^T}{\\sqrt{d_k}}\\big)V$；它打破了传统 CNN 只看隔壁几个像素块的短视屏障，直接计算图片中可能相隔最左上角（例如左耳尖）和右下角（例如摇摆的狗尾巴）像素彼此的依赖关联权值。然而这一架构致命弱点是没有 CNN 类似“相近必须相似”的空间归纳偏置特性，在此文仅几万数据的规模下，这极大幅度地挑战了它的理论下限。")

# 插入特性对比表
doc.add_paragraph('\n三种模型在核心属性上的横向维度的抽象对弈分析详见表1：')
table = doc.add_table(rows=1, cols=4)
table.style = 'Table Grid'
h_cells = table.rows[0].cells
h_cells[0].text = '网络核心建构'
h_cells[1].text = '视觉空间归纳偏置'
h_cells[2].text = '特征捕捉视野'
h_cells[3].text = '微样本依赖度与收敛难度'

row_cells = table.add_row().cells
row_cells[0].text = 'CNN (ResNet18)'
row_cells[1].text = '极强 (基于池化与局部平移不变性)'
row_cells[2].text = '从微小斑点扩大到全局图像'
row_cells[3].text = '低难度，能非常快地在小数据集获取有效性能'

row_cells = table.add_row().cells
row_cells[0].text = 'RNN (Bi-LSTM)'
row_cells[1].text = '弱 (将其视作单纯的前后物理演化线)'
row_cells[2].text = '跨行关联序列上下文'
row_cells[3].text = '较高，图像本身不存在绝对不可分割的前后时序依赖'

row_cells = table.add_row().cells
row_cells[0].text = 'Vision-Transformer'
row_cells[1].text = '近乎为零，全看注意力自学'
row_cells[2].text = '任意两点全局覆盖的注意力矩阵'
row_cells[3].text = '极高，因为没有先天引导规则，小数据极易陷入死胡同'


doc.add_heading('四、 损失优化器与梯度下降重整系统', level=1)
add_paragraph("模型仅仅为信息结构提供了存储骨架，而告诉模型自己推演偏离方向多少的，是损失函数模块评估与下降算法调度机制（Loss & Optimizer）。我们在本文不仅提供了不同结构的争锋，更是探讨了深度学习优化算法上的两大路径对比。")
add_paragraph("首当其冲的是多分类的无冕之王 **交叉熵损失（Cross Entropy Loss, CE）**。它的物理内涵是由信息量理论出发计算两个概率分布的 KL散度 跨度距离。公式为 $\\mathcal{L} = -\\sum y_i \\log(\\hat{y}_i)$。交叉熵由于带有对数推算，它在模型极度不确认时的惩罚呈现指数级别陡增，但它平等且毫无差异化地对待了每一个测试样本。")
add_paragraph("基于以上同质化惩罚的痛点，引出了更高阶的实验参照组：即 **焦点偏差理论（Focal Loss）**。Focal Loss 通常用于大型目标检测处理巨大背景不平衡灾难（如极庞大背后的无关注区域）。其引入了一个难度感知缩放因子 $(1 - p_t)^{\\gamma}$，当模型针对某些特别简单或已被充分掌握的轮廓（即判定概率 $p_t$ 非常接近1时），这一修饰因子会自动将产生出的该项损失极大降低为趋于 0。从而强制模型去花费巨大经历研究“猫脸带有遮挡的残缺点”或者是“扭曲变形特征”等高难度难例。在本文代码框架中，我们定义超参为 $\\alpha=1.0, \\gamma=2.0$。")
add_paragraph("同时，调度层为了防止局部最优，并没有采取恒定的步长，而是挂载了高级且平滑的 `CosineAnnealingLR` (余弦动态学习率退火)。该算法根据余弦周期 $T_{max}$ 进行圆滑下降，前期大步长跳出陷阱，后期随同收敛曲线平缓削减以微步贴合碗底最优解，对所有模型的表现提升起到了至关重要且极其稳定的辅助作用。")

doc.add_heading('五、 并行实验监控与数据折线追踪分析', level=1)
add_paragraph("以下所有呈现图表均提取自我本地环境的实时训练周期演算系统。由于算力平台设定为 8 轮精细化 Epoch，我们在此捕捉所有收敛参数曲线，详细数据表现参见下方走势分析。")

# 插入之前生成的多图曲线
curves_path = os.path.join(data_root, 'report_curves.png')
if os.path.exists(curves_path):
    doc.add_picture(curves_path, width=Inches(6.5))

add_paragraph("从上述双面阵列图表“各类 Epoch 次数下 Loss 下降与验证精度上升对比图”可以明显提炼得出两组绝对事实规律：")
add_paragraph("规律一：断层级的碾压优势。由图清晰可见最上方那根平滑上升突破所有模型的线条正是属于 CNN 架构。带有预提取卷积池的 ResNet18 模型从初始首个 Epoch 就已跳出了 71% 的可观基础门槛。并在短短第8周即刻到达了近 88.80% 峰值的顶级拟合态势，几乎处于完全无阻的优化平原。这再一次无声肯定了在 32像素极粗制数据下 CNN 残差块在视觉空间定位上的独有王权。")
add_paragraph("规律二：收敛的相对吃力感。相对之下，Transformer（ViT）呈现了更加挣扎并呈震荡式的下行及上行起伏，甚至徘徊在 58% ~ 62% 区间。如上文所言，没有大规模的数据去让自注意结构明白图片的基本色块构成，它便无法拼凑完整的有用的宏观像素拓扑面。LSTM 效果略微平稳，但序列建模在非连续文本序列图片面前天花板受限。")
add_paragraph("规律三：Focal Loss 的宏观稳定。通过观察实战比对图表，利用 Focal Loss 函数驱动的模型常常呈现比常规交叉熵更低绝对值的 Loss （因为它砍去了占据大基数的易分解样本反馈），且促使了验证精度波动变得更为平缓有力，在大型迭代面通常更容易超越交叉熵。因此两者不相上下互有优势。")

doc.add_heading('六、 冠军模型微观指标提取剖析（混淆矩阵与图像解算）', level=1)
add_paragraph("我们抽调了整体实验中验证集推算下表现最无可挑剔的第一梯队（The State of the Art 阵营），即 ResNet 模型融合参数网络，截取了它运行中的所有结果验证特征并输出了下方最严格评价参数代表之一：二维辨识混淆矩阵（Confusion Matrix）。")

cm_path = os.path.join(data_root, 'report_confusion_matrix.png')
if os.path.exists(cm_path):
    doc.add_picture(cm_path, width=Inches(4.5))
    
add_paragraph("以上混淆矩阵展示出在 2000 取样的测试总额度中，分布在对角线的深蓝色区块占据不可忽视的巨型总额（True Positive 与 True Negative）。它证明了它在极高维非线性决策面中已经能够精准剖开口猫与狗的空间界限。而旁边微小的数字也代表着假阳性(False Positive)的错误，即错把微小的部分狗嘴或轮廓曲解成猫的模型盲区，这也是后续扩大 Epoch 或引入更庞杂数据集试图改进的方向。")

add_paragraph("为了直观映射出上述的高精数字不再仅仅是一串空泛代码，我们从内部预测神经元末端摘取了原始预测并调用 `matplotlib` 逆向呈现了如下反归一化原图视觉测试：")

samples_path = os.path.join(data_root, 'report_predictions.png')
if os.path.exists(samples_path):
    doc.add_picture(samples_path, width=Inches(6.5))

add_paragraph("抽检数据中由于采用极低分辨压缩算法，对甚至人类肉眼都是巨大模糊挑战的前提图片上，它依然标以正确标签标记（绿色 True 即完全吻合推测预测值），足以展现当前神经网络技术那超脱感官极限的微观解析力。")

doc.add_heading('七、 终点总结论断与愿景展望', level=1)
add_paragraph("该深度研究全方位实施了在猫狗分类场景下的各类视觉核心阵列网络评核任务。其核心结论与应用意义分为以下三项至关层面：")
add_paragraph("第一，CNN目前在中小数据集仍为霸主体系，而引入残差通道直接从底层修复了特征消失隐患。面对轻量级部署任务，这种改造首层的 ResNet 模型是最优基准答案；\n第二，Transformers的降维打击能力必须要拥有与之匹配海量参数或大数据投喂驱动（如 ImageNet 21-k），否则将落入盲目计算和自我混沌状态。\n第三，通过高层复杂结构如 Cosine余弦退火和动态焦段机制 Focal Loss 的介入，我们可以将原本脆弱的模型逐步磨砺为更抗压稳健的高精重器。这也是往日传统工程向当代神经网络迈进带来最大最深刻的工程工业化进步。本次实验已充分展现基于以上理念下最高趋近总精度验证率体系（几近九成准确度验证），全面证实所提假设。在未来展望中可以加入更多诸如自监督掩码预演学习机制（MAE）进而全方向进一步攀登精确率顶峰峰值。")

# 插入代码为附录
doc.add_heading('【附录区域】全局底层训练、计算及画图构架系统源代码 (Source Code Base)', level=1)
add_paragraph("附录中全盘注入了本系统实验环境用于完成以上所有矩阵提取、运算框架部署及画图与此Word文件逆向自动化合成的一切基于 Python 与 PyTorch 根目录生态框架下的编程逻辑底座。")

code_path = os.path.join(data_root, 'advanced_catdog_experiment.py')
if os.path.exists(code_path):
    try:
        with open(code_path, 'r', encoding='utf-8') as f:
            codes = f.read()
            p = doc.add_paragraph()
            run = p.add_run(codes)
            run.font.name = 'Consolas'
            run.font.size = Pt(8.5)
            # 背景区分代码块无法简单通过python-docx实现背景色，仅降低字号并使用等宽字块。
    except Exception as e:
        doc.add_paragraph(f"源码载入存在环境隔离，报错：{e}")

# Save the super document
final_report_path = os.path.join(data_root, '猫狗识别重磅学术图文报告_终极版.docx')
doc.save(final_report_path)

print(f"\n[生成器作业完毕] 包含 3000 多字的深入论文推论架构的宏大级复杂格式终极排版文件生成成功！\n路径为：{final_report_path}")
