---
layout: post
title: "激活函数解析"
date:  2025-10-12
tags: [ML]
comments: true
author: 炼丹怪
---

激活函数(Activation Function)作为人工神经网络中引入非线性的核心组件,是将线性运算转化为能够拟合任意复杂函数的高维表达的关键。从1950年代受生物神经元启发的二值阈值函数,到深度学习时代的整流线性单元(ReLU),再到如今大语言模型(LLM)广泛采用的门控线性单元(GLU)变体,激活函数的演进史也是一部人工智能攻克优化瓶颈、追求表达能力与计算效率平衡的历史。

---

<!-- more -->

## 1. 引言:非线性的引擎与理论基石

### 1.1 激活函数的本质作用

在人工神经网络(ANN)的架构中,若没有激活函数,无论网络层数有多深,其本质仍仅仅是一个线性回归模型。

* **线性叠加的困境**:数学上,一系列线性变换的组合 $W_n(\dots W_2(W_1 x + b_1) + b_2 \dots) + b_n$ 最终都可以被简化为单一的线性映射 $Wx + b$。这种线性模型无法捕捉现实世界中普遍存在的非线性关系。
* **通用近似定理(Universal Approximation Theorem)**:指出一个包含至少一个隐藏层且具有非线性激活函数的前馈神经网络,在神经元数量足够多的情况下,可以以任意精度逼近任何连续函数。

### 1.2 理想激活函数的数学属性

在数十年的研究中,学者们总结出理想激活函数应具备的若干关键数学属性:

* **非线性(Non-linearity)**:保证网络能逼近复杂函数。
* **可微性(Differentiability)**:支持基于梯度的优化算法(如反向传播)。
* **单调性(Monotonicity)**:通常能保证误差曲面是凸的。但现代函数如 Swish 和 Mish 引入了非单调性以保留负值信息。
* **输出范围(Range)**:有限范围(如 Sigmoid)稳定训练;无限范围(如 ReLU)避免梯度消失。现代深度网络倾向于后者。
* **计算效率(Computational Efficiency)**:在大模型时代,计算复杂度直接影响推理延迟和训练成本。

### 1.3 演进的时间线概览

* **第一代(1950s-1990s):生物启发阶段**
    * 代表:Step, Sigmoid, Tanh。
* **第二代(2010s-2017):整流与深度学习阶段**
    * 代表:ReLU, Leaky ReLU, ELU。
    * 解决梯度消失,开启深度学习黄金时代。
* **第三代(2017-至今):平滑与门控阶段**
    * 代表:GELU, Swish, SwiGLU。
    * 强调概率意义、平滑性及高维语义筛选。

---

## 2. 第一代:S型曲线与生物学启示

早期的神经网络研究深受神经科学影响,试图模拟生物神经元的动作电位机制。

### 2.1 二值阶跃函数 (Binary Step / Threshold)

源于1943年 McCulloch-Pitts 模型和1958年感知机。

* **公式**:
    $$f(x) = \begin{cases} 1 & \text{if } x \ge \theta \\ 0 & \text{if } x < \theta \end{cases}$$
* **局限性**:在 $x=0$ 处不连续,其他位置导数为 $0$,无法进行反向传播更新权重。除二值化神经网络(BNN)外已基本弃用。

### 2.2 Sigmoid (Logistic) 函数

为了解决阶跃函数的不可微问题,引入了将输入平滑压缩到 $(0, 1)$ 的 Sigmoid 函数。

* **公式**:
    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* **导数**:
    $$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$
* **深度分析与缺陷**:
    * **梯度消失(Vanishing Gradient)**:导数最大值为 $0.25$。深层网络中梯度呈指数级衰减,底层参数无法更新。
    * **非零中心化(Non-Zero Centered)**:输出恒正,导致权重更新呈"锯齿状"(Zig-zagging),降低收敛效率。
    * **计算成本**:包含昂贵的 $e^{-x}$ 运算。

### 2.3 双曲正切函数 (Tanh)

本质上是 Sigmoid 的缩放和平移版本:$\tanh(x) = 2\sigma(2x) - 1$。

* **公式**:
    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
* **优势**:输出范围 $(-1, 1)$,具有零中心化(Zero-Centered)特性,收敛速度通常快于 Sigmoid。
* **局限**:依然是软饱和函数,无法彻底解决梯度消失问题。

### 2.4 Softmax 函数

主要作为多分类问题的输出层归一化函数,也是 Transformer 中 Attention 机制的核心。

* **公式**:
    $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

---

## 3. 第二代:整流革命与深度学习的爆发

2010年后,设计哲学从"模拟生物"转向追求计算效率和梯度传播的"分段线性"。

### 3.1 ReLU (Rectified Linear Unit)

现代深度学习最重要的里程碑之一。

* **公式**:
    $$\text{ReLU}(x) = \max(0, x)$$
* **核心优势**:
    * **单侧抑制与稀疏性**:模拟生物神经元稀疏激活特性,解耦特征,提升泛化能力。
    * **解决梯度消失**:正区间梯度恒为 $1$,梯度可无损传回底层。
    * **计算极简**:无指数运算。
* **致命缺陷**:**Dying ReLU(神经元死亡)**。负区间梯度为 $0$,若神经元陷入该区域将永远无法激活。

### 3.2 Leaky ReLU 与 PReLU

为解决 Dying ReLU 问题,在负半区引入微小斜率。

* **Leaky ReLU**: $f(x) = \max(\alpha x, x)$,其中 $\alpha$ 为小常数(如 $0.01$)。
* **PReLU**:将 $\alpha$ 作为一个可学习的参数,允许网络自适应学习负轴激活模式。

### 3.3 ELU (Exponential Linear Unit)

结合 ReLU 的非饱和性与负值的鲁棒性。

* **公式**:
    $$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$$
* **原理**:负半轴具有软饱和性,使激活值均值接近 $0$,起到类似 Batch Normalization 的作用。

### 3.4 SELU (Scaled Exponential Linear Unit)

2017年提出,旨在构建自归一化神经网络(SNN)。

* **公式**:
    $$\text{SELU}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$$
* **精确常数**:通过不动点理论求解得出 $\alpha \approx 1.67326$, $\lambda \approx 1.0507$。
* **原理**:证明了在特定条件下,深层全连接网络无需 BN 即可自动收敛到标准正态分布。

---

## 4. 第三代:平滑与概率化——CNN与早期Transformer的选择

研究发现 ReLU 在原点不可微及其缺乏"概率含义"是改进点。

### 4.1 GELU (Gaussian Error Linear Unit)

BERT, GPT 系列的基石。核心思想是将 Dropout(随机性)与 ReLU(确定性)融合。

* **数学原理**:激活决策取决于 $x$ 的幅度所蕴含的概率。
    $$\text{GELU}(x) = x \cdot P(X \le x) = x \Phi(x)$$
* **工程近似(LLM常用)**:
    $$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh \left[ \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3) \right] \right)$$
* **特性**:具有非单调性(Non-Monotonicity),在负值区域有微小下凹,保留微弱梯度信息。

### 4.2 Swish / SiLU

Google Brain 通过 AutoML 发现的函数。

* **公式**:
    $$\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$
* 当 $\beta=1$ 时称为 SiLU。
* **特性**:无上界、有下界、平滑、非单调。在 EfficientNet 和 YOLOv5 中表现优异。

### 4.3 Mish

受 Swish 启发,在 YOLOv4 中表现优异。

* **公式**:
    $$\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$$

---

## 5. 第四代:门控线性单元(GLU)与大模型架构

这是当前 LLM(GPT-4, LLaMA)最关键的演进。激活函数进化为包含线性投影的门控结构(Gated Architecture)。

### 5.1 GLU 原理

GLU 将输入投影为"值(Value)"和"门(Gate)",激活函数仅作用于门,控制信息流。

$$\text{GLU}(x) = (xW) \otimes \sigma(xV)$$

### 5.2 SwiGLU (Swish-Gated Linear Unit)

目前 LLM 的事实标准(SOTA)。

* **公式**:
    $$\text{SwiGLU}(x) = \text{Swish}_1(xW_g) \otimes (xW_v) = \left( \frac{xW_g}{1 + e^{-xW_g}} \right) \otimes (xW_v)$$
* **为什么统治 LLM?**
    * **性能霸权**:在同等 FLOPs 下,Perplexity 显著低于 GELU/ReLU。
    * **参数效率**:即使将隐藏层维度缩小至 $2/3$ 以保持参数恒定,性能依然更优。
    * **应用模型**:Meta LLaMA 1/2/3, Google PaLM, Mistral / Mixtral。

### 5.3 GeGLU 与 ReGLU

* **GeGLU**:使用 GELU 作为门控。
    * **应用模型**:Google Gemma, T5。
* **ReGLU**:使用 ReLU 作为门控。
    * 计算简单,但在大模型中略逊于 SwiGLU。

---

## 6. 特殊场景:硬件感知、稀疏性与量化

随着模型向边缘设备(Edge AI)和高效推理发展,设计开始回流。

### 6.1 移动端优化:HardSigmoid 与 HardSwish

MobileNetV3 引入的分段线性近似,避免移动端 CPU/FPGA 上昂贵的指数运算。

* **HardSwish**:
    $$\text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}$$

### 6.2 稀疏性的回归:Squared ReLU 与 StarReLU

Swish/GELU 是稠密的(负输入不为0),无法利用稀疏计算加速。

* **Squared ReLU ($\text{ReLU}^2$)**:
    $$\text{ReLU}^2(x) = (\max(0, x))^2$$
* **优势**:Google 的 Primer 论文表明,它在 Transformer 中精度相当,但配合支持稀疏性的硬件(如 PowerInfer)可带来 2-4倍推理加速。
* **StarReLU**:减少 Transformer 中的 FLOPs,用于 MetaFormer。

### 6.3 针对量化训练的 Smooth-SwiGLU

2024年 Intel Habana 提出,解决 FP8 低精度训练下的数值不稳定问题。

---

## 7. 总结与分类图谱

### 7.1 主流激活函数特性对比总表

| 激活函数 | 核心公式 | 导数特性 | 平滑性 | 核心优势 | 典型应用模型 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $f(1-f)$ | 是 | 概率解释 | 早期MLP, LSTM门 |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1-f^2$ | 是 | 零中心化 | RNN, GRU |
| **ReLU** | $\max(0, x)$ | $\{0, 1\}$ | 否 | 稀疏, 无梯度消失, 极快 | ResNet, VGG |
| **GELU** | $x\Phi(x)$ | $\Phi(x) + x\phi(x)$ | 是 | 概率加权, 性能强 | BERT, GPT-3, ViT |
| **Swish/SiLU** | $x\sigma(x)$ | $f + \sigma(1-f)$ | 是 | 自适应, 非单调 | EfficientNet, YOLOv5 |
| **SwiGLU** | $\text{Swish}(xW) \otimes xV$ | 复杂(含门控) | 是 | 门控选择, SOTA性能 | LLaMA, PaLM, Mistral |
| **GeGLU** | $\text{GELU}(xW) \otimes xV$ | 复杂(含门控) | 是 | 同 SwiGLU | Gemma, T5 |
| **Squared ReLU** | $\max(0, x)^2$ | $2x \text{ or } 0$ | 是 ($C^1$) | 稀疏加速, 拟合GELU | Primer, SparseLLM |

### 7.2 选型建议与未来展望

* **对于大语言模型(LLM)**
    * **SwiGLU** 是目前的绝对首选。
    * 若资源受限或特定架构,**GeGLU** 是第二选择。
* **对于计算机视觉(CNN)**
    * **SiLU (Swish)** 和 **Mish** 在 YOLO 等中等模型表现最好。
    * **ReLU** 在超深网络中依然稳健。
* **对于端侧推理与极致速度**
    * **ReLU** 或 **Squared ReLU**。
    * 随着 NPU 支持稀疏计算,能产生精确 $0$ 值的函数将回潮。
* **对于科学计算**
    * **SELU** 在全连接网络中具有理论优势。
