---
layout: post
title: "优化器解析：从 SGD 到 Muon 的演进之路"
date: 2025-10-12
tags: [ML, Optimization, Deep Learning]
comments: true
author: 炼丹怪
---

深度学习的崛起，本质上是算力、数据与优化算法三者共振的结果。本文将穷尽式地梳理从早期的随机梯度下降（SGD）到如今专为大语言模型（LLM）设计的 Muon 算法的演进过程，揭示优化器从“标量微积分”向“高维几何控制”的认知革命。

---

<!-- more -->

## 1. 引言：算法进化的历史脉络

在这一宏大的技术叙事中，**优化算法（Optimization Algorithms）**扮演着指挥棒的角色，决定了神经网络能否从随机初始化的混沌状态收敛至具有泛化能力的有序结构。从早期的随机梯度下降（SGD）到如今专为大语言模型（LLM）设计的 Muon 算法，优化器的演进并非简单的修补，而是一场从“标量微积分”向“高维几何控制”的认知革命。

随后，我们将沿着时间轴与问题域的双重线索，深入剖析优化算法的三个代际跨越：

* **动量法时代**：从处理简单非凸优化到引入物理动量。
* **自适应学习率时代**：解决稀疏性与病态曲率问题（Adam/AdamW）。
* **矩阵结构优化时代**：以结构感知（Structure-Aware）为核心（Lion, Sophia, Muon）。

特别地，本报告将重点解构 **Muon (MomentUm Orthogonalized by Newton-Schulz)** 等前沿算法。这类算法标志着优化范式的根本性转移——从将参数视为孤立数值的“元素级”优化，转向尊重参数矩阵谱结构（Spectral Structure）的“张量级”优化。通过对更新公式的详细推导与原理阐释，我们试图回答一个核心问题：在万亿参数模型时代，什么样的优化动力学才是最高效的？

---

## 2. 计算的基石：反向传播与自动微分的数学重构

反向传播算法（Backpropagation）是训练神经网络的引擎。尽管其核心思想常被简化为“链式法则的应用”，但在现代深度学习框架中，其实现早已超越了标量链式法则的范畴，演变为基于计算图（Computational Graph）的张量流操作。

### 2.1 自动微分（AD）的两种模式与计算图


在讨论反向传播之前，必须明确其在数值计算领域的定位。计算梯度的计算机方法主要有三种：数值微分（Finite Difference）、符号微分（Symbolic Differentiation）和自动微分（Automatic Differentiation, AD）。深度学习完全依赖于自动微分。

自动微分的核心在于将复杂的复合函数分解为一系列基本运算操作（加法、乘法、激活函数等），构建计算图。根据导数传播的方向，AD 分为前向模式（Forward-Mode）和反向模式（Reverse-Mode）。

#### 2.1.1 前向模式 vs 反向模式的复杂度分析

假设我们有一个函数 $f: R^n \to R^m$，输入向量 $x \in R^n$，输出向量 $y \in R^m$。我们需要计算雅可比矩阵（Jacobian Matrix） $J \in R^{m \times n}$。

* **前向模式**：从输入向输出传播扰动。每次计算只能获得雅可比矩阵的一列（即输出相对于某一个输入分量的导数）。若要计算完整的 $J$，需要进行 $n$ 次前向传播。计算复杂度约为 $O(n)$。这对于输入维度极高（如图像像素、词向量）的神经网络来说是不可接受的。
* **反向模式**：从输出向输入传播误差（伴随变量）。每次反向传播可以获得雅可比矩阵的一行（即某一个输出分量相对于所有输入的导数）。对于标量损失函数 $L$（即 $m=1$），反向模式只需一次遍历即可获得 $L$ 关于所有 $n$ 个参数的梯度。计算复杂度约为 $O(m)$，在 $m=1$ 时即为 $O(1)$。

反向传播算法正是反向模式自动微分在标量损失函数下的特例。由于神经网络通常具有百万级甚至亿级的参数输入（$n$ 极大），但最终只有一个标量损失（$m=1$），反向模式是唯一可行的计算路径。

### 2.2 矩阵微积分视角下的反向传播推导

为了深入理解现代优化器（如 Muon）为何关注矩阵结构，我们不能仅停留在标量导数上，必须在矩阵微积分（Matrix Calculus）的框架下推导梯度。

#### 2.2.1 符号定义与前向传播

考虑神经网络中的第 $l$ 层，其参数为权重矩阵 $W_l$ 和偏置向量 $b_l$。

1.输入激活值：
$$a_{l-1} \in R^{d_{in}}$$

2.线性变换输出：
$$z_l = W_l a_{l-1} + b_l$$

其中
$$W_l \in R^{d_{out} \times d_{in}}$$

3.非线性激活：
$$a_l = \sigma(z_l)$$

损失函数为 $L$。我们的目标是计算
$$\frac{\partial L}{\partial W_l}$$
和
$$\frac{\partial L}{\partial b_l}$$

#### 2.2.2 误差项（Error Term）的定义

引入误差项 $\delta_l$，定义为损失函数相对于该层线性输出 $z_l$ 的梯度：

$$\delta_l \triangleq \frac{\partial L}{\partial z_l} \in R^{d_{out}}$$

（注意：此处采用分母布局，梯度形状与原变量一致）

#### 2.2.3 链式法则的矩阵形式推导

根据链式法则，损失相对于权重矩阵 $W_l$ 的梯度可以分解为：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial z_l} \cdot \frac{\partial z_l}{\partial W_l}$$

这里涉及张量对矩阵的导数，直接处理较为困难。我们采用迹（Trace）技巧或逐元素推导来通过维度匹配验证结果。

考虑 $z_l$ 的第 $i$ 个分量 $z_i$：

$$z_i = \sum_{j} W_{ij} a_{j} + b_i$$

（为简化，省略层级下标 $l$ 和 $l-1$）

计算偏导数：

$$\frac{\partial L}{\partial W_{ij}} = \sum_{k} \frac{\partial L}{\partial z_k} \frac{\partial z_k}{\partial W_{ij}}$$

由于 $z_k$ 只在 $k=i$ 时包含 $W_{ij}$，求和项中仅一项非零：

$$\frac{\partial L}{\partial W_{ij}} = \delta_i \cdot a_j$$

将上述标量结果重新组合成矩阵形式，可以发现 $\frac{\partial L}{\partial W}$ 的 $(i,j)$ 元素是 $\delta_i$ 与 $a_j$ 的乘积。这对应于向量的外积（Outer Product）：

$$\frac{\partial L}{\partial W_l} = \delta_l a_{l-1}^\top$$

这一公式是所有神经网络训练的基础。它告诉我们，权重的更新方向是由当前层的误差信号 $\delta_l$ 和上一层的输入信号 $a_{l-1}$ 的相关性决定的。这也是为何输入数据的归一化（Normalization）如此重要——它直接影响了梯度矩阵的数值稳定性。

#### 2.2.4 误差项的递推传播

为了计算 $\delta_l$，我们需要从后一层 $\delta_{l+1}$ 递推回来。

$$\delta_l = \frac{\partial L}{\partial z_l} = \left( \frac{\partial z_{l+1}}{\partial z_l} \right)^\top \frac{\partial L}{\partial z_{l+1}}$$

分解中间过程：

$$z_{l+1} = W_{l+1} \sigma(z_l) + b_{l+1}$$

应用全微分或雅可比矩阵乘法：

$$\frac{\partial z_{l+1}}{\partial z_l} = W_{l+1} \cdot \text{diag}(\sigma'(z_l))$$

因此，误差传播公式为：

$$\delta_l = \left( W_{l+1} \text{diag}(\sigma'(z_l)) \right)^\top \delta_{l+1}$$

$$\delta_l = \text{diag}(\sigma'(z_l)) W_{l+1}^\top \delta_{l+1}$$

在逐元素运算符号 $\odot$ 下，这通常写作：

$$\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot \sigma'(z_l)$$

### 2.3 批量（Batch）计算的矩阵化

在实际训练中，我们处理的是小批量（Mini-batch）数据 $X \in R^{B \times d_{in}}$。此时，梯度计算变为矩阵乘法，极大地利用了 GPU 的并行能力。

$$\frac{\partial L}{\partial W_l} = \frac{1}{B} \Delta_l^\top A_{l-1}$$

其中 $\Delta_l \in R^{B \times d_{out}}$ 是批量误差矩阵，$A_{l-1} \in R^{B \times d_{in}}$ 是批量输入矩阵。注意这里的转置位置，取决于数据是行优先还是列优先存储。

这一数学结构揭示了一个关键约束：梯度矩阵的秩（Rank）受限于批量大小 $B$。当 $B < \min(d_{in}, d_{out})$ 时，单步更新的梯度矩阵是低秩的。这对优化器的设计有深远影响，也是后续 Muon 等算法试图通过正交化来“修复”矩阵结构的原因之一。

---

## 3. 随机梯度下降（SGD）与动量机制：从山谷到平原

拥有了梯度，优化的征程才刚刚开始。损失函数的几何景观（Geometry of Loss Landscape）决定了我们该如何利用梯度。



### 3.1 随机梯度下降（SGD）的动力学特征

最朴素的更新规则是：

$$w_{t+1} = w_t - \eta g_t$$

其中 $g_t$ 是小批量梯度的无偏估计。

* **SGD 的优势**：
    * **逃离鞍点**：高维非凸优化中，鞍点（Saddle Points）比局部极小值更常见。全量梯度下降在鞍点处梯度为零，容易停滞。而 SGD 引入的噪声提供了在非下降方向上的随机扰动，有助于逃离鞍点。
    * **泛化能力**：SGD 倾向于收敛到“平坦”的极小值（Flat Minima），这些区域对参数扰动不敏感，通常意味着更好的泛化性能。
* **SGD 的缺陷**：
    * **病态曲率（Ill-conditioning）**：在损失函数的等高线呈现狭长椭圆状（即 Hessian 矩阵的最大特征值与最小特征值之比很大）时，SGD 会在峡谷壁之间剧烈震荡，收敛速度极慢。
    * **局部震荡**：由于梯度的随机性，SGD 在接近最优点时难以稳定，需要精细的“学习率退火”策略。

### 3.2 动量（Momentum）：物理视角的引入

为了解决震荡问题，Polyak 于 1964 年引入了动量法。其物理直觉源自有阻尼的谐振子。即使当前梯度为零（如谷底），积累的动量也能推动参数继续移动。

更新公式：

$$v_{t+1} = \mu v_t + (1-\tau) g_t$$

$$w_{t+1} = w_t - \eta v_{t+1}$$

（注意：$\tau$ 是阻尼系数，有时取 0，有时取 1，不同框架实现不同）

动量法本质上是一个低通滤波器，它平滑了梯度的短期波动，放大了长期一致的梯度方向。在狭长峡谷中，垂直于谷底方向的震荡正负抵消，而沿谷底方向的梯度累积增强，从而加速收敛。

### 3.3 Nesterov 加速梯度（NAG）：预判未来

Nesterov 动量（NAG）是对标准动量的微小但关键的修正。标准动量计算的是当前位置的梯度，而 NAG 计算的是“根据动量迈出一步后”位置的梯度。

$$g_{lookahead} = \nabla L(w_t + \mu v_t)$$

$$v_{t+1} = \mu v_t - \eta g_{lookahead}$$

$$w_{t+1} = w_t + v_{t+1}$$

这种“前瞻性”使得算法在遇到梯度反向（如冲过谷底）时能提前减速，理论收敛速率从 $O(1/T)$ 提升至 $O(1/T^2)$（在凸优化假设下）。

---

## 4. 自适应学习率时代：从 Adagrad 到 AdamW 的统治

随着深度学习进入 NLP 和推荐系统领域，特征的稀疏性（Sparsity）成为了新挑战。不同参数的更新频率差异巨大，全局统一的学习率 $\eta$ 不再适用。这催生了自适应学习率算法家族。

### 4.1 Adagrad：几何适应

Adagrad（Adaptive Gradient）的核心思想是：对于频繁更新的参数，降低其学习率；对于稀疏更新的参数，提高其学习率。

更新公式：

$$G_t = G_{t-1} + g_t \odot g_t$$

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

$G_t$ 是历史梯度的平方和。在文本数据中，低频词对应的参数 $g_t$ 大多为 0，$G_t$ 增长缓慢，因此保持了较大的有效学习率。

* **局限**：$G_t$ 单调递增，导致分母无限变大，学习率在训练后期会过早衰减至 0，导致欠拟合。

### 4.2 RMSprop：处理非平稳目标

为了解决 Adagrad 学习率消失的问题，Geoffrey Hinton 引入了指数移动平均（EMA）来替代平方和。这使得算法仅关注“最近”的梯度规模，从而可以适应非平稳（Non-stationary）的目标函数。

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$$

RMSprop 成为了 RNN 训练的标准配置，因为它能有效处理时间步长上的梯度爆炸/消失问题。

### 4.3 Adam：自适应力矩估计

Adam（Adaptive Moment Estimation）结合了 Momentum 的一阶动量和 RMSprop 的二阶动量，是目前最流行的优化器。

完整算法流程：

1.  计算梯度 $g_t$
2.  更新一阶矩（均值）：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
3.  更新二阶矩（方差）：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4.  偏差修正（Bias Correction）：由于 $m, v$ 初始化为 0，早期估计偏向 0。
    $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
    $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
5.  参数更新：
    $$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam 的物理意义在于：它根据梯度的**信噪比（Signal-to-Noise Ratio）**调整步长。$\hat{m}_t$ 代表信号，$\sqrt{\hat{v}_t}$ 代表噪声（波动）。信噪比高的方向步长更大，反之则小。

### 4.4 AdamW：解耦权重衰减的革命

在很长一段时间里，人们认为 L2 正则化（在损失函数中加 $\frac{\lambda}{2} ||w||^2$）和权重衰减（在更新公式中减去 $\eta \lambda w$）是等价的。这对于 SGD 确实成立。然而，Loshchilov 和 Hutter 在 2017 年指出，对于自适应算法（如 Adam），两者并不等价。

如果将 L2 正则项加入损失函数，其梯度 $\lambda w$ 会被混入 $g_t$ 中，进而被归一化项 $\sqrt{\hat{v}_t}$ 缩放：

$$\text{Adam w/ L2}: \quad w_{t+1} \approx w_t - \eta \left( \frac{g_{task}}{\sqrt{\hat{v}_t}} + \frac{\lambda w_t}{\sqrt{\hat{v}_t}} \right)$$

这导致权重衰减的力度取决于梯度的方差，通常使得正则化力度过小（因为 $w$ 通常比梯度大）。

AdamW（Decoupled Weight Decay）修正了这一点，将权重衰减独立于梯度更新之外：

$$\text{AdamW}: \quad w_{t+1} = w_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right) - \eta \lambda w_t$$

这一修正对于 Transformer 类大模型的训练稳定性至关重要，因为这些模型对正则化非常敏感。AdamW 随后成为 BERT、GPT 等模型的默认优化器。

---

## 5. 大规模与分布式训练：LARS, LAMB 与 Adafactor

随着模型规模从百万级跃升至十亿级（Billion），训练时长成为瓶颈。解决方案是增大批量大小（Batch Size）以利用大规模并行计算（如 TPU Pods）。但这带来了“泛化鸿沟”（Generalization Gap）和收敛不稳定的问题。

### 5.1 LARS：层级自适应速率

在训练 ResNet-50 时，研究者发现不同层的梯度范数与权重范数的比率差异巨大。首层卷积层可能变化剧烈，而深层变化缓慢。统一的学习率会导致某些层发散或停滞。

**LARS（Layer-wise Adaptive Rate Scaling）**引入了信任比率（Trust Ratio）：

$$\gamma_l = \frac{||w_l||}{||\nabla L_l|| + \beta ||w_l||}$$

每一层的学习率被动态调整为 $\eta_{global} \times \gamma_l$。这确保了每一步更新的幅度相对于权重的模长是受控的，允许在超大 Batch 下使用极大的学习率而不发散。

### 5.2 LAMB：大批量 BERT 的加速器

**LAMB（Layer-wise Adaptive Moments for Batch training）**是 LARS 思想在 Adam 上的推广。它结合了 Adam 的动量机制和 LARS 的信任比率裁剪。

更新逻辑：

1.  计算 Adam 风格的更新步 $r_t$。
2.  计算层级信任比率 $\phi_l = ||w_l|| / ||r_t||$。
3.  实际更新：$w_{t+1} = w_t - \eta \cdot \phi_l \cdot r_t$。

LAMB 使得 BERT 的预训练批次大小可以达到 64k 甚至更大，将训练时间从 3 天缩短至 76 分钟。其核心贡献在于证明了：在大规模并行下，层级自适应（Layer-wise Adaptation）比逐元素自适应（Element-wise Adaptation）更能维持训练稳定性。

### 5.3 Adafactor：显存的极致压缩

在训练巨型模型时，Adam 优化器需要存储 $m$ 和 $v$ 两个状态矩阵，这意味着优化器显存占用是模型参数的 2 倍（如果是 FP32 状态甚至是 3 倍）。

Adafactor 通过矩阵分解技术解决了这一问题。它不再存储完整的 $n \times m$ 二阶矩矩阵 $V$，而是将其近似为行向量 $R \in R^n$ 和列向量 $C \in R^m$ 的外积：

$$V_{ij} \approx R_i C_j$$

显存占用从 $O(nm)$ 降低到 $O(n+m)$。虽然这损失了一定的精度，但对于 T5、PaLM 等模型，Adafactor 是使得训练在有限显存下可行的关键技术。

---

## 6. 后 AdamW 时代的新星：Lion 与 Sophia

进入 2023 年，优化器领域迎来了新的爆发。研究者试图打破 AdamW 的垄断，寻找更高效（计算/显存）或收敛更快的算法。

### 6.1 Lion：进化的符号优化器

**Lion（Evolved Sign Momentum）**是由 Google Brain 通过符号化程序搜索（Symbolic Program Search）发现的算法。

更新公式：

$$c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$w_{t+1} = w_t - \eta \cdot \text{sign}(c_t) - \eta \lambda w_t$$

$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$$

核心特性：

* **符号更新（Sign Update）**：Lion 只使用动量的符号（$\pm 1$），这意味着所有参数的更新幅度是恒定的（仅受 $\eta$ 控制）。这对应于 $L_\infty$ 范数下的最速下降，具有很强的正则化效果。
* **显存优势**：不需要存储二阶矩 $v$，比 AdamW 节省 33%-50% 显存。
* **吞吐量**：由于没有平方和开方运算，计算速度略快。

实验表明，Lion 在 ViT 和扩散模型上表现优异，但在 LLM 上需要更精细的学习率调优，且有时不如 AdamW 稳定。

### 6.2 Sophia：二阶信息的回归

**Sophia（Second-order Clipped Optimization）**试图以低成本引入 Hessian 曲率信息。

Adam 用梯度平方近似 Hessian 对角线，这在理论上是粗糙的。Sophia 使用一种轻量级的对角 Hessian 估计器 $\hat{h}$，并结合**元素级裁剪（Clipping）**来防止更新爆炸：

$$w_{t+1} = w_t - \eta \cdot \text{clip}\left( \frac{m_t}{\max(\hat{h}_t, \epsilon)}, -\rho, \rho \right)$$

Sophia 在 GPT-2 预训练中展现了比 AdamW 快 2 倍的收敛速度（按步数计）。它证明了在 LLM 训练中，利用更准确的曲率信息是加速收敛的可行路径。

---

## 7. 范式转移：Muon 与矩阵结构优化

2024 年，**Muon（MomentUm Orthogonalized by Newton-Schulz）**的提出标志着优化器从“元素级自适应”向“张量级结构优化”的范式转移。这是本报告重点关注的前沿领域。

### 7.1 现有优化器的几何盲区

AdamW、Lion 等算法在处理权重矩阵 $W \in R^{d_{out} \times d_{in}}$ 时，本质上将其视为长度为 $d_{out} \times d_{in}$ 的扁平向量。这种处理方式忽略了矩阵的谱结构（Singular Value Structure）。

在深度线性网络理论中，为了使信号在深层网络中无损传播，权重矩阵应保持等距性（Isometry），即其奇异值应集中在 1 附近。如果某些奇异值过大或过小，信号会在前向/反向传播中发生爆炸或消失。标准的梯度下降并不保证更新后的矩阵保持良好的谱分布。

### 7.2 Muon 核心原理：牛顿-舒尔茨正交化

Muon 专门针对**线性层（Linear Layers）**的二维矩阵参数设计。其核心思想是：将动量矩阵 $M_t$ 强制投影到正交矩阵流形上，作为更新方向。

#### 7.2.1 算法流程

1.  **动量积累（标准 SGD）**：
    $$M_t = \mu M_{t-1} + G_t$$
    （这里 $G_t$ 是当前 Batch 的梯度）

2.  **正交化（Orthogonalization）**：
    我们需要找到一个正交矩阵 $O_t$，使得它在某种意义上最接近 $M_t$。数学上，这等价于取 $M_t$ 的 SVD 分解 $U \Sigma V^\top$，然后令 $O_t = U V^\top$。

    然而，SVD 在 GPU 上计算极其昂贵且难以并行。Muon 使用了**牛顿-舒尔茨迭代（Newton-Schulz Iteration）**来高效逼近这一过程。

    **Newton-Schulz 迭代公式**：
    * 令 $X_0 = \frac{M_t}{||M_t||_{spec}}$（谱范数归一化）。
    * 迭代 $k$ 次（通常 $k=5$）：
        $$X_{k+1} = \frac{1}{2} X_k (3I - X_k^\top X_k)$$
    * 最终 $O_t = X_k$。

    该迭代仅涉及矩阵乘法（MatMul），非常适合利用 GPU/TPU 的 Tensor Cores。

3.  **参数更新**：
    $$W_{t+1} = W_t - \eta \cdot \alpha_t \cdot O_t$$
    其中 $\alpha_t$ 是特定的缩放因子（通常基于 RMS），用于恢复更新的幅度。

#### 7.2.2 为什么正交化有效？

Muon 通过强制更新量正交，实际上是在执行一种谱正则化（Spectral Regularization）。它“漂白”了梯度的相关性，使得更新方向在所有奇异向量方向上是均衡的。

* **信号传播**：正交更新有助于保持权重矩阵的条件数（Condition Number）接近 1，从而允许更深的网络和更大的学习率。
* **Sample Efficiency**：实验表明，Muon 在 NanoGPT 和 CIFAR-10 训练中，达到相同 Loss 所需的步数显著少于 AdamW。

### 7.3 Muon vs Shampoo

Muon 并非首个利用矩阵结构的优化器。Shampoo 优化器早在 2018 年就利用 Kronecker 积来逼近全 Hessian 矩阵的逆。

| 特性 | Shampoo | Muon |
| :--- | :--- | :--- |
| **核心机制** | 预调节器（Preconditioner）基于 Kronecker 积 $L^{-1/4} G R^{-1/4}$ | 更新方向正交化 $\text{Newton-Schulz}(M)$ |
| **计算代价** | 需计算矩阵逆的根（SVD 或 Schulz 迭代），较重 | 仅需矩阵乘法，极轻量 |
| **内存占用** | 需存储 $L$ 和 $R$ 统计量 | 仅需存储动量 $M$ |
| **定位** | 二阶优化器近似 | 一阶动量+结构约束 |

Muon 可以看作是 Shampoo 的一种“瞬时”且“硬约束”的变体，它去掉了复杂的预调节器累积，直接对当前动量进行“整形”。

### 7.4 NorMuon：弥补 Muon 的短板

Muon 的一个潜在问题是它对矩阵的所有元素应用了统一的缩放（虽然方向是正交的）。但在 Transformer 中，不同神经元（矩阵的行）的激活方差可能差异很大。

**NorMuon（Normalized Muon）**提出结合 Adam 的自适应方差与 Muon 的正交化。

1.  计算正交化方向 $O_t$。
2.  计算 $O_t$ 的列方向 RMS 统计量 $v_t$。
3.  用 $v_t$ 对 $O_t$ 进行行级（Row-wise）归一化。

实验显示，NorMuon 在 1.1B 参数模型上比标准 Muon 进一步提升了 11% 的效率，证明了“结构化正交”与“逐元素自适应”并非互斥，而是可以互补的。

---

## 8. 综合对比与未来展望

为了直观展示优化器的演进，我们构建以下对比矩阵：

### 表 8-1：主流优化器特性对比

| 优化器 | 核心机制 | 内存复杂度 | 适用场景 | 关键优势 | 劣势 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SGD+Momentum** | 一阶动量 | $1 \times$ 模型状态 | 计算机视觉 (ResNet) | 泛化性好，内存小 | 需精细调参，收敛慢 |
| **AdamW** | 动量 + 自适应方差 + 解耦衰减 | $2 \times$ 模型状态 | LLM (BERT/GPT), 通用 | 稳健，收敛快，无需太多调参 | 内存占用大 |
| **Adafactor** | 分解二阶矩 | $< 1.1 \times$ 模型状态 | 超大模型 (PaLM, T5) | 极致省显存 | 收敛性略逊 AdamW |
| **LAMB** | 层级信任比率 | $2 \times$ 模型状态 | 大批量分布式训练 | 支持超大 Batch，训练极快 | 计算信任比率有额外开销 |
| **Lion** | 符号动量 | $1 \times$ 模型状态 | 扩散模型，ViT | 内存小，更新幅度一致 | LLM 上表现不稳定 |
| **Sophia** | 对角 Hessian 裁剪 | $2 \times$ 模型状态 | LLM 预训练 | 收敛速度快 (Step-wise) | 需 Hessian 估计开销 |
| **Muon** | 矩阵正交化 (Newton-Schulz) | $1 \times$ 模型状态 | 线性层 (LLM/Conv) | Sample Efficiency 极高，利用 Tensor Cores | 仅适用于 2D 参数，需混合使用 |

### 8.1 结论与洞察

* **从标量到张量**：优化器的设计维度正在升级。Muon 的成功表明，简单的将数以亿计的参数视为一维向量是低效的。未来的优化器将更多地利用参数的张量结构（Tensor Structure）。

* **硬件协同设计（Hardware-Algorithm Co-design）**：Muon 之所以受到关注，很大程度上是因为它使用了矩阵乘法（MatMul），这正是现代 GPU/TPU 最擅长的操作。相比之下，Adam 中的除法和开方运算在硬件上是低效的。未来的算法将更倾向于 "MatMul-heavy" 而非 "Element-wise heavy"。

* **混合优化策略**：随着模型异构性增加，单一优化器统治全场的时代可能终结。正如 Muon 通常只用于线性层，而用 AdamW 优化 Embedding 和 LayerNorm 一样，**分层混合优化（Layer-wise Hybrid Optimization）**将成为常态。

在迈向万亿参数与 AGI 的征途中，反向传播提供了梯度的罗盘，而不断演进的优化器——尤其是像 Muon 这样融合了几何洞察与硬件亲和性的新物种——正在为这艘巨轮安装上曲率驱动引擎。