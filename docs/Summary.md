# 术语和概念

## ML 基础概念

- 回归问题：输出为连续值，如房屋价格、气温、销售额预测。使用的方法如线性回归
- 分类问题：输出离散值，如图像分类、垃圾邮件识别、疾病检测等。使用的方法如 softmax 回归

### 线性回归

可以把上述两种回归视为最简单的单层神经网络。

假如一模型是：
y = x_1w_1 + x_2w_2 + b
那么，输入层由 x_1 x_2 构成（实际上可以组成向量），有两个，这个个数叫特征数，或者特征向量维度，
输出层特征向量维度为 1，
w_1 w_2 的权重和 b 的偏置/偏差（bias）合称参数 parameter
此处，输出层神经元和输入层各输入完全连接，称该输出层为全连接层（fully-connected layer）/稠密层（dense layer）
技巧是尽量矢量/矩阵整体运算，不要单个元素运算（即自己实现线性代数）

model training 模型训练：通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。

训练数据集（training data set）或训练集（training set）中，一栋房屋称一个样本（sample），
真实售价 y 叫标签（label）（标签可以是多维的，对应于实际的多个通道/多个物体），
预测标签的两个因素 x_1 x_2 叫作特征（feature），用于表征样本特点。

loss function 损失函数：衡量误差的函数。所以平方误差函数（预测 - 预期的差的平方的1/2，1/2使损失函数求导后系数为1，形式简单）又叫平方损失（square loss）。通常整个模型预测质量用训练集所有样本误差均值衡量。

模型训练的目标即找出一组参数组合，使损失函数（指所有样本的损失-误差均值）最小

显然，像采用平方损失的线性回归这种简单模型，模型训练目标或损失函数最小可以直接从模型公式求得解析解 analytical solution。
然多数 ML 模型不能或难以求得解析解，因此只能数值解（通过优化算法迭代模型参数使损失函数降低到误差范围内）

各种**求数值解的方法**叫**优化算法**，深度学习中广泛使用 mini-batch stochastic gradient descent 小批量随机梯度下降

mini-batch stochastic gradient descent:
1. 选一组模型参数初值（随机或预训练，迁移？）
2. 参数多次迭代，以降低损失函数：随机均匀采样固定数量 n 的训练数据样本（这就是所谓的小批量 mini-batch，样本个数叫 batch size 批量大小），用 learning rate 学习率 eta（可正可负）乘 mini-batch 的平均损失对模型参数的梯度作为参数本次迭代减小量。(公式见 https://zh.d2l.ai/chapter_deep-learning-basics/linear-regression.html)

此处，batch size 和 learning rate 为人为设定，为区别模型自学习的参数，称超参数 hyperparameter。俗语调参即调节超参数。少数情况下超参数也可通过模型训练习得。

三个词一样：模型预测 = 模型推断 = 模型测试

由此，n 个样本的 d 个特征构成 n x d 输入层特征向量，模型权值构成 d 维列向量，偏置成为一维标量
每次的小批量训练模型的预测输出即成为输入特征向量和权值列向量的向量积加上偏置，是一 n 维列向量，
其与标签组成的n维列向量之差平方的1/2得到的n维列向量即为平方损失向量（函数），该向量元素的均值即该次预测的样本误差，
然后对（当然也可以）
数学语言描述参见 https://zh.d2l.ai/chapter_deep-learning-basics/linear-regression.html

一些情况下，每个样本的损失函数对于参数的梯度之和 不等于 所有样本损失函数之和对于参数的梯度？

重新推导一遍，然后用教程实现一遍

### softmax 回归

不同于线性回归的单一输出，softmax 为多个输出。以 softmax 回归为例介绍分类模型。

通常用离散数值来表示类别，于是也可以有类似线性回归的数据集。因此原则上也可以用之前的回归模型进行建模。
但这种离散使用连续值模型的做法常常影响分类质量，因此还是需要专门模型。

softmax 回归是一个单层全连接神经网络

为什么要使用 softmax 进行压缩-截取-求和（这玩意咋称呼？），而不是直接使用输出层的输出大小作为预测类别 i 的置信度,
因为：
- 输出层输出值范围不确定，难以直接判断（因为输出的绝对值没有意义）
- 训练用标签是离散值（其实就是各种维度上的单位向量），那么既然输出值范围不确定，那输出误差就难以衡量（误差要用于学习）

因此使用 softmax 或者说 softmax operator 压缩变换（咋称呼？）为正且和为1的概率分布（公式定义见 https://zh.d2l.ai/chapter_deep-learning-basics/softmax-regression.html）

然后，从数学上看，显然 softmax 和线性回归是一致的，同样是 n 样本 d 特征构成 n×d 实空间输入特征向量，
假设其输出特征/类别数为 m，则
区别只是 softmax 权值为 d×m 矩阵而非列向量（虽然向量可以看作是矩阵的特殊情况），其叠加的偏置为一 m 维行向量，
但注意此时输入特征向量和权值矩阵的向量积得到结果是一 n×m 矩阵，这个矩阵和偏置行向量可以求和的原因是？
当然最终结果整体还要套上 softmax 构成本次预测的输出。

那么损失函数？
不同于连续模型的预测，分类问题并不需要预测概率完全等同标签概率，只需要其概率分布能够得出正确结论即可。
此时平方损失过于严格（过于严格区分同样正确的判断），从而使用**更适合衡量两个概率分布差异的函数**，交叉熵（cross entropy）：
https://zh.d2l.ai/chapter_deep-learning-basics/softmax-regression.html
数学公式如此，但实际可以根据优化成直接根据标签的位值选取某一预测概率的对数。

最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率

问题：查阅资料，了解最大似然估计。它与最小化交叉熵损失函数有哪些异曲同工之妙？

### 图像分类数据集（Fashion-MNIST）

## 深度学习计算

## CNN (Convolutional Neural Network) - 含有卷积层（Convolutional layer）的神经网络

### 二维卷积层

二维卷积层是最常见的卷积层，即高×宽二维。

虽得名于卷积（convolution）运算，但通常在卷积层中使用更直观的互相关（cross-correlation）运算
前者是分析里的概念，后者是统计里的概念。
实际上两种运算类似。为了得到卷积运算的输出，我们只需将核数组左右翻转并上下翻转，再与输入数组做互相关运算。可见，卷积运算和互相关运算虽然类似，但如果它们使用相同的核数组，对于同一个输入，输出往往并不相同。
一个事实，上下左右翻转（下称双重镜像）相当于旋转180度
卷积层为什么可以使用互相关运算替代卷积运算？因为核数组是学习得到的，不论一开始是否翻转，只要前后的运算形式一致即可。具体而言，因为如果使用卷积表示，那么每次运算都要双重镜像，同时学习出的核和互相关学习出的核也是双重镜像关系，因此二者“抵消”，对于最终的运算结果即模型预测，没有任何区别。
为与大多数深度学习文献一致，如无特别说明，本书中提到卷积运算均指互相关运算。

二维卷积层中，一个二维输入数组（矩阵，高维就是张量等）和一个二维核（kernel）数组（矩阵，高维就是张量等）通过互相关运算输出一个二维数组，后加上一标准偏差得到输出。
核 = 卷积核 = 过滤器（filter）。
卷积核窗口 = 卷积窗口。
核数组/矩阵的维度 = 卷积窗口的形状 = 卷积核的高和宽（pxq的形状表示 p 高 q 宽，对应矩阵的行数x列数）。
卷积层模型参数包括卷积核 + 标量偏差。卷积核通常随机初值，然后不断迭代卷积核和偏差（= 反向传播误差修正？）。
如果该层卷积核形状 pxq，则称 pxq 卷积层。

二维卷积层输出（即一个二维数组）可看作输入在空间维度（宽和高）上某一级的表征，称为特征图（feature map）。
感受野（receptive field）：影响输出数组/矩阵中某一个元素/单元 x 的前向计算中所有 **可能的** （实际输入尺寸可能比感受野要小，因为可能存在跨层）输入区域 A，叫做 x 在 A 上的感受野（receptive field）。感受野的关系可不限于近邻两层，前面的层可以和后面的任意一层跨层构成关系。

图5.1（https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html）为例，输入中阴影部分的4个元素是输出中阴影部分元素的感受野。我们将图5.1中形状为 2×2 的输出记为 Y ，并考虑一个更深的卷积神经网络：将 Y 与另一个形状为 2×2 的核数组做互相关运算，输出单个元素 z 。那么，z 在 Y 上的感受野包括 Y 的全部4个元素，在输入上的感受野包括其中全部9个元素。

由此可见，更深的 CNN 加深了特征图的层数，加强了输入层的表征，使得输出层单元素的在输入层上的感受野变得更加广阔，从而捕获输入层上更大尺度的特征。

常使用“元素”一词描述数组或矩阵中的成员。神经网络，将元素也称为“单元”。含义明确时，不对这两个术语做严格区分。

一些问题：
如何通过变化输入和核数组将互相关运算表示成一个矩阵乘法？
如何构造一个全连接层来进行物体边缘检测？

### 填充（padding）和步幅（stride）

一般，如果输入形状 h×w，卷积窗口形状 k_h×k_w，则显然输出形状为 \(h - k_h + 1\)×\(w - k_w + 1\)。
也即卷积层输出形状由输入形状核卷积窗口形状决定。此处引入卷积层的两个超参数，padding 和 stride。

padding：在输入高宽两侧填充元素（通常是0元素）

一般我们会设置 padding 使输入和输出具有相同形状，可以方便构造网络时推测每个层的输出形状。
因前述关系可知，将总共的 padding 取为 p_h = k_h - 1，p_w = k_w - 1 可以做到。
又因为在形状上进行填充需要考虑位置，那么，
如果卷积窗口尺寸为奇数，即 2k+1，那么我们倾向于在行/列的上下/左右，对称的填充 k 行/列；
如果卷积窗口尺寸为偶数，即 2k，那么一种可能的填充是在行/列的上下/左右，不对称的填充 k/k-1 行/列。
特别注意，并不一定是方阵，行列长度可以不同。
（语言描述显得繁琐，但实际理解十分简单）
（因此？）CNN 通常使用奇数形状卷积核，方便在长宽上具有相同的填充个数。
对任意二维输入数组/矩阵 X，i 行 j 列元素表为 X\[i,j\]，若使用奇数卷积核，且输出数组/矩阵 Y 具有和 X 相同的形状，
那么 Y\[i,j\] 显然是由以 X\[i,j\] 为中心的卷积窗口（互相关）输出所得。

stride：每次滑动的行数和列数
Q: 请推算出步幅、填充、卷积窗口形状和输入间的关系

**小结**
- 填充可增加输出的高宽。常用于使输出与输入具有相同的高宽
- 步幅可分数减小输出的高宽到 1/n

### 多输入通道和多输出通道

## 循环神经网络

## 优化算法

## 计算性能

命令式和符号式混合编程
异步计算
自动并行计算
多 GPU 计算

## CV

### 图像增广 image augmentation

对于深度神经网络，必要大规模数据集。
图像增广：
- 解释一，通过对训练图像做一系列随机改变产生相似但不同的训练样本，扩大训练集规模。
- 解释二，随机改变训练样板可降低模型对某些属性的依赖，提高泛化能力

常见手段：
- 翻转和裁剪
- 变化颜色：亮度、对比度、饱和度、色调
- 多方法叠加

### 微调 fine tuning

必要性：
- 适用于 ImageNet 数据集的复杂模型直接使用可能在自制小数据集上训练过拟合
- 单靠个人或小团体很难收集到足额的图像数据集，从而训练得到的模型精度达不到实用要求

解决办法：
- 第一种，收集更多的数据。然而收集和标注数据会花费大量的时间和资金。如收集 ImageNet 数据集花费了数百万美元的研究经费。虽然目前的数据采集成本已降低不少，但成本仍然不可忽略。
- 第二种，应用迁移学习（transfer learning）：从源数据集习得的模型迁移到目标数据集

如，虽然 ImageNet 数据集的图像大多跟椅子无关，但在该数据集上训练的模型可以抽取较通用的图像特征，从而能够帮助识别边缘、纹理、形状和物体组成等。这些类似的特征对于识别椅子也可能同样有效。

Unsupervised pre-training 无监督预训练
Supervised pre-training 有监督预训练 = 迁移学习

迁移学习常用技术：fine tuning，4 个步骤：
1. 训练源模型
2. 复制了源模型上除输出层外的所有模型设计及其参数创建目标模型（该步骤基于两个假设：一、模型参数包含了所有源数据集上习得的知识且适用于目标数据集；二、源模型输出层与源数据集标签紧密相关，因此目标模型不采用）
3. 目标模型添加输出维度等同于目标数据集类别的输出层
4. 在目标数据集上再次训练目标模型，即输出层完全重新构建，其余层则基于源模型参数微调得到。

当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力。（来源？）

热狗识别的例子，Gluon 的 model_zoo 包提供常用预训练模型，更多 CV 预训练模型，可用 GluonCV 工具包。

> **Note**
> 一般微调参数会使用较小的学习率，而从头训练输出层可以使用较大的学习率

### 目标检测和边界框

目标检测（object detection）或物体检测：很多时候不仅想知道图中有哪些类，也想知道目标的位置。

一般 bounding box（边界框，缩写 bbox）从图像左上角到右下角

### 锚框

object detection 算法通常从输入图像采样大量区域然后判断是否包含感兴趣目标，
并调整区域边缘以更准确预测目标真实边界框（ground-truth bounding box）。
不同模型可能使用不同区域采样方法。
引入其中一种：
每个像素为中心生成多个大小和宽高比（aspect ratio）不同的 bbox。这些 bbox 就是锚框（anchor box）
https://zh.d2l.ai/chapter_computer-vision/anchor.html
的 9.4.1 中，s是缩小的比例，但是为什么不使用 r 而是 sqrt(r)？

经验。
已给出一系列大小 s_1, ..., s_n 和一组宽高比 r_1, ..., r_m。如果全部排列组合，
且针对每一个输入像素都使用 abox，那么输入图像会得到 whnm 个 abox。
虽然这样可能覆盖了全部真实 bbox，但计算复杂度过高。因此通常只取包含 s_1 或 r_1 的组合：
(s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
即对每个像素有 n+m-1 个 abox，从而对于整个图像有 wh(n + m - 1) 个 abox。

### 多尺度目标检测

虽然已使用前述办法减少了锚框数量，但数量依然很多（561 728像素图像，每个点 5 个锚框，就需要 200 多万个）。
因此一种简单的

# Trouble Shooting

### GPU 相关问题

如果在尝试运行一个 TensorFlow 程序时出现以下错误:

```
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory

```

请确认你正确安装了 GPU 支持, 参见 [相关章节](#install_cuda).

### 在 Linux 上

出现错误:

```
...
 "__add__", "__radd__",
             ^
SyntaxError: invalid syntax

```

解决方案: 确认使用 Python 2.7.

### 在 Mac OS X 上

出现错误:

```
import six.moves.copyreg as copyreg

ImportError: No module named copyreg

```

解决方案: TensorFlow 使用的 protobuf 依赖 `six-1.10.0`. 但是, Apple 的默认 python 环境 已经安装了 `six-1.4.1`, 该版本可能很难升级. 这里提供几种方法来解决该问题:

1.  升级全系统的 `six`:

    ```
     sudo easy_install -U six

    ```
2.  通过 homebrew 安装一个隔离的 python 副本:

    ```
     brew install python

    ```
3.  在[`virtualenv`](#virtualenv_install) 内编译或使用 TensorFlow.

出现错误:

```
>>> import tensorflow as tf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py", line 4, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python2.7/site-packages/tensorflow/python/__init__.py", line 13, in <module>
    from tensorflow.core.framework.graph_pb2 import *
...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py", line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02 \x03(\x0b\x32 .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
TypeError: __init__() got an unexpected keyword argument 'syntax'

```

这是由于安装了冲突的 protobuf 版本引起的, TensorFlow 需要的是 protobuf 3.0.0. 当前 最好的解决方案是确保没有安装旧版本的 protobuf, 可以使用以下命令重新安装 protobuf 来解决 冲突:

```
brew reinstall --devel protobuf

```

> 原文：[Download and Setup](http://tensorflow.org/get_started/os_setup.md) 翻译：[@doc001](https://github.com/PFZheng) 校对：[@yangtze](https://github.com/sstruct)

TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op 的执行步骤 被描述成一个图. 在
执行阶段, 使用会话执行执行图中的 op.
例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.
TensorFlow 支持 C, C++, Python 编程语言. 目前, TensorFlow 的 Python 库更加易用, 它提供了大量辅助
函数简化构建图的工作, 这些函数尚未被 C 和 C++ 库支持
三种语言的会话库 (session libraries) 是一致的
