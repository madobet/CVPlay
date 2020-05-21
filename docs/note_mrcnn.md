论文：<http://cn.arxiv.org/pdf/1703.06870v3>

本文主要是针对论文的详细解析，选出文章各部分的关键点，方便阅读立即。

**目录：**

[摘要：](#%E6%91%98%E8%A6%81%EF%BC%9A)

[1、Introduction](#1%E3%80%81Introduction)

[3、Mask R-CNN](#3%E3%80%81Mask%20R-CNN)

[3.1 Implementation Details](#3.1%20Implementation%20Details)

[4、Experiments: Instance Segmentation](#4%E3%80%81Experiments%3A%20Instance%20Segmentation)

[4.1 Main Results](#4.1%20Main%20Results)

[4.2  Ablation Experiments（剥离实验）](#4.2%C2%A0%C2%A0Ablation%20Experiments%EF%BC%88%E5%89%A5%E7%A6%BB%E5%AE%9E%E9%AA%8C%EF%BC%89)

[4.3. Bounding Box Detection Results      ](#4.3.%20Bounding%20Box%20Detection%20Results%C2%A0%20%C2%A0%20%C2%A0%C2%A0)

[4.4. Timing ](#4.4.%20Timing%C2%A0)

[5. Mask R-CNN for Human Pose Estimation](#5.%20Mask%20R-CNN%20for%20Human%20Pose%20Estimation)

[Appendix A: Experiments on Cityscapes](#Appendix%20A%3A%20Experiments%20on%20Cityscapes)

[Implementation:](#Implementation%3A)

[Results：](#Results%EF%BC%9A)

[Appendix B: Enhanced Results on COCO](#Appendix%20B%3A%20Enhanced%20Results%20on%20COCO)

[Instance Segmentation and Object Detection](#Instance%20Segmentation%20and%20Object%20Detection)

[Keypoint Detection](#Keypoint%20Detection)

* * *

## 摘要：

-   Mask RCNN 可以看做是一个通用实例分割架构。
-   Mask RCNN 以 Faster RCNN 原型，增加了一个分支用于分割任务。
-   Mask RCNN 比 Faster RCNN 速度慢一些，达到了 5fps。
-   可用于人的姿态估计等其他任务；

![](https://img-blog.csdn.net/20181017160239157?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 1、Introduction

-   实例分割不仅要正确的找到图像中的 objects，还要对其精确的分割。所以 Instance Segmentation 可以看做 object dection 和 semantic segmentation 的结合。

-   Mask RCNN 是 Faster RCNN 的扩展，对于 Faster RCNN 的每个 Proposal Box 都要使用 FCN 进行语义分割，分割任务与定位、分类任务是同时进行的。

-   引入了 RoI Align 代替 Faster RCNN 中的 RoI Pooling。因为 RoI Pooling 并不是按照像素一一对齐的（pixel-to-pixel alignment），也许这对 bbox 的影响不是很大，但对于 mask 的精度却有很大影响。使用 RoI Align 后 mask 的精度从 10% 显著提高到 50%，第 3 节将会仔细说明。

-   引入语义分割分支，实现了 mask 和 class 预测的关系的解耦，mask 分支只做语义分割，类型预测的任务交给另一个分支。这与原本的 FCN 网络是不同的，原始的 FCN 在预测 mask 时还用同时预测 mask 所属的种类。

-   没有使用什么花哨的方法，Mask RCNN 就超过了当时所有的 state-of-the-art 模型。

-   使用 8-GPU 的服务器训练了两天。

-   相比于 FCIS，FCIS 使用全卷机网络，同时预测物体 classes、boxes、masks，速度更快，但是对于重叠物体的分割效果不好。

## 3、Mask R-CNN

-   **Mask R-CNN 基本结构：**与 Faster RCNN 采用了相同的 two-state 步骤：首先是找出 RPN，然后对 RPN 找到的每个 RoI 进行分类、定位、并找到 binary mask。这与当时其他先找到 mask 然后在进行分类的网络是不同的。
-   **Mask R-CNN 的损失函数**：![](https://private.codecogs.com/gif.latex?L%20%3D%20L%7B_%7Bcls%7D%7D%20+%20L%7B_%7Bbox%7D%7D%20+%20L%7B_%7Bmask%7D%7D)
-   **Mask 的表现形式 (Mask Representation)：**因为没有采用全连接层并且使用了 RoIAlign，可以实现输出与输入的像素一一对应。
-   **RoIAlign：**RoIPool 的目的是为了从 RPN 网络确定的 ROI 中导出较小的特征图 (a small feature map，eg 7x7)，ROI 的大小各不相同，但是 RoIPool 后都变成了 7x7 大小。RPN 网络会提出若干 RoI 的坐标以\[x,y,w,h] 表示，然后输入 RoI Pooling，输出 7x7 大小的特征图供分类和定位使用。问题就出在 RoI Pooling 的输出大小是 7x7 上，如果 RON 网络输出的 RoI 大小是 8\*8 的，那么无法保证输入像素和输出像素是一一对应，首先他们包含的信息量不同（有的是 1 对 1，有的是 1 对 2），其次他们的坐标无法和输入对应起来（1 对 2 的那个 RoI 输出像素该对应哪个输入像素的坐标？）。这对分类没什么影响，但是对分割却影响很大。RoIAlign 的输出坐标使用插值算法得到，不再量化；每个 grid 中的值也不再使用 max，同样使用差值算法。

![](https://img-blog.csdn.net/20181017160425513?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

-   **Network Architecture:** 为了表述清晰，有两种分类方法

1.  使用了不同的 backbone：resnet-50，resnet-101，resnext-50，resnext-101；
2.  使用了不同的 head Architecture：Faster RCNN 使用 resnet50 时，从 CONV4 导出特征供 RPN 使用，这种叫做 ResNet-50-C4
3.  作者使用除了使用上述这些结构外，还使用了一种更加高效的 backbone——FPN

![](https://img-blog.csdn.net/20181017170531183?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 3.1 Implementation Details

使用 Fast/Faster 相同的超参数，同样适用于 Mask RCNN

-   **Training:**

           1、与之前相同，当 IoU 与 Ground Truth 的 IoU 大于 0.5 时才会被认为有效的 RoI，![](https://private.codecogs.com/gif.latex?L%7B_%7Bmask%7D%7D)
只把有效 RoI 计算进去。

           2、采用 image-centric training，图像短边 resize 到 800，每个 GPU 的 mini-batch 设置为 2，每个图像生成 N 个 RoI，对于 C4                    backbone 的 N=64，对于 FPN 作为 backbone 的，N=512。作者服务器中使用了 8 块 GPU，所以总的 minibatch 是 16，                      迭代了 160k 次，初始 lr=0.02，在迭代到 120k 次时，将 lr 设定到 lr=0.002，另外学习率的 weight_decay=0.0001，                            momentum = 0.9。如果是 resnext，初始 lr=0.01, 每个 GPU 的 mini-batch 是 1。

           3、RPN 的 anchors 有 5 种 scale，3 种 ratios。为了方便剥离、如果没有特别指出，则 RPN 网络是单独训练的且不与 Mask R-                  CNN 共享权重。但是在本论文中，RPN 和 Mask R-CNN 使用一个 backbone，所以他们的权重是共享的。

            （Ablation Experiments 为了方便研究整个网络中哪个部分其的作用到底有多大，需要把各部分剥离开）

-   **Inference：**在测试时，使用 C4 backbone 情况下 proposal number=300，使用 FPN 时 proposal number=1000。然后在这些 proposal 上运行 bbox 预测，接着进行非极大值抑制。mask 分支只应用在得分最高的 100 个 proposal 上。顺序和 train 是不同的，但这样做可以提高速度和精度。mask 分支对于每个 roi 可以预测 k 个类别，但是我们只要背景和前景两种，所以只用 k-th mask，k 是根据分类分支得到的类型。然后把 k-th mask resize 成 roi 大小，同时使用阈值分割 (threshold=0.5) 二值化

## 4、Experiments: Instance Segmentation

### 4.1 Main Results

在下图中可以明显看出，FCIS 的分割结果中都会出现一条竖着的线 (systematic artifacts)，这线主要出现在物体重的部分，作者认为这是 FCIS 架构的问题，无法解决的。但是在 Mask RCNN 中没有出现。

![](https://img-blog.csdn.net/20181017224820503?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://img-blog.csdn.net/20181017224519524?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 4.2  Ablation Experiments（剥离实验）

-   **Architecture:**
    从 table 2a 中看出，Mask RCNN 随着增加网络的深度、采用更先进的网络，都可以提高效果。注意：并不是所有的网络都是这样。
-   **Multinomial vs. Independent Masks:(mask 分支是否进行类别预测)** 从 table 2b 中可以看出，使用 sigmoid(二分类) 和使用 softmax(多类别分类) 的 AP 相差很大，证明了分离类别和 mask 的预测是很有必要的
-   **Class-Specific vs. Class-Agnostic Masks:**                                                                                                                            目前使用的 mask rcnn 都使用 class-specific masks，即每个类别都会预测出一个 mxm 的 mask，然后根据类别选取对应的类别的 mask。但是使用 Class-Agnostic Masks，即分割网络只输出一个 mxm 的 mask，可以取得相似的成绩 29.7vs30.3
-   **RoIAlign:** tabel 2c 证明了 RoIAlign 的性能
-   **Mask Branch:**
    tabel 2e，FCN 比 MLP 性能更好

### 4.3. Bounding Box Detection Results      

![](https://img-blog.csdn.net/20181017231827951?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

-   Mask RCNN 精度高于 Faster RCNN
-   Faster RCNN 使用 RoI Align 的精度更高
-   Mask RCNN 的分割任务得分与定位任务得分相近，说明 Mask RCNN 已经缩小了这部分差距。

### 4.4. Timing 

-   **Inference：**195ms 一张图片，显卡 Nvidia Tesla M40。其实还有速度提升的空间，比如减少 proposal 的数量等。
-   **Training：**ResNet-50-FPN on COCO trainval35k takes 32 hours  in our synchronized 8-GPU implementation (0.72s per 16-image mini-batch)，and 44 hours with ResNet-101-FPN。

## 5. Mask R-CNN for Human Pose Estimation

![](https://img-blog.csdn.net/20181017234023362?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

让 Mask R-CNN 预测 k 个 masks，每个 mask 对应一个关键点的类型，比如左肩、右肘，可以理解为 one-hot 形式。

-   使用 cross entropy loss，可以鼓励网络只检测一个关键点;
-   ResNet-FPN 结构
-   训练了 90k 次，最开始 lr=0.02，在迭代 60k 次时，lr=0.002,80k 次时变为 0.0002

## Appendix A: Experiments on Cityscapes

包含 fine annotations images：2975 train ，500 val，1525 test

图片大小 2048x1024

使用 COCO 的 AP 作为评价指标

数据十分不平衡！

![](https://img-blog.csdn.net/20181017234631335?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### Implementation:

-   使用 ResNet-FPN-50 作为 back bone，不适用 ResNet-FPN-101 是因为数据集小，没什么提升。
-   训练时，图像短边从\[800,1024] 随机选择, 可以减小过拟合。
-   在预测时，图像短边都是 1024
-   每个 GPU 的 mini-batch 为 1, 共 8 个 GPU
-   训练 24k 次，初始 lr 为 0.01,18k 时减小到 0.001,
-   总共训练时间 8 小时

### Results：

![](https://img-blog.csdn.net/20181017235436579?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

-   person 和 car 存在大量的类内重叠，给分割网络提出了挑战，但是 Mask-RCNN 成功解决了
-   这个数据集十分不平衡，truck，bus，train 的数据量很少，所以使用的 coco 数据集预训练 Mask RCNN，分析上表，其他网络预测准确率低也主要低在 truck，bus，train 三个类别上，所以使用 coco 预训练还是很有用的。
-   验证数据集 val 和测试数据集 test AP 的差距较大，主要原因在于 truck，bus，train 三类训练数据太少，person，car 这种训练数据多的类别就不存在这种现象。即使使用 coco 数据集进行预训练也无法消除这种 bias。

## Appendix B: Enhanced Results on COCO

### Instance Segmentation and Object Detection

![](https://img-blog.csdn.net/20181018002654379?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdkb25nd2VpMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 使用一些技巧，可提高精度。可以在这里找到之后的更新：<https://github.com/facebookresearch/Detectron>

-   **Updated baseline：**使用不同的超参数**，**延长迭代次数至 180k 次，初始学习率不变，在 120k 和 160k 时减小 10 倍，NMS 的阈值从默认的 0.3 改为 0.5。
-   **End-to-end training:**之前的学习都是先训练 RPN，然后训练 Mask RCNN。
-   **ImageNet-5k pre-training:** 数据量比 coco 增加了 5 倍，预训练更有效
-   **Train-time augmentation:**训练数据增强
-   **Model architecture:**使用 152-layer 的 ResNeXt
-   **Non-local：**
-   **Test-time augmentation:**

### Keypoint Detection

用时再看吧。。。

###

* * *

 <https://blog.csdn.net/wangdongwei0/article/details/83110305>
