
Pooling 把图像的小区域变成一个池子，或者说把小池子作为一个整体单元，把检测单元“池化”。具体做法可以从中取最大或平均。

池化过程在一般卷积过程后。池化（pooling） 的本质，其实就是采样。Pooling 对于输入的 Feature Map，选择某种方式对其进行降维压缩，以加快运算速度。

采用较多的一种池化过程叫**最大池化（Max Pooling）**，其具体操作过程如下：

![](https://pic2.zhimg.com/v2-2ce695fbd365c2be94521992d52ccefd_b.jpg)

池化过程类似于卷积过程，如上图所示，表示的就是对一个 ![](https://www.zhihu.com/equation?tex=4%5Ctimes4)
 feature map 邻域内的值，用一个 ![](https://www.zhihu.com/equation?tex=2%5Ctimes2)
 的 filter，步长为 2 进行‘扫描’，选择最大值输出到下一层，这叫做 Max Pooling。

max pooling 常用的 ![](https://www.zhihu.com/equation?tex=s%3D2)
 ， ![](https://www.zhihu.com/equation?tex=f%3D2)
 的效果：特征图高度、宽度减半，通道数不变。

还有一种叫**平均池化（Average Pooling）**, 就是从以上取某个区域的最大值改为求这个区域的平均值，其具体操作过程如下：

![](https://pic2.zhimg.com/v2-a47095dd0902990d387e21ae24e6f0b9_b.jpg)

如上图所示，表示的就是对一个 ![](https://www.zhihu.com/equation?tex=4%5Ctimes4)
 feature map 邻域内的值，用一个 ![](https://www.zhihu.com/equation?tex=2%5Ctimes2)
 的 filter，步长为 2 进行‘扫描’，计算平均值输出到下一层，这叫做 Mean Pooling。

**【池化层没有参数、池化层没有参数、池化层没有参数】** （重要的事情说三遍）

**池化的作用：**

（1）保留主要特征的同时减少参数和计算量，防止过拟合。

（2）invariance(不变性)，这种不变性包括 translation(平移)，rotation(旋转)，scale(尺度)。降低卷积层对目标位置敏感度。

Pooling 层说到底还是一个特征选择，信息过滤的过程。也就是说我们损失了一部分信息，这是一个和计算性能的一个妥协，随着运算速度的不断提高，我认为这个妥协会越来越小。

现在有些网络都开始少用或者不用 pooling 层了。

池化（Pooling）是卷积神经网络中的一个重要的概念，它实际上是一种形式的降采样。有多种不同形式的非线性池化函数，而其中 “最大池化（Max pooling）” 是最为常见的。它是将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。直觉上，这种机制能够有效的原因在于，在发现一个特征之后，它的精确位置远不及它和其他特征的相对位置的关系重要。池化层会不断地减小数据的空间大小，因此参数的数量和计算量也会下降，这在一定程度上也控制了过拟合。通常来说，CNN 的卷积层之间都会周期性地插入池化层。

池化层通常会分别作用于每个输入的特征并减小其大小。目前最常用形式的池化层是每隔 2 个元素从图像划分出的区块，然后对每个区块中的 4 个数取最大值。这将会减少 75% 的数据量。

除了最大池化之外，池化层也可以使用其他池化函数，例如 “平均池化” 甚至 “L2 - 范数池化” 等。

下图为最大池化过程的示意图：

![](https://image.jiqizhixin.com/uploads/editor/7536b511-213c-46e3-8359-72afb8e24080/1525383043664.jpg)

\[描述来源：维基百科；URL：<https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>]

\[图片来源：<http://cs231n.github.io/convolutional-networks/>]

## 发展历史

### 描述

过去，平均池化的使用较为广泛，但是由于最大池化在实践中的表现更好，所以平均池化已经不太常用。由于池化层过快地减少了数据的大小，目前文献中的趋势是使用较小的池化滤镜，甚至不再使用池化层。

### 主要事件

| 年份 | 事件 | 相关论文 / Reference |
| 2012 | 采用重叠池化方法，降低了图像识别的错误率 | Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). |
| 2014 | 将空金字塔池化方法用于 CNN，可以处理任意尺度的图像 | He, K., Zhang, X., Ren, S., & Sun, J. (2014, September). Spatial pyramid pooling in deep convolutional networks for visual recognition. In european conference on computer vision (pp. 346-361). Springer, Cham. |
| 2014 | 提出了一种简单有效的多规模无序池化方法 | Gong, Y., Wang, L., Guo, R., & Lazebnik, S. (2014, September). Multi-scale orderless pooling of deep convolutional activation features. In European conference on computer vision (pp. 392-407). Springer, Cham. |
| 2014 | 使用较小的池化滤镜 | Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071. |
| 2017 | 提出一种 Learning Pooling 方法 | Sun, M., Song, Z., Jiang, X., Pan, J., & Pang, Y. (2017). Learning pooling for convolutional neural network. Neurocomputing, 224, 96-104. |

## 发展分析

### 瓶颈

容易过快减小数据尺寸

### 未来发展方向

目前趋势是用其他方法代替池化的作用, 比如胶囊网络推荐采用动态路由来代替传统池化方法，原因是池化会带来一定程度上表征的位移不变性，传统观点认为这是一个优势，但是胶囊网络的作者 Hinton et al. 认为图像中位置信息是应该保留的有价值信息，利用特别的聚类评分算法和动态路由的方式可以学习到更高级且灵活的表征，有望冲破目前卷积网络构架的瓶颈。

Contributor: Yueqin Li
