基于视觉的目标检测（Object Detection，下称 OD）

基于视觉的目标检测方法综述

2012 年前，OD 用 Haar HOG（梯度直方图） LBP（局部二值模式）和
AdaBoost SVM（支持向量机） DPM（Deformable Part Model）等 ML（机器学习）方法（都属 FB）

2012 年后，Alex 等对 CNN 网络改进，CNN 基础上出 DCNN（Deep CNN），DCNN 以 top5 错误率比第二名低 10% 取胜 ILSVRC-2012，从而 CNN 再次被重视。同年， J.R.R. Uijlings 等用 Selective Search（选择性搜索），从全图中选出有用区域提取特征进行检测，奠定 CNN 衍生的 TWO-STATE 方法的基础。OD 速度精度大提升，CNN 成为主流。

OD 效果受 光照条件、遮挡情况、应用场景、物体尺寸等多因素影响，FB 受手工特征设计质量高低 + ML 自身缺陷影响，效果差 + 实际运用限制多。
CNN 提取深层特征，速度精度大提升。但 CNN 仍有不足，如，需精度速度间权衡。

分为：

-   基于手工特征（模板）（Feature-Based，下称 FB）

    -   手工特征
        -   LBP （特征） Local Binary Pattern
            对光照强鲁棒性，针对物体纹理特征，物体纹理特征类似背景纹理则检测效果降低。
            包括 Ojala 等人对此作出的改进：
            Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns
            提出具有灰度不变性和旋转不变性的 LBP 特征。
        -   HOG （特征）
            Block 在 Windows 内滑动，内部包含多 Cell。
            Block 的 HOG 特征描述 = Block 的梯度直方图 = Block 内所有 Cell 梯度直方图串联
            为光照和阴影不变性，需对比度归一化 Window 内重叠的 Block，Window 内所有 Block 梯度直方图串联未 Window 的 HOG 特征。
            最后，滑动 Window 收集整幅图片的 HOG 特征。
            HOG 提取的是物体边缘特征，对图像噪声敏感
            Navneet Dalal and Bill Triggs. Histograms of Oriented Gradients for Human Detection
        -   Haar（特征）
            四种基本 Haar，白色区域像素 - 黑色区域的像素和。即利用灰度图物体表面明暗变化提取特征，光照鲁棒性差
    -   ML
        -   AdaBoost
            选取 最优弱分类器\*N -> 强分类器，强分类器\*N 级联 -> 最终分类器
            问题：训练前期最优弱分类器较快，后期最优弱分类器选取困难，需大量样本长时间训练 + 易过拟合
            Robust RealTime Face Detection（PAUL VIOLA and MICHAEL J. JONES）
        -   SVN 核函数 低维度计算高维数据。SVM 分类效果与核函数选取、核函数参数有关，
            核函数参数不定（训练得到），参数选取与训练过程模型收敛速度和过拟合情况有关
        -   08 年 Pedro DPM 算法（引文 4），图像金字塔 提取 同一图像不同尺度特征。
            本质改进 HOG，取消 block，改变 Cell HOG 归一化方式，
            对图像金字塔采集特征得到 HOG 特征金字塔，顶部粗略 HOG 特征，底部精细 HOG
            DPM 设计
            -   根滤波器 root filter：匹配目标粗略 HOG，得目标整体轮廓 + 匹配分数
            -   部件滤波器 parts filter：匹配目标精细 HOG，得目标部分细节 + 匹配分数
                最后综合两种滤波器匹配得分确定检测结果
                DPM 连续三年 CVPR VOC 冠军
                DPM 只对刚性物体和高度可变物体检测效果好，模型复杂，训练过程中，正样本选取复杂，两种滤波器的位置作为训练的潜在变量（latent values 要靠经验的意思？），训练难度大，检测精度低。（我：前面说好现在又是精度低，这不自相矛盾？）
        -   Uijlings （引文 5）Selective Search
            图像先分割，分割区域相似性融合，得到全图目标可能存在区域，
            提取可能区域 HOG 特征，利用 SVMs 分类得到检测结果。
            结合了分割和枚举搜索的有点，只对可能区域提取 HOG 判断是否有目标，不用对全图提取特征判断，提升检测精度。
            对刚性和非刚性物体均可较好特出可能区域，成为提取目标可能存在区域的常用方法。
    -   改进 FB
        -   Feng 引文 6 增加新的 Haar 特征 + AdaBoost 算法，人脸检测精度 up，但训练时间长，抗干扰差，检测慢
        -   Wang 引文 7 Ou 引文 8 改进 AdaBoost 选取最优弱分类器时权值更新方法，降低过拟合对检测结果影响，检测速度精度 up，抗干扰依然差
        -   Lv 引 9 HOG 结合 LBP 药材识别，检测图像纹理和边缘，药材识别率 up，但抗干扰差，泛化弱
        -   Yan 引 10 AdaBoost 弱分类器与 SVM 级联成最终分类器用于车标识别
        -   Yang 引 11 结合 AdaBoost 和 SVM 识别棉叶螨危害等级，比单独 AdaBoost 和 SVM 更高正确率（但依然模型复杂，训练困难，抗干扰差？）

-   基于 CNN
    -   TWO-STATE（？）
        -   DCNN Alex 等 引 12 CNN 基础上设计 dropout 层，减少 CNN 训练中学习到得参数，
            从而减少训练过拟合，并设计新激活函数 ReLUs（Rectified Linear Units）加快收敛速度
            Alex 120w+ 图训练一大 DCNN（Deep CNN），网络结构见图，图片输入 DCNN 然后多次卷积池化、全连接，最后输出 1000 维向量，检测 1000 个类别。
            缺点：检测慢，精度低，训练慢。
        -   R-CNN Ross Girshick 等 引 13，做法：图中选出候选区（Region proposal），后利用 CNN 检测候选区得结果。步骤：
            1. Selective Search 提取 ~2000 候选区
            2. 每个候选区缩到（wrap）固定尺寸（227x227）输入 CNN 得 4096 维特征向量
            3. 特定类别 SVM 特征向量打分
            4. 非极大值抑制确定类别
            PASCAL VOC 2010 数据集上类别 mAP（平均检测精度，mean Average Precision）53.7%，VOC 2012 数据集 mAP 53.3%，比之前最好检测结果提高 30%，精度大幅提高原因与 RCNN 训练方式有关
