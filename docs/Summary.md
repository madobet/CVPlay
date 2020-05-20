## 术语和概念

无监督预训练(Unsupervised pre-training)
所谓的 Supervised pre-training 有监督预训练也可以把它称之为迁移学习

## Trouble Shooting

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
