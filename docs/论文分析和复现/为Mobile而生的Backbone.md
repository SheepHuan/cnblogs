## 1 [MobileOne](https://arxiv.org/pdf/2206.04040.pdf)

### 1.1 重要分析(Insight)

1. 作者为了找到**端侧推理时模型架构中的瓶颈部分**，它基于CoreML在iPhone上进行了延迟测试。
2. 经过测试，作者认为对于移动设备上的高效架构，延迟与FLOPs适度相关，与Parameter count弱相关。这种相关性在桌面CPU上甚至更低。它举了例子,**skip connections和branch会带来巨大的访存开销**（想法1：有没有工作系统性地分析或总结过模型结构对于端侧推理的影响？）。**共享参数可以减少模型参数量但是带来了更高的FLOPs**。
   ![FLOPs vs Latency & Parameter Count vs Latency](http://pic.yanghuan.site/i/2023/05/26/6470a54fa42b4.png)
   <center style="">图 1-1</center> 
   
3. 为了分析激活函数的延迟，它设置30层的CNN网络，其中只有激活函数不同，进行了精度和延迟的测试对比。其中DynamicReLU 和 Dynamic Shift-Max提升了巨大的精度，但是延迟上也拉高了。MobileOne仅选用了ReLU作为激活函数。
   ![1685104265375.png](http://pic.yanghuan.site/i/2023/05/26/6470a68c605d4.png)
   <center style="">图 1-2</center> 

4. 作者认为影响运行性能的两个关键因素是**内存访问代价**和**并行度**。对于多分支网络来说每个分支的结果都必须存下来以计算下一个张量。为了避免这种内存瓶颈，网络中应当减少分支数量。另外如果采用一些类似Squeeze-Excite block(想法2：学习一下)中的Global Pooling的操作会导致强制内存同步的问题，从而也会带来延迟提升。作者设置了实验测试Squeeze-Excite block和skip connections对于模型延迟的影响。
   ![1685104848644.png](http://pic.yanghuan.site/i/2023/05/26/6470a8d38d491.png)
    <center style="">图 1-3</center> 

### 1.2 模型设计

根据1.1.1的分析和观察，作者设计模型时考虑到模型应当没有branch，且激活函数用ReLU，最后限制使用一些类似Squeeze-Excite block的模块。

![1685105206495.png](http://pic.yanghuan.site/i/2023/05/26/6470aa3bac069.png)
<center style="">图 1-4</center> 

1. block的设计主要参考了MobileNet-V1的3x3的depthwise卷积接一个1x1的pointwise卷积。然后是一个带着batchnorm的re-parameterizable的skip connection。  
PS：结构重参数化（structural re-parameterization）指的是首先构造一系列结构 (一般用于训练)，并将其参数等价转换为另一组参数 (一般用于推理或部署)，从而将这一系列结构等价转换为另一系列结构。**这个方法一般用于提升推理性能**。[解读模型压缩6：结构重参数化技术：进可暴力提性能，退可无损做压缩](https://zhuanlan.zhihu.com/p/370438999)
2. 训练和推理时用不同的结构（通过结构重参数化分支）。可提升模型性能。下图展示了重参数化的分支对模型精度的提升。
   ![1685105816100.png](http://pic.yanghuan.site/i/2023/05/26/6470ac9ab8065.png)
   <center style="">图 1-5</center> 
3. Model Scaling，一些work对模型的维度进行缩放，例如宽度、深度、分辨率来提升性能。MobileOne学了MobileNet-V2，在浅层时输入分辨率较大、在深层分辨率拉低。

### 1.3 实验评估

作为一个backbone,它首先在不同的任务上（目标检测、语义分割）进行了实验评估验证泛化性。随后和一些轻量网络进行了同任务的延迟和精度对比。我们主要关注延迟性能，下表中的Mobile设备应该指的是iPhone12，CPU和GPU应该是桌面端设备。

有趣的是，在Mobile设备上一些Transformer的变体模型也只有十几ms级别的延迟。

![1685106513786.png](http://pic.yanghuan.site/i/2023/05/26/6470af54901fb.png)
<center style="">图 1-6</center> 

![1685106685720.png](http://pic.yanghuan.site/i/2023/05/26/6470b00172f89.png)
<center style="">图 1-7</center> 

## 2 [MobileViT](https://arxiv.org/pdf/2110.02178.pdf)

### 2.1 分析和观点


1. 论文一开始作者就对比了ViT和CNN网络的区别，**ViT相比于CNN来说，更加重量级，难以优化、需要更多数据增强和L2正则化去避免过拟合、并且需要更多的Decoders去完成下游任务**。例如，ViT为了实现一个分割网络，它需要345 millions的参数量才能实现仅59 millions参数量的DeepLabV3。他们认为ViT需要更多的参数量是因为它缺乏CNN中的一些特性。
2. 结合CNN和Transformer可以得到鲁棒且高性能的ViTs（相比于普通的ViTs）。但是现有的一些工作的成果对于Mobile来说还是太大太重了。

### 2.2 MobileViT结构

![1685108218572.png](http://pic.yanghuan.site/i/2023/05/26/6470b5fd747ea.png)
<center style="">图 2-1</center> 

#### 2.2.1 标准ViT结构流程

如图2-1 (a)所示。
1. 标准的ViT模型会先将输入$X\in \mathbb{R}^{H\times W\times C}$在分辨率上reshape成一拉平(以patch为单位)的序列$X_f\in \mathbb{R}^{N\times PC}$。
2. 然后$X_f$经过`Linear`投影到一个固定维度空间$X_p\in \mathbb{R}^{N\times d}$。
3. 这个$X_p$变量经过`Position encoding`后就会送入$L\times$个Transformer里面学习每个patch内的表达。
4. 最后经过`Linear`层去获得结果。
注意这个流程中ViT模型忽视了整个空间上的归纳偏差(the spatial inductive bias that is inherent in CNNs)。所以ViT需要更多的参数去学习这个视觉表达。**MobileViT的核心idea就是要去让transformer像卷积一样去学整体的特征表达**。  

#### 2.2.2 MobileViT的流程

如上图(b)所示  
1. 对于输入$X\in \mathbb{R}^{H\times W\times C}$,MobileViT用$n\times n$的标准卷积层+一个point-wise($1\times 1$)卷积层得到一个$X_L\in \mathbb{R}^{H\times W\times d}$，$n \times n$的卷积用来编码局部的空间信息，而$1\times 1$的卷积通过去学习输入通道之间的线性组合。将张量投影到更高的$d$维，$d>C$。
2. MobileViT为了学到全局表达，它将$X_L$展开(`unfold`)为$N$个不重叠的扁平的patch,得到张量$X_U \in \mathbb{R}^{P\times N\times d}$,这里$P=wh,N=\frac{HW}{P}$,$P$是一个patch的大小。
3. $X_U$经过$L\times$个Transformer之后得到了张量$X_G\in\mathbb{R}^{P\times N\times d}$，转化公式如下
   $$X_G(p)={\rm{Transformer}}(X_U(p)),1\leq p \leq P$$
4. 为了让MobileViT既不损失patch之间的顺序关系也不丢失patch内部像素的空间顺序，它将$X_G$在经过折叠(`fold`)回$X_F\in\mathbb{R}^{H\times W\times d}$
5. $X_F$在通过$1 \times 1$卷积进行降维回到$H\times W\times C$的形状，然后和输入X进行拼接得到$H\times W\times 2C$的张量，再经过$n\times n$卷积去融合特征。

论文中对模型的设计的解释是这，$X_U(p)$编码的是经过卷积后的$n\times n$区域大小的局部信息。而$X_G(p)$是编码了在p-th位置的patch之间的全局信息，而$X_G$中的每个像素编码了变量$X$的所有像素的信息。所以作者认为MobileViT的有效感受野大小是$H\times W$。形象的分析如下：

![1685119189267.png](http://pic.yanghuan.site/i/2023/05/27/6470e0d6d08a3.png)
<center style="">图 2-2</center> 

如上图演示，红色的像素**通过Transformer注意**(步骤3)到了周围的蓝色像素（蓝色像素是周围其他patches中和红色像素对应的位置），而又因为蓝色像素早就先一步通过**卷积编码了自己这个patch内的邻居像素的信息**(步骤1)。所以当红色像素注意到了蓝色像素们的时候就就是获取到了整张图的所有像素的编码信息。黑色的块代表patch，灰色的块代表一个像素。

#### 2.2.3 MobileViT的思考和总结

1. Relationship to convolutions.
2. Light-weight.
3. Computational cost.
4. MobileViT architecture.

### 2.3 多尺度训练

文章中提到一般让基于ViT的模型的去学习多尺度的表达的方式是`fine-tuning`。而对于MobileViT来说，它和CNN一样可以在多尺度训练上获得更好的精度效果。

### 2.4 实验评估

待更新

## 3 [MobileViT v2](https://arxiv.org/pdf/2206.02680.pdf)

### 3.1 分析和观点

1. 作者认为MobileViT虽然实现了高性能的CV精度，但是相对于CNN网络来说仍存在高延迟的问题。
2. 作者认为MobileViT的性能瓶颈在于MHA(Multi-headed self-attention)。MHA算法复杂度是$O(k^2)$,$k$是tokens(patches)数量。
3. 另外MHA中还存在一些计算密集型的OP，例如注意力的矩阵乘。


基于以上观点，因此作者提出了一个新的优化视角：
**Can self-attention in transformer block be optimized for resource-constrained devices?**
这篇文章中提出了新的self-attention——separable self-attention，它仅有$O(k)$的复杂度，同时它替换了一些计算密集型的操作。

### 3.2 可分离的自注意力

![1685168407393.png](http://pic.yanghuan.site/i/2023/05/27/6471a118c8086.png)
<center>图 3-1</center> 

#### 3.2.1 朴素的MHA流程

这段结合代码，我再理解理解

1. 通常MHA有一个输入${\bf x} \in \mathbb{R}^{k\times d}$,$k$个$d$维的token embeddings。
2. ${\bf x}$被输入给3个分支，分别叫做Query $\mathcal{Q}$,Key $\mathcal{K}$,Value $\mathcal{V}$,每个分支都是由$h$个Linear Layers组成的($h$就是head数量，这$h$的layer是并行关系，不是串行关系?)，这使得transformer可以学到输入的不同视角。
3. 然后$\mathcal{Q}$和$\mathcal{K}$的输出进行点乘，结果再经过softmax操作($\sigma$)归一化后得到了注意力矩阵${\bf a}\in \mathbb{R}^{k\times k \times h}$。(注意$\mathcal{K}$的输出经过了Transpose,所以点乘)
4. 然后注意力矩阵${\bf a}$和 $\mathcal{V}$分支的输出再经过点成得到权重和(weighted sum)输出${\bf y_w} \in \mathbb{R}^{k\times d_h \times h}$,其中$d_h=\frac{d}{h}$
5. 然后$h$个$y_w$进行连接得到$k$维度的张量，经过最后的Linear Layer得到MHA的输出${\bf y} \in\mathbb{R}^{k\times d}$

#### 3.2.2 可分离的MHA

1. 可分离的MHA也有一个输入${\bf x} \in \mathbb{R}^{k\times d}$,$k$个$d$维的token embeddings。
2. 输入${\bf x}$也会被喂给三个分支，分别叫做Input $\mathcal{I}$,Key $\mathcal{K}$,Value $\mathcal{V}$。
3. Input分支将${\bf x}$的每个$d$维的token张量通过Linear层映射为一个标量。输出就是一个$k$维的向量，然后通过一个softmax得到上下文得分${\bf c_s \in  \mathbb{R}^{k}}$
4. 经过Key分支，输入${\bf x}$被线性投影到$d$维空间得到输出${\bf x_K} \in \mathbb{R}^{k\times d}$
5. 然后Input分支的输出和Key分支输出进行权重和计算得到${\bf c_v}$
   $${\bf c_v}=\sum_{i=1}^{k}{{\bf c_s}(i){\bf x_K}(i)}$$
   这里的${\bf c_v}$就是类似注意力矩阵${\bf a}$的存在，它也编码了输出$\bf x$中的所有tokens的信息
6. Value分支中输入${\bf x}$被线性投影到$d$维空间得到输出${\bf x_K} \in \mathbb{R}^{k\times d}$，然后接一个ReLU激活函数得到${\bf x_V}\in \mathbb{R}^{k\times d}$。
7. 为了让$\bf x$的所有tokens共享$\bf c_v$中的上下文编码信息，作者将$\bf x_V$和$\bf c_v$进行元素乘操作，结果再输入给其他Linear层,最后的输出结果${\bf y} \in \mathbb{R}^{k\times d}$

下图中，作者对比了不同数量的tokens(k值)的输入对于self-attention结构的延迟影响

![1685168139172.png](http://pic.yanghuan.site/i/2023/05/27/6471a00cd74aa.png)
<center style="">图 3-2</center> 

![1685171164604.png](http://pic.yanghuan.site/i/2023/05/27/6471abde8cbd0.png)
<center style="">图 3-3</center> 

#### 3.2.3 计算量的分析

<!-- ## 4 [Mobile-Former](https://arxiv.org/pdf/2108.05895.pdf)

![1685166847700.png](http://pic.yanghuan.site/i/2023/05/27/64719b0144124.png)
<center style="">图 4-1</center>  -->

## 5 性能评估

### 5.1 MobileOne

#### 5.1.1 导出模型

先基于Python Pytorch导出了onnx模型,预训练模型下载了unfused模型，地址如下:https://github.com/apple/ml-mobileone/tree/main

```Python
from model.mobileone import mobileone, reparameterize_model
import torch

def export(tag):
    model=mobileone(variant=tag)
    checkpoint = torch.load(f'ckpts/mobileone/mobileone_{tag}_unfused.pth.tar')
    model.load_state_dict(checkpoint)
    model.eval()      
    model_eval = reparameterize_model(model)

    # 默认输入是3通道
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model_eval(x)
    # Export the model
    torch.onnx.export(model_eval,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f"export_models/mobileone_{tag}-opset12.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    )

for tag in ['s0','s1','s2','s3','s4']:
    export(tag)
```

#### 5.1.2 Onnxruntime测试

我在HUAWEI Mate30 Pro(Kirin 990)上用CPU测试了5个模型

| **HW_M30P**  | **rounds** | **num_threads** | **mean (ms)** | **std (ms)** |
| :----------: | :--------: | :-------------: | :-----------: | :----------: |
| mobileone-s0 |     30     |        1        |   86.600000   |   0.489898   |
| mobileone-s0 |     30     |        2        |   78.000000   |      0       |
| mobileone-s0 |     30     |        3        |   75.600000   |   0.489898   |
| mobileone-s0 |     30     |        4        |   73.866667   |   0.339935   |
| mobileone-s1 |     30     |        4        |  130.833333   |   0.372678   |
| mobileone-s2 |     30     |        4        |  164.900000   |   0.300000   |
| mobileone-s3 |     30     |        4        |  207.133333   |   0.339935   |
| mobileone-s4 |     30     |        4        |  303.600000   |   0.489898   |

### 5.2 MobileViT v1,v2

#### 5.2.1 导出模型

在ml-cvnets仓库下，我新建了`export_onnx.py`,填写了一下代码，将两个模型导出到了onnx上。

```Python
import torch
from cvnets.models.classification.mobilevit import MobileViT
from cvnets.models.classification.mobilevit_v2 import MobileViTv2
from options.opts import get_eval_arguments

def export(model,name,opset):
    model.eval()      
    # 默认输入是3通道
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f"export_models/{name}-opset{opset}.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=opset,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    )
# python export_mobilevit.py  --common.config-file /workspace/ml-cvnets/config/classification/imagenet/mobilevit_v2.yaml
# python export_mobilevit.py  --common.config-file /workspace/ml-cvnets/config/classification/imagenet/mobilevit.yaml
if __name__=="__main__":
    
    from onnx import __version__, IR_VERSION
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
    opts = get_eval_arguments()
    model = MobileViT(opts)
    export(model,"mobilevit_v1",17)
   #  model = MobileViTv2(opts)
   #  export(model,"mobilevit_v2",18)
    

```
BUG: MobileViTv2中的`col2im`算子在opset_version 18上转换时报错。

#### 5.2.2 Onnxruntime测试
我在HUAWEI Mate30 Pro(Kirin 990)上用CPU测试了1个模型。  
有空再研究构造结构仅MHD模块不同的v1,v2的对比。
| HW_M30P      | rounds | num_threads | mean (ms)  | std (ms) |
| ------------ | ------ | ----------- | ---------- | -------- |
| mobilevit v1 | 30     | 1           | 304.966667 | 0.604612 |
| mobilevit v1 | 30     | 2           | 189.100000 | 0.300000 |
| mobilevit v1 | 30     | 3           | 163.866667 | 0.426875 |
| mobilevit v1 | 30     | 4           | 144.000000 | 0.000000 |
| mobilevit v2 | 30     | 1           |            |          |
| mobilevit v2 | 30     | 2           |            |          |
| mobilevit v2 | 30     | 3           |            |          |
| mobilevit v2 | 30     | 4           |            |          |


#### onnxsim优化

## 6 参考文献
[1] Vasu P K A, Gabriel J, Zhu J, et al. MobileOne: An Improved One Millisecond Mobile Backbone[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 7907-7917.  
[2] Mehta S, Rastegari M. Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer[J]. arXiv preprint arXiv:2110.02178, 2021.  
[3] Mehta S, Rastegari M. Separable self-attention for mobile vision transformers[J]. arXiv preprint arXiv:2206.02680, 2022.  
[4] Chen Y, Dai X, Chen D, et al. Mobile-former: Bridging mobilenet and transformer[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 5270-5279.  
