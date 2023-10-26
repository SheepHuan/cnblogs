## 1 端侧推理框架经验总结
总结下最近用过的一些框架，并介绍他们的主要特点和转换过程。
1. onnxruntime
2. ncnn
3. mnn
4. tensorflow lite
5. huawei hiai
6. paddlelite

## 2 模型部署转换过程
我们以torchvision库中的resnet50模型为例，介绍模型转换的过程。
### 2.1 pytorch 转换到 others
pytorch是目前主流的模型训练框架，非常好用和简洁，但是目前我们主要还是使用一些第三方的推理框架进行模型部署，这里介绍一下pytorch如何转换成主流的onnx模型和tflite模型。这两个模型都可以后续拓展到其他的推理框架。
#### 2.1.1 pytorch to onnx
主要依赖于pytorch自身的`torch.onnx.export`函数实现，具体参考[pytorch官方文档](https://pytorch.org/docs/stable/onnx.html)。

我们主要注意一下几个点：
1. 模型的输入形状是动态的还是固定的。
2. 模型的导出的onnx opset的版本，不同版本支持导出的算子是不同的。
3. 模型的输入输出节点的名称，这个在后面转换到其他框架的时候会用到。
4. 是否需要导出模型的参数。

```python
import torchvision

model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()      
x = torch.randn(1, 3, 256, 256, requires_grad=True)
out = model(x)
# 固定形状
torch.onnx.export(  model,            
                    x,               
                    f"resnet50.onnx",
                    export_params=True,         #表示导出参数
                    opset_version=16,           # 设置opset 版本是16
                    do_constant_folding=True,   # 开启常量折叠
                    input_names = ['input'],    # 输入张量的名称
                    output_names = ['output'],  # 输出张量的名称
                )
# 动态形状
torch.onnx.export(  model,            
                    x,                        
                    f"resnet50-dynamic_axes.onnx",
                    export_params=True,     
                    opset_version=16,     
                    do_constant_folding=True, 
                    input_names = ['input'], 
                    output_names = ['output'],
                    dynamic_axes= {
                        "input": {0: 'batch_size', 2 : 'in_width', 3: 'int_height'},
                        "output": {0: 'batch_size', 2: 'out_width', 3:'out_height'}}
                    )
```

#### 2.1.2 pytorch to tf-lite
pytorch和tensorflow框架之间差异比较大，通常的做法是将pytorch模型先导出为onnx模型，然后将onnx模型转换为tensorflow模型，最后再将tensorflow模型转换为tf-lite模型。这个转换路径很长，而且中间很容易出错。我推荐使用阿里巴巴开源的转换框架[`tinynn`](https://github.com/alibaba/TinyNeuralNetwork.git)，它直接将pytorch模型转换为TF-Lite模型，安装和使用过程都非常简单，可参考其[github官方文档](https://github.com/alibaba/TinyNeuralNetwork/blob/main/examples/converter/README_zh-CN.md)。

```python
# pip install git+https://github.com/alibaba/TinyNeuralNetwork.git
from tinynn.converter import TFLiteConverter
import torchvision

model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()      
x = torch.randn(1, 3, 256, 256, requires_grad=False)
converter = TFLiteConverter(model, x,"resnet50.tflite",float16_quantization=False)
converter.convert()
```

#### 2.1.3 onnxsim库
我们可以用onnxsim工具对onnx模型进行优化，它可以实现模型的简化，提高模型执行效率和可读性。它同时支持**python api调用**和**命令行调用**。
1. python api
```python
from onnxsim import simplify
import onnx

onnx_model = onnx.load("resnet50.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "resnet50-sim.onnx")
```
2. 命令行
```bash
onnxsim resnet50.onnx resnet50-sim.onnx
```

### 2.2 onnx 转换到 others
这些步骤之前建议onnx先经过onnxsim优化，这样可以提高后续转换的效率。
#### 2.2.1 onnx to ncnn
ncnn是nihui开源的一个模型推理库，在Android设备上支持arm cpu和vulkan gpu两个设备上的推理，非常适合移动端的部署。而且它的cpu推理性能非常优秀，基本超过了大多数的框架。

它仅支持自己的模型格式`*.bin`和`*.param`分别代表模型的图和模型的参数。它的模型转换工具是一个名为`onnx2ncnn`的可执行文件。我们可以自己手动编译出`onnx2ncnn`，也可以在其官方仓库里直接下载编译好的可执行文件,比如进入该[链接](https://github.com/Tencent/ncnn/releases),ubuntu-2204.zip里面就有`onnx2ncnn`文件,github wiki上有[文档](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx#onnx-to-ncnn)说明如何使用该工具。

```bash
onnx2ncnn resnet50-sim.onnx resnet50.param resnet50.bin
```

#### 2.2.2 onnx to mnn
mnn是阿里巴巴开源的一个移动端模型推理库，它在android设备上支持arm cpu、vulkan gpu、 opencl gpu、nnapi和huawei npu的设备。

它支持多种模型格式转换到`*.mnn`模型，它的模型转换工具是一个叫做`MNNConvert`的可执行文件。它可以实现模型的转换，也可以实现模型的优化，比如fp16量化、图优化等。其[官方文档](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)说明了如何转换模型。

```bash
 MNNConvert --framework ONNX --modelFile resnet50-sim.onnx --MNNModel resnet50.mnn
```

#### 2.2.3 onnx to hiai

hiai是华为针对其自己的手机端麒麟芯片开发的模型推理加速框架，它支持CPU和NPU的推理，特别是它的达芬奇NPU，同时支持fp16和int8的模型推理，性能和能效非常高。但是我们需要注意hiai只能支持onnx opest 7~11，而且存在算子约束（部分算子参数有限制）。

它的模型转换工具是一个叫做`omg`的可执行文件。它支持多种模型格式转换到`*.om`模型，比如caffe、tensorflow、onnx等。转换工具的下载和使用方法，以及HIAI支持的OP算子在官方文档中。
1. [模型转换工具文档](https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/overall-parameter-0000001052966900)。
2. [模型算子约束文档](https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/npu-operator-constraints-0000001052845677)。

```bash
# 输出可以不写成resnet50.om，自己会加
omg --model resnet50-sim.onnx --framework 5 --output resnet50
```

#### 2.2.4 onnx to paddlie-lite

待更新
<!-- paddle-lite是百度开源的一个移动端模型推理库，它在android设备上支持arm cpu、opencl gpu、以及众多国产npu芯片的模型推理。它主要支持3种模型，每种模型格式各有优缺点。 -->

### 2.3 性能测试
待更新
