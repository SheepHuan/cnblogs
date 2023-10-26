## 1 nn-meter 预测流程


## 2 构建tflite预测器

### 2.1 环境搭建
1. follow它的readme提示，准备nn-meter的安装
    ```bash 
    git clone https://github.com/microsoft/nn-Meter
    cd nn-Meter
    conda create -n nnmeter_tflite python=3.8
    # 当前nn-meter#8006ed6eaa62816c70737c9ff26a7445589bd36e支持到了2.11版本
    pip install -r docs/requirements/requirements_builder.txt
    # 安装nn-Meter
    pip install .
    ```
2. 将tflite的benchmark工具推送到手机设备上
我们从nn-meter上下载[benchmark文件](https://github.com/microsoft/nn-Meter/releases/tag/v2.0-data)，我选择了tflite_benchmark_tools_v2.7.zip文件。

    ```bash
    # 创建几个临时文件夹给nn-Meter存放文件
    adb shell "mkdir -p /mnt/sdcard/tflite_model"
    adb shell "mkdir -p /mnt/sdcard/tflite_kernel"
    # 推送benchmark文件到远程手机上
    adb push benchmark_model_cpu_gpu_v2.7 /data/local/tmp
    # 给benchmark设置可执行权限
    adb shell chmod +x /data/local/tmp/benchmark_model_cpu_gpu_v2.7
    ```
3. 创建workspace,准备后端
    ```bash
    nn-meter create --tflite-workspace /root/workspace/nn-Meter/workspace/RedmiK30Pro-sd865-tflite2.7cpu
    ```
    创建完之后，会出现configs/*.yaml文件,主要需要修改backend_config.yaml，其余两个不需要啥修改。
    - backend_config.yaml, 设置远程手机上的目录、benchmark位置，以及远程手机的地址(序列号或者IP)，这个参数结合2.1#step 2。
        ```yaml
        REMOTE_MODEL_DIR: /mnt/sdcard/tflite_bench
        BENCHMARK_MODEL_PATH: /data/local/tmp/benchmark_model_cpu_gpu_v2.7
        DEVICE_SERIAL: '3a9c4f5'
        KERNEL_PATH: /mnt/sdcard/tflite_kernel
        ```
    - predictorbuild_config.yaml,设置预测器相关的参数。
    - ruletest_config.yaml,设置OP融合规则相关的参数。

### 2.2 测试融合规则
在配置完环境和参数后，我们可以运行.py脚本自动化的执行OP融合测试和预测器了。nn-Meter提供了一些端到端的测试代码和分步的测试代码。
```python
# 参考文档: https://github.com/microsoft/nn-Meter/blob/main/docs/builder/test_fusion_rules.md#end-to-end-demo
workspace ="/root/workspace/nn-Meter/workspace/RedmiK30Pro-sd865-tflite2.7cpu"

from nn_meter.builder import profile_models, builder_config
builder_config.init(workspace) # initialize builder config with workspace
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases, detect_fusion_rule

# generate testcases
origin_testcases = generate_testcases()

# connect to backend
backend = connect_backend(backend_name='tflite_cpu')

# run testcases and collect profiling results
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)
```
执行结束后，我们的`{workspace}/fusion_rule_test/`文件夹下会出现测试结果。

### 2.3 构建kernel预测器

```python
# 参考文档: https://github.com/microsoft/nn-Meter/blob/main/docs/builder/build_kernel_latency_predictor.md#end-to-end-demo
workspace ="/root/workspace/nn-Meter/workspace/RedmiK30Pro-sd865-tflite2.7cpu"

from nn_meter.builder import builder_config
builder_config.init(workspace)

# build latency predictor for kernel
from nn_meter.builder import build_latency_predictor
build_latency_predictor(backend="tflite_cpu")
```

### 2.4 构建model预测器
同样根据[文档](https://github.com/microsoft/nn-Meter/blob/main/docs/builder/customize_predictor.md#use-customized-predictor-for-latency-prediction)步骤将2.2和2.3的OP融合规则和Kernel Predictor放到一个文件夹下,同时增加一个yaml配置文件，就可以注册一个Model Latency Predictor了。

1. 拷贝文件和重命名
    ```bash
    # 1. 将finegrained2.pkl复制到指定目录然后rename
    cp workspace/RedmiK30Pro-sd865-tflite2.7cpu/predictor_build/results/predictors/*finegrained2.pkl /root/workspace/nn-Meter/workspace/predictor/redmik30p_sd865_tflite2.7cpu

    #!/bin/bash
    # 遍历当前目录下所有的文件
    for file in *
    do
        # 判断文件名是否以"_finegrained2.pkl"结尾
        if [[ $file == *_finegrained2.pkl ]]
        then
            # 替换文件名中的"_finegrained2.pkl"为".pkl"
            new_name=${file/_finegrained2.pkl/.pkl}
            # 重命名文件
            echo "$new_name"
            mv "$file" "$new_name"
        fi
    done

    # 2. 融合规则
    cp workspace/RedmiK30Pro-sd865-tflite2.7cpu/fusion_rule_test/results/detected_fusion_rule.json /root/workspace/nn-Meter/workspace/predictor/redmik30p_sd865_tflite2.7cpu/fusion_rules.json
    ```
    目录树如下
    ```tree
    redmik30p_sd865_tflite2.7cpu
    |-- add.pkl
    |-- addrelu.pkl
    |-- avgpool.pkl
    |-- bn.pkl
    |-- bnrelu.pkl
    |-- channelshuffle.pkl
    |-- concat.pkl
    |-- conv-bn-relu.pkl
    |-- dwconv-bn-relu.pkl
    |-- fc.pkl
    |-- fusion_rules.json
    |-- global-avgpool.pkl
    |-- hswish.pkl
    |-- maxpool.pkl
    |-- relu.pkl
    |-- se.pkl
    |-- split.pkl
    ```
2. 写一个yaml文件索引文件位置
    ```yaml
    name: redmik30p_sd865_tflite2.7cpu
    version: 1.0
    category: cpu
    package_location: /root/workspace/nn-Meter/workspace/predictor/redmik30p_sd865_tflite2.7cpu
    kernel_predictors:
        - conv-bn-relu
        - dwconv-bn-relu
        - fc
        - global-avgpool
        - hswish
        - relu
        - se
        - split
        - add
        - addrelu
        - maxpool
        - avgpool
        - bn
        - bnrelu
        - channelshuffle
        - concat
    ```
3. 注册预测器
    ```bash
    # 注册
    nn-meter register --predictor /root/workspace/nn-Meter/workspace/predictor/redmik30p_sd865_tflite2.7cpu.yaml
    # 
    nn-meter --list-predictors
    ```
    成功注册后一般会显示
    ```text
    (nn-Meter) Successfully register predictor: redmik30p_sd865_tflite2.7cpu
    (nn-Meter) Supported latency predictors:
    (nn-Meter) [Predictor] cortexA76cpu_tflite21: version=1.0
    (nn-Meter) [Predictor] adreno640gpu_tflite21: version=1.0
    (nn-Meter) [Predictor] adreno630gpu_tflite21: version=1.0
    (nn-Meter) [Predictor] myriadvpu_openvino2019r2: version=1.0
    (nn-Meter) [Predictor] redmik30p_sd865_tflite2.7cpu: version=1.0
    ```
    
## 3 测试
### 3.1 预测值和实际值的差别
1. 基于Tensorflow2 API导出一个resnet50的模型
    ```Python

    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    # 加载模型
    model = ResNet50(weights='imagenet')

    full_model = tf.function(lambda x: model(x))
    shape = [1,224,224,3] #  model.inputs[0].shape
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./frozen_models",
                    name="frozen_graph.pb",
                    as_text=False)


    # 将模型转换为 TensorFlow Lite 格式，并保存为 .tflite 文件
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('resnet50.tflite', 'wb') as f:
        f.write(tflite_model)

    ```

2. 用nn-Meter预测
    ```Text

    ```
3. benchmark运行
   ```bash
    /data/local/tmp/benchmark_model_cpu_gpu_v2.7 --num_threads=4 \
    --graph=/mnt/sdcard/tflite_models/resnet50.tflite  \
    --warmup_runs=30 \
    --num_runs=50
    ```
    ```Text
    STARTING!
    Log parameter values verbosely: [0]
    Min num runs: [50]
    Num threads: [4]
    Min warmup runs: [30]
    Graph: [/mnt/sdcard/tflite_models/resnet50.tflite]
    #threads used for CPU inference: [4]
    Loaded model /mnt/sdcard/tflite_models/resnet50.tflite
    INFO: Initialized TensorFlow Lite runtime.
    INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
    INFO: Replacing 75 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions.
    The input model file size (MB): 102.161
    Initialized session in 98.471ms.
    Running benchmark for at least 30 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
    count=30 first=104448 curr=87126 min=86737 max=104448 avg=88622.5 std=3079

    Running benchmark for at least 50 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
    count=50 first=87163 curr=89038 min=86939 max=93704 avg=88199.2 std=1353

    Inference timings in us: Init: 98471, First inference: 104448, Warmup (avg): 88622.5, Inference (avg): 88199.2
    Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
    Memory footprint delta from the start of the tool (MB): init=134.562 overall=208.699
    ```

### 3.2 预测一些未训练的kernel模型
这里我选了一个SSD模型，它的算子种类如下:
```text
Add
BatchNormalization
Cast
Concat
Constant
ConstantOfShape
Conv
Exp
Gather
MaxPool
Mul
NonMaxSuppression
ReduceMin
Relu
Reshape
Shape
Slice
Softmax
Squeeze
Sub
TopK
Transpose
Unsqueeze
```
执行预测后`nn-meter predict --predictor redmik30p_sd865_tflite2.7cpu --predictor-version 1.0 --onnx /root/workspace/nn-Meter/workspace/models/ssd-12.onnx`
```Text
(nn-Meter) Start latency prediction ...
(nn-Meter) Empty shape information with Constant_339
(nn-Meter) Empty shape information with Shape_340
(nn-Meter) Empty shape information with Gather_341
(nn-Meter) Empty shape information with Constant_342
(nn-Meter) Empty shape information with Constant_343
(nn-Meter) Empty shape information with Unsqueeze_344
(nn-Meter) Empty shape information with Unsqueeze_345
(nn-Meter) Empty shape information with Unsqueeze_346
(nn-Meter) Empty shape information with Concat_347
(nn-Meter) Empty shape information with Reshape_348
(nn-Meter) Empty shape information with Constant_350
(nn-Meter) Empty shape information with Shape_351
(nn-Meter) Empty shape information with Gather_352
(nn-Meter) Empty shape information with Constant_353
(nn-Meter) Empty shape information with Constant_354
...
(nn-Meter) Empty shape information with Unsqueeze_scores
Traceback (most recent call last):
  File "/root/anaconda3/envs/nnmeter_tflite/bin/nn-meter", line 8, in <module>
    sys.exit(nn_meter_cli())
  File "/root/anaconda3/envs/nnmeter_tflite/lib/python3.8/site-packages/nn_meter/utils/nn_meter_cli/interface.py", line 266, in nn_meter_cli
    args.func(args)
  File "/root/anaconda3/envs/nnmeter_tflite/lib/python3.8/site-packages/nn_meter/utils/nn_meter_cli/predictor.py", line 56, in apply_latency_predictor_cli
    latency = predictor.predict(model, model_type) # in unit of ms
  File "/root/anaconda3/envs/nnmeter_tflite/lib/python3.8/site-packages/nn_meter/predictor/nn_meter_predictor.py", line 111, in predict
    self.kd.load_graph(graph)
  File "/root/anaconda3/envs/nnmeter_tflite/lib/python3.8/site-packages/nn_meter/kernel_detector/kernel_detector.py", line 19, in load_graph
    new_graph = convert_nodes(graph)
  File "/root/anaconda3/envs/nnmeter_tflite/lib/python3.8/site-packages/nn_meter/kernel_detector/utils/ir_tools.py", line 14, in convert_nodes
    type = node["attr"]["type"]
KeyError: 'type'
```

爆出的`Empty shape information`发生在`nn_meter/ir_converter/onnx_converter/converter.py#OnnxConverter.fetch_attrs`函数中。这导致返回的attr变量为空，最终报错。
nn-Meter在计算时间要获取OP的输入输出shape，这里的shape等算子并不像conv这也的OP，由很多可以建模的参数,nn-meter设计时没有考虑，所以这里报错了。这里总结了这个模型报错的的算子类型。

```Text
Add
Cast
Concat
Constant
ConstantOfShape
Exp
Gather
Mul
NonMaxSuppression
ReduceMin
Reshape
Shape
Slice
Softmax
Squeeze
Sub
TopK
Transpose
Unsqueeze
```
感觉这里的问题有点复杂,有些算子是nn-Meter训练过的,但是还是报错了比如`Add`,`Concat`.
<!-- 总结如下： -->
<!-- |    **算子类型**    | **是否报错** | **nn-Meter是否支持** | 
| :----------------: | :----------: |:----------: |
|        Add         |            |           |
| BatchNormalization |            |           |
|        Cast        |            |           |
|       Concat       |            |           |
|      Constant      |            |           |
|  ConstantOfShape   |            |           |
|        Conv        |            |           |
|        Exp         |            |           |
|       Gather       |            |           |
|      MaxPool       |            |           |
|        Mul         |            |           |
| NonMaxSuppression  |            |           |
|     ReduceMin      |            |           |
|        Relu        |            |           |
|      Reshape       |            |           |
|       Shape        |            |           |
|       Slice        |            |           |
|      Softmax       |            |           |
|      Squeeze       |            |           |
|        Sub         |            |           |
|        TopK        |            |           |
|     Transpose      |            |           |
|     Unsqueeze      |            |           | -->

### 3.3 nn-Meter的问题
1. 对于Tensorflow模型来说,nn-Meter可能会在ShapeInference出现问题.
2. nn-Meter目前对于仅支持一些CNN常用的算子.
3. nn-Meter支持的模型数据类型只有float类型和int32类型.

## 4. 参考资料
1. https://github.com/microsoft/nn-Meter
2. https://blog.csdn.net/ouening/article/details/104335552

## 5. 参考代码
1. 打印onnx模型的OP
    ```python
    import onnx
    model_file = "/root/workspace/nn-Meter/workspace/models/mobilenetv3small_0.onnx"   # ONNX模型文件路径
    model = onnx.load(model_file)
    op_types = set()
    for node in model.graph.node:
        op_types.add(node.op_type)
    op_types = list(op_types)
    [print(op_type) for op_type in op_types]
    ```
