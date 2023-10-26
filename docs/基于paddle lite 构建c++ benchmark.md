
根据PaddleLite v2.12版本的[文档](https://www.paddlepaddle.org.cn/lite/v2.12/performance/benchmark_tools.html),Paddle已经提供了一个再Linux,MacOS以及Android平台上的C++ Benchmark用于评测模型性能.

## 1 PaddleLite 编译官方benchmark

### 1.1 环境准备

```bash
# 1. 拉取分支
git clone https://github.com/PaddlePaddle/Paddle-Lite.git --branch release/v2.13
# 2. 确认NDK版本号,PaddleLite的编译叫脚本要求NDK文件夹的名称是r*c的形式,而SDK中的NDK版本是版本号的形式,所以我们手动下载NDK
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip

# 3.
cd Paddle-Lite && git checkout release/v2.13

```

### 1.2 编译benchmark和测试

1. ARM CPU + OpenCL

    ```bash
    # 4. 我们目前只要Android的部分功能即可
    bash lite/tools/build_android.sh --arch=armv8 --toolchain=clang --with_opencl=ON --with_profile=ON --android_api_level=27 --with_benchmark=ON full_publish


    ```

    在`Paddle-Lite/build.lite.android.armv8.clang/lite/api/tools/benchmark`下有个`benchmark_bin`的可执行文件

2. 下载[MobileNetV1](https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz)

3. 推送benchmark文件和model到手机上并推理

    ```bash
    # 推送
    adb -s 3a9c4f5 shell mkdir /data/local/tmp/benchmark
    adb -s 3a9c4f5 push 3rd-party/Paddle-Lite/build.lite.android.armv8.clang/lite/api/tools/benchmark/benchmark_bin /data/local/tmp/benchmark
    adb -s 3a9c4f5 shell "chmod +x /data/local/tmp/benchmark_bin"
    adb -s 3a9c4f5 push MobileNetV1  /data/local/tmp/benchmark
    
    # 推理
    adb -s 3a9c4f5 shell "cd /data/local/tmp/benchmark;
    ./benchmark_bin \
        --model_file=MobileNetV1/inference.pdmodel \
        --param_file=MobileNetV1/inference.pdiparams \
        --input_shape=1,3,224,224 \
        --warmup=10 \
        --repeats=20 \
        --backend=arm"

    ```
    
    结果

    ```text
    ======= Opt Info =======
    Load paddle model from MobileNetV1/inference.pdmodel and MobileNetV1/inference.pdiparams
    Save optimized model to MobileNetV1/opt.nb

    ======= Device Info =======
    Brand: Redmi
    Device: lmi
    Model: Redmi K30 Pro
    Android Version: 10
    Android API Level: 29

    ======= Model Info =======
    optimized_model_file: MobileNetV1/opt.nb
    input_data_path: All 1.f
    input_shape: 1,3,224,224
    output tensor num: 1
    --- output tensor 0 ---
    output shape(NCHW): 1 1000 
    output tensor 0 elem num: 1000
    output tensor 0 mean value: 0.001
    output tensor 0 standard deviation: 0.00219647

    ======= Runtime Info =======
    benchmark_bin version: 523d53971
    threads: 1
    power_mode: 0
    warmup: 10
    repeats: 20
    result_path: 

    ======= Backend Info =======
    backend: arm
    cpu precision: fp32

    ======= Perf Info =======
    Time(unit: ms):
    init  = 19.825      
    first = 77.280      
    min   = 60.708      
    max   = 173.598     
    avg   = 98.043
    ```

    这个benchmark需要自己指定输入的shape。


## 2 构建自定义的benchmark

### 2.1 编译

```bash
# --with_profile=ON 需要打开full_publish选项
bash lite/tools/build_android.sh --arch=armv8 --toolchain=clang --android_stl=c++_shared --with_opencl=ON --with_profile=ON --android_api_level=27 full_publish
```

通过上述命令编译出`inlucde/`,`lib/`文件，位于`Paddle-Lite/build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx`文件夹下。

```Text
cxx
|-- include
|   |-- paddle_api.h
|   |-- paddle_image_preprocess.h
|   |-- paddle_lite_factory_helper.h
|   |-- paddle_place.h
|   |-- paddle_use_kernels.h
|   |-- paddle_use_ops.h
|   `-- paddle_use_passes.h
`-- lib
    |-- libpaddle_api_full_bundled.a
    |-- libpaddle_api_light_bundled.a
    |-- libpaddle_full_api_shared.so
    `-- libpaddle_light_api_shared.so
```
我们将`inlucde/`,`lib/`文件引入cmake工程就能用了。

### 2.2 编写benchmark
我的代码开源在[github](https://github.com/SheepHuan/UnifiedModelBenchmark/blob/onnxruntime_android/src/paddlelite_android.cpp)上


### 2.3 编译项目

我实在Linux x86设备上通过NDK交叉编译Android C++的。

```bash
cmake -DTARGET_OS:STRING="android" -DTARGET_FRAMEWROK:STRING="paddlelite" -DCMAKE_TOOLCHAIN_FILE="/root/android_sdk/ndk/25.0.8775105/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-29 -G "Ninja" ..
cmake --build . --target aiot_benchmark
```

### 2.4 测试

```bash
# opencl
adb -s 3a9c4f5 shell 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/local/tmp/hcp/libs" && /data/local/tmp/hcp/aiot_benchmark --graph="/data/local/tmp/hcp/MobileNetV1" --graph_is_dir=true --nums_warmnup=10 --nums_run=15 --num_threads=1 --cpu_power_mode=0 --backend=opencl --input_shape=1,3,224,224'


# arm cpu

adb -s 3a9c4f5 shell 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/local/tmp/hcp/libs" && /data/local/tmp/hcp/aiot_benchmark --graph="/data/local/tmp/hcp/MobileNetV1" --graph_is_dir=true --nums_warmnup=10 --nums_run=15 --num_threads=1 --cpu_power_mode=0 --backend=arm --input_shape=1,3,224,224'

```

```text
# opencl
Input name[0]: inputs
round: 0, lat: 934.000
round: 1, lat: 30.000
round: 2, lat: 36.000
round: 3, lat: 33.000
round: 4, lat: 42.000
round: 5, lat: 44.000
round: 6, lat: 40.000
round: 7, lat: 42.000
round: 8, lat: 44.000
round: 9, lat: 44.000
round: 10, lat: 44.000
round: 11, lat: 49.000
round: 12, lat: 46.000
round: 13, lat: 44.000
round: 14, lat: 60.000
Output name[0]: softmax_0.tmp_0/target_tran

# arm cpu
Input name[0]: inputs
round: 0, lat: 78.000
round: 1, lat: 62.000
round: 2, lat: 62.000
round: 3, lat: 62.000
round: 4, lat: 62.000
round: 5, lat: 62.000
round: 6, lat: 62.000
round: 7, lat: 62.000
round: 8, lat: 62.000
round: 9, lat: 62.000
round: 10, lat: 62.000
round: 11, lat: 62.000
round: 12, lat: 64.000
round: 13, lat: 62.000
round: 14, lat: 62.000
Output name[0]: save_infer_model/scale_0.tmp_1

```
## 参考资料

1. PaddleLite开源代码地址: https://github.com/PaddlePaddle/Paddle-Lite

2. PaddleLite API文档地址: https://paddlelite.paddlepaddle.org.cn/v2.10/api_reference/cxx_api_doc.html

3. PaddleLite 文档地址: https://paddlepaddle.github.io/Paddle-Lite/
