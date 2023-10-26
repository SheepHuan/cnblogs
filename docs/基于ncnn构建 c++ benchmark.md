
## 1 编译ncnn

### 1.1 依赖

1. [ncnn@20230517](https://github.com/Tencent/ncnn/tree/20230517)
2. [protobuf@3.20.x](https://github.com/protocolbuffers/protobuf/tree/3.20.x)

ncnn这个版本依赖3.20.x的protobuf库，我们自己编译然后安装到系统库里面。protobuf给了3.20.x版本的[编译文档](https://github.com/protocolbuffers/protobuf/blob/3.20.x/src/README.md#c-installation---unix)。如果protobuf版本不匹配的话，在ncnn/tools文件夹下的一些工具将不会编译出来

```bash
git clone https://github.com/protocolbuffers/protobuf/
cd protobuf
git checkout 3.20.x
git submodule update --init --recursive
./autogen.sh
./configure
make -j16 # $(nproc) ensures it uses all cores for compilation
make check
make install
ldconfig # refresh shared library cache.
```

### 1.2 编译libncnn

#### 1.2.1 android

我们按照ncnn的编译指南, [build for android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)，走完命令就行

```bash
# 1. 设置NDK环境变量
export ANDROID_NDK=/root/android_sdk/ndk/25.0.8775105
cd /root/workspace/UnifiedModelBenchmark/3rd-party/ncnn
mkdir -p build-android-aarch64-vulkan
cd build-android-aarch64-vulkan

# 2. 编译ncnn library
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-29 \
    -DNCNN_VULKAN=ON \
    -DNCNN_SHARED_LIB=ON ..

make -j16

# 3. 按照到install/文件夹下
make install

```

运行完指令后,build-android-aarch64-vulkan/install文件夹下由以下文件

```text
.
|-- include
|   `-- ncnn
|       |-- allocator.h
|       |-- benchmark.h
|       |-- blob.h
|       |-- c_api.h
|       |-- command.h
|       |-- cpu.h
|       |-- datareader.h
|       |-- gpu.h
|       |-- layer.h
|       |-- layer_shader_type.h
|       |-- layer_shader_type_enum.h
|       |-- layer_type.h
|       |-- layer_type_enum.h
|       |-- mat.h
|       |-- modelbin.h
|       |-- ncnn_export.h
|       |-- net.h
|       |-- option.h
|       |-- paramdict.h
|       |-- pipeline.h
|       |-- pipelinecache.h
|       |-- platform.h
|       |-- simpleocv.h
|       |-- simpleomp.h
|       |-- simplestl.h
|       `-- vulkan_header_fix.h
`-- lib
    |-- libncnn.so
```

#### 1.2.2 linux amd64

```bash
# 1. 设置NDK环境变量

cd /root/workspace/UnifiedModelBenchmark/3rd-party/ncnn
mkdir -p build
cd build

# 2. 编译ncnn library
cmake ..
make -j16

# 3. 按照到install/文件夹下
make install
```


### 1.3 onnx2ncnn工具

NCNN的cmake文件中发现编译android库时，不会编译tools工具，所以我们应该再走一次1.2.2步骤，编译得到amd64架构下的onnx2ncnn的可执行文件。

```cmake
...
if(ANDROID OR IOS OR NCNN_SIMPLESTL OR CMAKE_CROSSCOMPILING)
    option(NCNN_DISABLE_RTTI "disable rtti" ON)
    option(NCNN_BUILD_TOOLS "build tools" OFF)
    option(NCNN_BUILD_EXAMPLES "build examples" OFF)
else()
    option(NCNN_DISABLE_RTTI "disable rtti" OFF)
    option(NCNN_BUILD_TOOLS "build tools" ON)
    option(NCNN_BUILD_EXAMPLES "build examples" ON)
endif()
...

```

搞完1.2.2的指令后，我们可以在`build/tools/onnx`文件夹下可执行文件

``` text
onnx
|-- CMakeFiles
|-- Makefile
|-- cmake_install.cmake
|-- onnx.pb.cc
|-- onnx.pb.h
`-- onnx2ncnn
```

我们可以将`*.onnx`文件转换为ncnn的`*.param`,`*.bin`的模型文件。

1. onnxsim简化模型

    ```bash
    pip3 install -U pip && pip3 install onnxsim
    onnxsim input_onnx_model output_onnx_model
    ```

2. onnx2ncnn转化模型

    ```bash
    ./onnx2ncnn output_onnx_model out_ncnn.param out_ncnn.bin
    ```


## 2 搭建ncnn benchmark

### 2.1 代码思路

follow [官方wiki](https://github.com/Tencent/ncnn/wiki#input-data-and-extract-output)

1. 初始化`ncnn::Net`对象，并设置`ncnn::Option`来配置运行设置。
2. 让`ncnn::Net`对象先后`load_param`和`load_model`.
3. 为`ncnn::Mat`对象构造`ncnn::Extractor`对象
4. 为`ncnn::Extractor`构造`ncnn::Mat`作为输入输出的载体。
5. 调用`ncnn::Extractor::input(name_in,mat_in)`和`ncnn::Extractor::extract(name_out,mat_out)`完成一次推理.

### 2.2 处理多输入多输出

处理多输入多输出的问题，我是通过定义`key:value`的形式，由外部命令行传入模型的输入输出name及其`NCHW`格式的shape信息，然后在代码中解析出来，循环调用`ncnn::Extractor::input(name_in,mat_in)`和`ncnn::Extractor::extract(name_out,mat_out)`完成多变量输入和多变量输出

```C++
  for (int i = 0; i < nums_warmup; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        // 输入多个输入
        timer.start();
        for (int i = 0; i < input_names.size(); i++)
        {
            std::vector<int> shape = input_info[input_names[i]];
            ncnn::Mat in = ncnn::Mat(shape[3], shape[2], 1, shape[1]);
            in.fill(0.5f);
            ex.input(input_names[i], in);
        }
        for (int i = 0; i < output_names.size(); i++)
        {
            ncnn::Mat out;
            ex.extract(output_names[i], out);
        }
        timer.end();
        warmup_time = warmup_time + timer.get_time();
    }
```
