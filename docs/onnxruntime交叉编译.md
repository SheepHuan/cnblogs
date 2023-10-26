
onnxruntime官方文档没有及时更新，有些地方和实际有些出入。这里记录一下onnxruntime-v1.16.0版本的编译指令。

## 1 linux x64交叉编译linux aarch64

### 1.1 准备要求

官方的给的的cmake编译文档是：Build for inferencing | onnxruntime。

它在这里给了几点要求：

1. corresponding toolchain，linux这里指的是aarch64-linux-gnu-g++这个软件。可以在ubuntu上直接用apt安装了arm提供的交叉编译的工具链，可以用这个条命令实现。

    ```bash
    sudo apt install g++-9-aarch64-linux-gnu
    ```

    也可以参考本文1.3去下载各个arm开发板官方的提供的工具链，定义一些环境变量就能跑起来。
2. pre-compiled protoc，protoc这个软件的版本要和onnxruntime依赖的版本一致，具体的版本和下载链接可以到onnxruntime/cmake/deps.txt at main · microsoft/onnxruntime (github.com)找到。我们的host是x64的Linux，可以选择下载protoc_linux_x64版本的protoc。
3. (optinal) Setup sysroot to enable python extension.Skip if not using Python。我不需要这个就没编译了。
4. Generate CMake toolchain file，我们需要写一个toolchain.cmake来保存编译参数。官方给的缺少对于cpu processor信息的参数。这会导致编译报错。

    ```bash
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_SYSTEM_VERSION 1)
    SET(CMAKE_C_COMPILER /bin/aarch64-linux-gnu-gcc-9)
    SET(CMAKE_CXX_COMPILER /bin/aarch64-linux-gnu-g++-9)
    SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    SET(CMAKE_SYSTEM_PROCESSOR aarch64)
    ```

### 1.2 编译指令

```bash
cd  /tmp/onnxruntime@v1.16.0

cmake -S cmake/ \
-B build/ \
-Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-Donnxruntime_BUILD_SHARED_LIB=ON \
-DCMAKE_TOOLCHAIN_FILE=/tmp/toolchain.cmake \
-DONNX_CUSTOM_PROTOC_EXECUTABLE=/tmp/protoc/bin/protoc \
-DCMAKE_INSTALL_PREFIX=/tmp/onnxruntime@v1.16.0/build/install

cmake --build build/

cmake --install build/

```

编译结束后就可以在build/install目录下看到编译好的onnxruntime.so和头文件了。

![1696565426009.jpg](http://pic.yanghuan.site/i/2023/10/06/651f885e475b4.jpg)
<center>图1.2-1 编译结果</center>

### 1.3 自定义交叉编译工具链

我这里根据地平线的官方中提供交叉工具链进行编译，踩了坑。

#### 1.3.1 参考文档和坑

1. [地平线RDK套件-编译#交叉编译开发环境](https://developer.horizon.cc/documents_rdk/linux_development/environment_build?_highlight=%E4%BA%A4%E5%8F%89&_highlight=%E7%BC%96%E8%AF%91#%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83)
2. [地平线社区-交叉编译异常](https://developer.horizon.cc/forumDetail/183278759317820492)

坑：

1. 地平线提供的ubutnu20.04的系统内置的gcc/g++版本是9.3.0的，而Ubuntu20.04提供的gcc交叉编译工具链的版本是9.4.0,两个版本的编译器依赖的glibc/glibstdc++库的小版本不一致，运行时会依赖报错。
    ![1696571910813.png](![1696572071608.jpg](http://pic.yanghuan.site/i/2023/10/06/651fa2aa90a55.jpg)
    <center>图1.3-1 地平线X3 GCC版本</center>
2. 地平线的交叉工具链中，`aarch64-linux-gnu-gcc`和`aarch64-linux-gnu-g++`并不是实际的二进制文件，而是一个shell脚本，负责将重新调用它同目录下的`aarch64-linux-gnu-gcc-rel`和`aarch64-linux-gnu-g++-rel`。在交叉编译时直接用`aarch64-linux-gnu-g++`，onnxruntime的会显示该编译器不可用。

#### 1.3.2 解决步骤

1. 下载地平线交叉编译工具，并解压，并设置环境变量

    ```bash
    # 下载
    curl -fO http://archive.sunrisepi.tech/toolchain/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu.tar.xz

    # 解压
    tar -xf gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu.tar.xz
    # 设置环境变量
    export PATH=$PATH:/root/tools/lib/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin
    export LD_LIBRARY_PATH=/root/tools/lib/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/lib/x86_64-linux-gnu:/root/tools/lib/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/aarch64-linux-gnu/lib:$LD_LIBRARY_PATH
    ```

2. 写一个toolchain.cmake工具链

    ```Cmake
    SET(TOOLCHAIN_PATH /root/tools/lib/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu)
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_SYSTEM_VERSION 1)
    SET(CMAKE_C_COMPILER "${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-gcc-rel")
    SET(CMAKE_CXX_COMPILER "${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-g++-rel")
    set(CMAKE_CXX_FLAGS "-mfix-cortex-a53-835769 ${CMAKE_CXX_FLAGS}")
    SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    SET(CMAKE_SYSTEM_PROCESSOR aarch64)
    ```

注意，我这里设置了`CMAKE_CXX_FLAGS`，因为地平线原来提供的交叉编译的`aarch64-linux-gnu-g++`里面调用`aarch64-linux-gnu-g++-rel`加了一个`-mfix-cortex-a53-835769`的参数

3. protoc的下载安装1.2里面的步骤做就行。

4. cmake指令

    ```bash
    # 编译
    cmake -S cmake/ \
    -B build/ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON \
    -DCMAKE_TOOLCHAIN_FILE=/root/code/aarch_linux_cross_toolchain.cmake \
    -DONNX_CUSTOM_PROTOC_EXECUTABLE=/root/tools/lib/protoc/bin/protoc
    -DCMAKE_INSTALL_PREFIX=build/install
    # 构建
    cmake --build build/
    # 安装
    cmake --install build/
    ```

## 2 linux x64编译

直接按照官方文档的指导，运行build.sh。

```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

## 3 常见平台交叉编译

平台        | 操作系统版本 | GCC版本 | Toolchain下载
--------    | ------------ | ----- | -----  |
NVIDIA Jetson Nano  | Ubuntu 18.04 | 7.5.0 | [releases.linaro.org](https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/aarch64-linux-gnu/)
地平线 旭日X3       | Ubuntu 20.04 | 9.3.0 | [社区文档](https://developer.horizon.cc/documents_rdk/linux_development/environment_build#%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83)
