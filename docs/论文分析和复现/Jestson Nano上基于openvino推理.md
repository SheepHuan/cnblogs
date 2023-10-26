
intel NCS2计算棒是由openvino支持的，但是目前openvino只有2022.3.1版本支持NCS2了，之后的版本都不支持计算棒了。

本文记录一下再NVIDIA Jetson Nano上用openvino实现NCS2的调用。
通过交叉编译openvino samples，通过benchmark app实现模型推理。
## 环境配置

1. openvino归档文件，可以再这个[链接](https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.1/linux)上下载

    ![1697453851285.png](http://pic.yanghuan.site/i/2023/10/16/652d171d42c40.png)

    然后解压
    ```bash
    wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.1/linux/l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64.tgz

    tar -zxvf l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64.tgz
    ```

2. 安装openvino依赖

    ```bash
    cd l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/install_dependencies
    sudo -E ./install_openvino_dependencies.sh
    ```
3. 确保一个python 3.7或者3.9版本
4. 确保cmake版本高于3.13，否则会报错。Jetson Nano上的cmake只有3.10，我手动安装了一个

## 编译openvino samples


直接进入`samples/cpp`，直接在板子上编译太慢了，幸好项目不大。
```bash
export PATH=/home/yanghuan/Downloads/cmake-3.28.0-rc1-linux-aarch64/bin:$PATH
export LD_LIBRARY_PATH=/home/yanghuan/Downloads/l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/runtime/lib/aarch64:$LD_LIBRARY_PATH
cd l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/samples/cpp
./build_samples.sh -i bin/ -b build/
```
这样在`bin/`目录下就是可执行文件了。

![1697457565868.png](http://pic.yanghuan.site/i/2023/10/16/652d259edc921.png)

## 配置NCS2
这里参考[官网文档](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_configurations_for_ncs2.html)

最后重启插上NCS2。

```bash
export LD_LIBRARY_PATH=/home/yanghuan/Downloads/l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/runtime/lib/aarch64:$LD_LIBRARY_PATH
cd ~/Downloads/l_openvino_toolkit_debian9_2022.3.1.9227.cf2c7da5689_arm64/samples/cpp/bin/samples_bin
# 查询设备
 ./hello_query_device
```

![1697458010558.png](http://pic.yanghuan.site/i/2023/10/16/652d275ba5435.png)



## 测试模型延迟

这里提交一个mobilenetv2的模型试试
1. NCS2推理

    ```bash
    ./benchmark_app \
    -m /home/yanghuan/Downloads/mobilenet_v2.onnx \
    -d MYRIAD \
    -b 1 \
    -shape [1,3,256,256] \
    -hint latency \
    -niter 10
    ```
    ![1697458463327.png](http://pic.yanghuan.site/i/2023/10/16/652d29218bb9c.png)

2. CPU推理
    ```bash
    ./benchmark_app \
    -m /home/yanghuan/Downloads/mobilenet_v2.onnx \
    -d CPU \
    -b 1 \
    -shape [1,3,256,256] \
    -hint latency \
    -niter 10
    -nthreads 2
    ```
    ![1697458561178.png](http://pic.yanghuan.site/i/2023/10/16/652d298223d79.png)
