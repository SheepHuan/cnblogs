## 1 复现PaddleLite v2.11 demo

为什么复现2.11的版本呢？因为2.12的复现不出来.

### 2.1 准备

1. 设备: 九鼎创展 x1808开发板. 默认ssh账户root,密码123456
   - 刷机要点: 按住K1键，然后板子上电，进入LOADER模式，使用瑞芯微提供的AndroidTools升级固件即可。
2. 软件:
   - `uname -a` : Linux root 4.4.194 #7 SMP PREEMPT Fri Dec 11 12:06:00 CST 2020 aarch64 GNU/Linux
   - `dmesg | grep Galcore | grep version` :  [   21.137075] Galcore version 6.4.0.227915
3. 可执行文件:
   - [PaddleLite v2.11 demo](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo_v2_11_0.tar.gz)
   - [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)

### 2.2 实现

1. 解压tar.gz
   ```bash
    tar -zxvf PaddleLite-generic-demo_v2_11_0.tar.gz
    tar -zxvf mobilenet_v1_int8_224_per_layer.tar.gz
   ```
2. 将 `mobilenet_v1_int8_224_per_layer/`放入`PaddleLite-generic-demo/image_classification_demo/assets/models`
3. 由于磁盘空间的问题，我在开发板上挂载了一个tf card.
    ```bash
    # 查看磁盘分区
    fdisk -l
    # 对指定分区进行格式化
    mkfs .ext4 /dev/mmcblk1p1
    # 创建挂载目录
    mkdir -p /mnt/tfcard
    # 挂载磁盘
    mount /dev/mmcblk1p1 /mnt/tfcard
    ```
4. 修改`PaddleLite-generic-demo/image_classification_demo/shell`中的`run_with_ssh.sh`中的工作目录
    ```bash
    ...
    # line 25
    WORK_SPACE="/mnt/tfcard/test"
    ...
    ```
5. 执行脚本命令
   - NPU
        ```bash
        # ./run_with_ssh.sh <model dir> <OS> <arch> <device> <ip> <port> <username> <password>
        # 调用NPU执行
        ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 rockchip_npu 172.16.6.98 22 root 123456
        ```
        结果
        ```text
        ...
        iter 4 cost: 6.769000 ms
        warmup: 1 repeat: 5, average: 6.729200 ms, max: 6.775000 ms, min: 6.679000 ms
        results: 3
        Top0  Egyptian cat - 0.514779
        Top1  tabby, tabby cat - 0.421183
        Top2  tiger cat - 0.052648
        Preprocess time: 2.457000 ms
        Prediction time: 6.729200 ms
        Postprocess time: 0.450000 ms
        ...
        ```
    - CPU
        ```bash
        ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 172.16.6.98 22 root 123456
        ```
        结果
        ```text
        iter 0 cost: 266.263000 ms
        iter 1 cost: 266.100006 ms
        iter 2 cost: 265.746002 ms
        iter 3 cost: 265.558014 ms
        iter 4 cost: 265.686005 ms
        warmup: 1 repeat: 5, average: 265.870605 ms, max: 266.263000 ms, min: 265.558014 ms
        results: 3
        Top0  Egyptian cat - 0.512545
        Top1  tabby, tabby cat - 0.402567
        Top2  tiger cat - 0.067904
        Preprocess time: 2.487000 ms
        Prediction time: 265.870605 ms
        Postprocess time: 0.430000 ms

        ```