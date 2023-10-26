## 创建 Docker 容器

```bash
# 1. 要求映射usb文件夹到docker /dev/bus/usb:/dev/bus/usb
docker run -it --privileged=true -v /dev/bus/usb:/dev/bus/usb -v /home/yanghuan/workspace/aiot_benchmark:/root/workspace --net=host --name=aiot_benchmark --hostname=aiot_benchmark ubuntu:20.04 /bin/bash

# 2. 安装android udev规则,此时 cat /etc/group 确保有plugdev用户组
apt-get install android-sdk-platform-tools-common
```

![1681447491778.png](http://pic.yanghuan.site/i/2023/04/14/6438da456802f.png)

```bash
# 3.将当前用户加入plugindev用户组
usermod-a -G plugdev root
# 4. 关闭的退出容器
exit
# 5. 宿主机上重新登录docker
docker start aiot_benchmark
docker attach aiot_benchmark
# 6. 确保已加入用户组，并检查连接情况
id
lsusb
# 没有lsusb的，apt install usbutils
```

![1681448105689.png](http://pic.yanghuan.site/i/2023/04/14/6438dcaba72ec.png)

## 安装 Android SDK

### 安装 sdkmanager

参考[sdkmanager](https://developer.android.google.cn/studio/command-line/sdkmanager?hl=zh-cn)文档中描述

1. 从 Android Studio 下载页面中下载最新的“command line tools only”软件包，然后将其解压缩。

2. 将解压缩的 cmdline-tools 目录移至您选择的新目录，例如 android_sdk。这个新目录就是您的 Android SDK 目录。

3. 在解压缩的 cmdline-tools 目录中，创建一个名为 latest 的子目录。

4. 将原始 cmdline-tools 目录内容（包括 lib 目录、bin 目录、NOTICE.txt 文件和 source.properties 文件）移动到新创建的 latest 目录中。现在，您就可以从这个位置使用命令行工具了。
5. 加入环境变量
    ```bash
    echo "export PATH=$PATH:/root/android_sdk/cmdline-tools/latest/bin" >> /root/.bashrc
    source ~/.bashrc
    ```

6. 安装 open jdk 11
    ```bash
    apt-get install openjdk-11-jdk -y
    ```

7. 安装 platfrom-tools 和其他包
     ```bash
    # 设置代理
    export https_proxy="http://172.16.101.180:7890"
    sdkmanager --install  "platform-tools" "platforms;android-29" "ndk;25.0.8775105"
    ```

![1681449532864.png](http://pic.yanghuan.site/i/2023/04/14/6438e23ec8548.png)

8. 设置 adb 环境变量
   ```bash
    echo "export PATH=$PATH:/root/android_sdk/platform-tools" >> /root/.bashrc
    source ~/.bashrc
   ```

9.  连接设备（确保没有其他的 adb 服务了）
   ![1681450096396.png](http://pic.yanghuan.site/i/2023/04/14/6438e471c4478.png)
