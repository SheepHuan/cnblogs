# CUDA安装

## 准备工作

- [CUDA下载地址](https://developer.nvidia.com/cuda-toolkit-archive)
- [cdnn下载地址](https://developer.nvidia.com/rdp/cudnn-download)(需要登陆英伟达账户,可以用谷歌账户登录,记得选择和CUDA版本对应的)

1. 查看电脑GPU驱动版本,对应官网表格选择对于版本CUDA下载. [点击查看](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

   <img src="https://i.loli.net/2020/11/01/SjHkvelgE8OBcaX.png" alt="image.png" style="zoom: 67%;" />

   ![image.png](https://i.loli.net/2020/11/01/qYdj5TZFCbIVmtn.png)

2. 查看其它软件平台对应CUDA版本需求 (待更新)

   - OpenCV

   - Tensorflow

   - Visual Stdio ([点击查看](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) CUDA11.1对Visual Stdio版本安装指引)

     点击[CUDA下载页面](https://developer.nvidia.com/cuda-toolkit-archive)对应版本的在线文档,查找对VS的设置指引

## 安装

1. CUDA程序的解压目录可以选择一个临时目录(解压完会自动删除)

2. 全选(默认),**不要更改CUDA默认安装目录**(否则添加环境变量很麻烦)

3. 将cdnn\cuda下的文件夹 include,bin,lib\x64下的文件复制到CUDA安装目录下的include,bin,lib\x64中.

4. 为CUDA添加环境变量,添加到系统变量的Path中

   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0

5. 测试安装结果

   - 打开命令行(CMD)

     ```
     cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\demo_suite
     ```

     ![image.png](https://i.loli.net/2020/11/01/4RCihdY5p7lQ8L3.png)

   - 分别执行bandwidthTest.exe和deviceQuery.exe

     ```
     bandwidthTest
     deviceQuery
     ```

     ![image.png](https://i.loli.net/2020/11/01/qvEbMFpOI4eiXD3.png)

     ![image.png](https://i.loli.net/2020/11/01/LBECehU5xFT3Wmq.png)

     

   - 检查输出结果输出结果(最后几行位置)都应该有Result=PASS

     如上图





**暂结束!**