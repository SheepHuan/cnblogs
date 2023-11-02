## 1. Linux 内存压缩 zRAM

### 1.1 基本介绍

ZRAM 的原理是划分一块内存区域作为虚拟的块设备（可以理解为支持透明压缩的内存文件系统），当系统内存不足出现页面交换时，可以将原本应该交换出去的页压缩后放在内存中，由于部分被『交换出去』的页得到了压缩，因此可用的物理内存就能随之变多。

### 1.2 使用方法

```bash

docker run -it  --privileged --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --name zram-test --hostname zram nvcr.io/nvidia/l4t-jetpack:r35.4.1 /bin/bash

```

1. 安装zram配置工具

```bash
apt install zram-config
```

2. 修改`/usr/bin/init-zram-swapping`

3. 设置开启自启动,创建`/etc/systemd/system/zram.service`，写入一下内容

```text
Description=Swap with zram
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=true
ExecStart=/usr/bin/init-zram-swapping

[Install]
WantedBy=multi-user.target
```


### 1.3 效果

1. 测试不同压缩算法的效率（压缩时间、压缩比率）


## 参考资料

1. [五万字 | 深入理解Linux内存管理](https://zhuanlan.zhihu.com/p/550400712)
2. [Linux内核：内存管理——ZRAM](https://zhuanlan.zhihu.com/p/631300401)
3. [配置 ZRAM，实现 Linux 下的内存压缩，零成本低开销获得成倍内存扩增](https://zhuanlan.zhihu.com/p/484408336)
4. [在Linux云服务器上开启zram](https://a-jiua.github.io/blog/2023/08/22/%E5%9C%A8Linux%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E5%BC%80%E5%90%AFzrazm/)
