# Linux 多显卡切换

参考：https://blog.csdn.net/renhanchi/article/details/80664546

```bash
#查看当前所有显卡和BusID
lspci | egrep -i 'VGA|3D'
#注意这里的Bus-ID，如02:00.0/82:00.0是16进制的02和130
```

![image-20210807135658989](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/imgsimage-20210807135658989.png)

```bash
#填写/etc/X11/xorg.conf文件
#/etc/X11       图形界面配置文件，BUSID写成10进制
sudo vim /etc/X11/xorg.conf
#这里将Section "Device"中的BusID、Driver、VendorName 修改为对应的显卡的BusID、驱动名称和供应商名称
```

![image-20210807135904932](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/imgsimage-20210807135904932.png)

```bash
#查询GPU的driver 和 vendor id
lspci -v -s 06:00.0
#这行命令可以根据BusID查询到对应的Driver Name和Vendor Name
```

![image-20210807140343614](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/imgsimage-20210807140343614.png)

![image-20210807140457146](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/imgsimage-20210807140457146.png)