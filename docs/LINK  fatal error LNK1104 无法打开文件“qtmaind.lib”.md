# LINK : fatal error LNK1104: 无法打开文件“qtmaind.lib”

### VS2019+QT5.13出现该问题，可以尝试这样解决。

1.找打qtmaind.lib所在的绝对路径

![image.png](https://i.loli.net/2019/11/25/xMqPyzQRtkJa4Fo.png)



2.修改vs项目属性: 项目->项目属性->配置属性->链接器->输入->附加依赖项

![image.png](https://i.loli.net/2019/11/25/pFsBRxN3ivV9O7G.png)

![image.png](https://i.loli.net/2019/11/25/JRcLw42e5EbSy7k.png)

将 qtmaind.lib修改为绝对路径 :C:\Qt\Qt5.13.2\5.13.2\msvc2017_64\lib\qtmaind.lib

![image.png](https://i.loli.net/2019/11/25/xpsPrXtJ6EBMq3C.png)



成功运行demo了

![image.png](https://i.loli.net/2019/11/25/ogcNnKYliSWIL2v.png)