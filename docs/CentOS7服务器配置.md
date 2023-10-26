# CentOS7服务器配置

## 1.更换yum软件源

### 下载阿里源

```
cd /etc/yum.repos.d
sudo wget -nc http://mirrors.aliyun.com/repo/Centos-7.repo
```

###  更改阿里yum源为默认源

Centos-7.repo可能要看具体服务器的文件名

```
sudo mv Centos-7.repo CentOS-Base.repo
```

###  更新本地yum缓存

```
# 全部清除
sudo yum clean all
# 更新列表
sudo yum list
# 缓存yum包信息到本机，提高搜索速度
sudo yum makecache
```

## 2.安装Anaconda3

[清华镜像源]( https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ )下载

### 2.1更换conda源

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

![image.png](https://i.loli.net/2019/11/30/6SEnhYf8NMTzrKC.png)

### 2.2更换pip源

 https://blog.csdn.net/dapanbest/article/details/81096241 