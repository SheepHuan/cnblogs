# Linux C网络编程

## 1.Linux套接字

### 1.1 套接字介绍

套接字(Sockets)，即为网络进程ID，是由运行这个进程的计算机的IP地址和这个进程使用的端口(Port)组成.

可以只用'netstat-all' 查看当前系统中网络应用进程的套接字和端口. 可以使用 > 输出重定向到文件.

### 1.2 套接字的结构

Linux在头文件<sys/socket.h>中定义了通用的套接字结构类型,可供不同协议调用

```c
struct sockaddr
{
	unsigned short int sa_family;	//表示套接字的协议类型,如常见的IPv4,IPv6
    unsigned char sa_data[14];		//14个字节的协议地址,包含了IP地址和端口
}
```

除了sockaddr之外,Linux还在<netinet/in.h>中定义了另外一种结构类型 sockaddr_in ,它和sockaddr等效且可以互相转换.通常用于TCP/IP协议

```c
struct sockaddr_in
{
	int sa_len;    					//长度单位,通常使用默认值16
    short int sa_family;			//协议族
    unsigned short int sin_port;	//端口号
    struct in_addr sin_addr;		//IP地址
    unsigned char sin_zero[8];		//填充0,保证与struct sockaddr同样大小
}
```

对其中struct in_addr sin_addr说明如下

```c
struct sin_addr
{
	in_addr_t s_addr; //32 IPv4地址
}
```

**常见协议对应的sa_famliy值**

| 可选值   | 说明         |
| :------- | ------------ |
| AF_INET  | IPv4协议     |
| AF_INET6 | IPv6协议     |
| AF_LOCAL | UNIX协议     |
| AF_LINK  | 链路地址协议 |
| AF_KEY   | 密钥套接字   |



## 2.Linux网络操作函数

### 2.1 字节顺序转换函数族

往往网络上的不同机器,数据存储模式不同,小型机通常为小端模式,大型机为大端模式.所以往往需要字节转换.

Linux 提供了htonl ,htons,ntohl ,ntohs函数处理大端和小端模式转换

解释:**htonl/htons:host to network long/short**  ;同理 **ntohl/ntohs:network to host long/short**

```c
#include <arpa/inet.h>
unit32_t htonl(unit32_t hostlong);
unit16_t htons(unit16_t hostshort);
unit32_t ntohl(unit32_t netlong);
unit16_t ntohs(unit16_t netshort);
```

32为long数据通常存放IP地址,16为通常存放端口号

htonl:将32为PC机数据(小端)转为32位网络传输数据(大端)

### 2.2 字节操作函数族

套接字与字符串不同,套接字是多字节数据而不是以空字符串结尾.Linux提供了若干函数,在内存上直接操作套接字

#### 2.2.1

第一组函数是与BSD系统兼容的函数,包括bzero,bcopy,bcmp.

bzero:将参数s指定的前n个字节设置为0,通常用来对套接字地址清零

```c
#include <strings.h>
void bzero(void *s,size_t n);
```

bcopy:从参数src指定的内存区域,拷贝指定数目字节内容到dest指定内存区域

```c
#include <strings.h>
void bcopy(const void* s1,void *dest,size_t n);
```

bcmp:用于比较是参数s1和参数s2指定内存区域的前n字节.如果相同返回0,否则返回非0

```c
#include <strings.h>
int bcmp(const void *s1,const void *s2,size_t n);
```

2.2.2

### 2.3 IP地址转换函数族

IP地址通常是点分十进制表示,Linux网络编程中会使用32二进制值.Linux提供了若干函数保证二者相互转换

### 2.4 域名转换函数族

实际网络编程中往往会遇到www.baidu.com这样的域名Linux提供了函数让域名转为IP地址和让IP地址转域名

## 3. 套接字编程

### 3.1 创建套接字描述符函数

Liunx使用socket函数创建套接字描述符

```c
#include <sys/type.h>
#include <sys/socket.h>
int socket(int domain,int type,int protocol);
```

函数调用成功,则返回套接字描述符(正整数),否则返回-1

参数说明

- domain: 套接字协议族,支持类型如下

  | 协议族名称       | 描述         |
  | ---------------- | ------------ |
  | AF_UNIX,AF_LOCAL | 本地交互协议 |
  | AF_INET          | IPv4协议     |
  | AF_INET6         | IPv6协议     |
  | ...              | ...          |

  

- type:指定当前套接字类型

  | 类型名称       | 描述         |
  | -------------- | ------------ |
  | SOCK_STREAM    | 数据流       |
  | SOCK_DGRAM     | 数据报       |
  | SOCK_SEQPACKET | 顺序数据报   |
  | SOCK_RAW       | 原始套接字   |
  | SOCK_RDM       | 可靠传递消息 |
  | SOCK_PACKET    | 数据包       |

  

- protoclo:通常情况下设置为0,表示使用默认协议



### 3.2 绑定套接字函数

在创建套接字后需要将本地地址和套接字绑定,可以调用bind函数

```c
#include <sys/type.h>
#include <sys/socket.h>
int bind(int sockfd,const struct sockaddr *addr,socklen_t addrlen);
```

sockfd是创建套接字时对于的套接字描述符.addr是本地地址.addrlen是套接字对应的地址结构长度;

bind函数执行成功返回0,否则返回-1

bind函数绑定模式有5种:

...等待更新



### 3.3 建立连接函数

使用socket函数建立套接字并绑定地址后,即可使用connect函数和服务器建立连接

```c
#include <sys/type.h>
#include <sys/socket.h>
int connect(int sockfd,const struct sockaddr *addr,socken_t addrlen);
```

参数sockfd是套接字创立后函数socket返回的套接字描述符;

参数addr指定远程服务器的套接字地址,包括服务器地址和端口号;

参数addrlen 指定套接字地址长度;

调用connect函数成功后,返回0,否则为-1

### 3.4 倾听套接字切换函数

socket函数直接创立的是主动套接字,用来发送请求的.如果是服务器需要倾听套接字,接受请求.使用listen函数将套接字转换为倾听套接字

```c
#include <sys/types.h>
#include <sys/socket.h>
int listen(int sockfd,int backlog);
```

参数sockfd使套接字描述符；backlog是请求队列的最大长度

baclog的作用：

...待更新

### 3.5接受连接函数

当服务器接收到一个连接后，可以使用函数accept从倾听套接字的完成连接队列中接受一个连接。如果完成连接队列为空，则进程进入睡眠。

```c
#include <sys/types.h>
#include <sys/socket.h>
int accept(int sockfd,struct sockaddr *addr ,socklen_t *addrlen);
```

### 3.6 关闭连接函数

```c
#include <unistd.h>
int close(int fd);
```

### 3.7 套接字读写函数

```c
int read(int fd,char *buf,int len);
int write(int fd,char *buf,int len);
```

### 3.8 套接字地址获取函数

```c
#include <sys/socket.h>
int getsockname(int sockfd,struct sockaddr *addr,socklen_t *addrlen);
int getpeername(int sockfd,struct sockaddr *addr,socklen_t *addrlen);
```

### 3.9 发送和接受函数

```c
#include <sys/type.h>
#include <sys/socket.h>
ssize_t send(int socketfd,const void *buf,size_t len,int flags);
ssize_t recv(int socketfd,const void *buf,size_t len,int flags);
```



## 4. Linux TCP编程

### 4.1 TCP工作流程

- 服务器先用socket函数建立套接口,用这个套接口完成通信的监听和数据收发
- 服务器利用bind函数绑定一个IP地址和端口号,使套接口与指定端口号,IP地址相关联
- 服务器调用listen函数,使服务器和这个端口和IP处于监听状态,等待网络中某一客户的连接请求
- 客户机用socket函数建立套接口,设定远程IP和端口.
- 客户机调用connect函数连接远程计算机的指定的端口
- 服务器调用accept函数接受远程计算机的连接请求,建立起与客户机之间的通信连接
- 连接连接后,客户机里用write函数或者send函数乡socket中写入数据,也可以使用read函数或recv函数读取服务器发送来的数据
- 服务器利用read函数或者recv函数读取客户机发送来的数据,也可以使用write函数或者send函数来发送数据
- 完成通信后,使用close函数关闭socket连接

## 5. Linux UDP编程

待更新...