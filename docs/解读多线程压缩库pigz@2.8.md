# 解读PIGZ@2.8项目

[pigz](https://github.com/madler/pigz),是一个开源多线程压缩库，是GZip的并行实现。今天粗浅地研究下它的多线程解压的并行实现，并尝试封装出API。

## 1 代码分析

### 1.1 代码分布

```bash
.
|-- CMakeLists.txt # 我自己写的cmake文件
|-- Makefile
|-- README
|-- pigz.1
|-- pigz.c # pigz工具的入口，main函数在这里面。
|-- pigz.pdf
|-- pigz.spec
|-- try.c
|-- try.h
|-- yarn.c
|-- yarn.h
`-- zopfli # 一个压缩算法
    |-- CONTRIBUTING.md
    |-- CONTRIBUTORS
    |-- COPYING
    |-- README
    `-- src
        `-- zopfli
            |-- blocksplitter.c
            |-- blocksplitter.h
            |-- cache.c
            |-- cache.h
            |-- deflate.c
            |-- deflate.h
            |-- hash.c
            |-- hash.h
            |-- katajainen.c
            |-- katajainen.h
            |-- lz77.c
            |-- lz77.h
            |-- squeeze.c
            |-- squeeze.h
            |-- symbols.c
            |-- symbols.h
            |-- tree.c
            |-- tree.h
            |-- util.c
            |-- util.h
            `-- zopfli.h

3 directories, 36 file

```

### 1.2 `pigz.c`的解压缩


1. main函数一开始就是一个`unamed struct`（匿名结构体）`g`变量，用来存全局变量信息。
2. `pigz.c`里面的`local int option(char *arg)`函数基本上就是在解析命令行参数的。在line 4514行

3. 我们可以看到

    ```c
    ...
    case 'd':  if (!g.decode) g.headis >>= 2;  g.decode = 1;  break;
    ...
    ```
    
    我们输入-d选项时，全局变量会将`g.decode`设置为1，表示当前时要解压缩。

4. 如何获取到`-p`的选项的值的？`g.procs`时存储了最大压缩线程。=

5. 在line 3914，处理完options后，进入`local void process(char *path)`函数计算，一开始在处理输入文件名。大约在line 4054开始读取gzip的文件头。

6. 