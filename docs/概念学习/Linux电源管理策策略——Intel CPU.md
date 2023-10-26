针对Intel的CPU电源策略的考虑：
1. P-State,C-State分别意味着什么?
2. 为什么调整CPU的P-State并添加负载，CPU状态会调整回来。
3. 是Linux调整，还是CPU自己调整的，C-State是OS在调整


## 1 基本概念
### 1.1 C-State and P-State
P-state是指处理器状态（processor state），也被称为处理器频率或电压状态。它指的是处理器在不同工作负载下采用的电压和频率的组合，以达到最佳的性能和能耗平衡。

C-state是指CPU状态（CPU state），也被称为睡眠状态。它指的是处理器在空闲或轻负载状态下进入的低功耗状态，以节省能量并减少发热，延长处理器的寿命。C-state的级别越高，功耗越低，但唤醒需要更多的时间和资源。

### 1.2 Linux 功耗管理策略

#### 1.2.1 管理策略

在kernel doc的[Power Management Strategies](https://docs.kernel.org/admin-guide/pm/strategies.html#power-management-strategies)里面,intel的开发人员介绍了当前Linux kernel支持的两类高层的功耗管理策略:

1. 系统范围的功耗管理([System-Wide power management](https://docs.kernel.org/admin-guide/pm/system-wide.html))
2. 工作状态的功耗管理([Working-State Power Management](https://docs.kernel.org/admin-guide/pm/working-state.html))

他举了一个例子，这两个电源策略负责处理不同的工作场景，比如离开笔记本电脑（关闭盖子）时，就会启动System-Wide power management策略控制系统进入睡眠状态，让大部分组件空闲下来，降低系统功耗。而在打开笔记本进行工作时应当用Working-State Power Management来动态管理功耗变换。

#### 1.2.1 System-Wide power management

这个策略主要在管理电脑的睡眠状态,根据[文档](https://docs.kernel.org/admin-guide/pm/sleep-states.html#system-sleep-states)介绍，在目前Linux kernel中支持**4种系统睡眠状态**,包含**休眠**和3种系统挂起的变体。

1. suspend-to-idle,这个模式就是通过冻结用户空间，暂停计时并将所有IO设备置于低功耗状态。
2. standby,除了冻结用户空间，暂停计时并将所有IO设备置于低功耗状态之外，nonboot的cpu将会offline,在转换到这个状态后所有底层的系统函数将会暂停/挂起。因此这个状态会更加解决能耗，但恢复到工作状态的时间就会更久了。
3. suspend-to-ram,这个状态下，设备和CPU的状态都被写入RAM，系统所有的组都会处于低功耗状态,内核的控制权将会在转换到从S2RAM后交回BIOS。
4. hibernation,休眠状态下，内核将还会停止所有系统活动，并创建内存快照存入磁盘。写入磁盘后，系统进入目标低功耗状态或者关闭电源。如果需要唤醒，就需要重新引导内行人，并加载内存快照恢复到休眠前。

#### 1.2.2 Working-State Power Management

这个管理模式下主要分为两个部分`CPU Idle Time Management`和`CPU Performance Scaling`。

##### 1.2.2.1 [CPU Idle Time Management](https://docs.kernel.org/admin-guide/pm/cpuidle.html#cpu-idle-time-management)

这个就是C-State的概念吧  
这里文档中指出了CPU Idle的概念，对于类似英特尔的CPU的超线程技术，一个核心中可能存在多个硬件线程(Logical CPU,或简称CPU)，如果某核心中一个线程被要求进入空闲状态，那么该线程（CPU）就会进入空闲状态，除非同一个核心内的其他硬件线程也要求进入空闲状态，那么整个核心都会空闲，否则整个核心中不会发生其他状况。

[**空闲循环任务**](https://docs.kernel.org/admin-guide/pm/cpuidle.html#the-idle-loop)  
正常的CPU进入空闲状态，就给CPU一个空闲循环任务。空闲循任务的执行需要两个主要步骤：

1. 调用`CPUIdle`子系统去选择一个idle state给CPU进入。
2. 调用`CPUIdle`子系统的一个代码模块（驱动模块），实际请求处理器硬件进入`governor`选择进入的状态。governor的作用是找到最适合当前条件的空闲状态。

[**Tickless**](https://docs.kernel.org/admin-guide/pm/cpuidle.html#idle-cpus-and-tick)
简单来说，CPU会配置一个scheduler tick作为计时器来决定CPU调度的时分共享策略。意味着每个任务都会被分配一段CPU时间来运行代码，时间结束后就会切换热舞，空闲CPU的空闲任务来说这是有问题的。因为计时器的触发时相对频繁的，而计时器频繁地请求目标CPU唤醒并再次进入空闲状态，这会导致CPU的空闲状态不超过tick周期，这样会发生能耗浪费。内核中可以配置tickless参数让CPU空闲时停止tick触发来使得系统更节能。如果系统支持tickless，则使用`menu` governor否则使用`ladder`。

一般有4中可用的`CPUIdle` governors，分别是`menu`,`TEO`,`ladder`和`haltpoll`。

<!-- [The `menu` Governor](https://docs.kernel.org/admin-guide/pm/cpuidle.html#the-menu-governor)是默认调度器，它的设计原则很简单，为CPU选择一个空闲状态时，它尝试预测器闲置持续时间并使用预测值进行空闲状态选择。 -->

##### 1.2.2.2 [CPU Performance Scaling](https://docs.kernel.org/admin-guide/pm/cpufreq.html#cpu-performance-scaling)

这个概念就是指的是P-State,指的是让CPU以多种不同的时钟频率和电压配置运行。  
Linux内核中用`CPUFreq`子系统支持CPU的性能缩放，它包含了3个层次的代码: the core, scaling governors和 scaling drivers。

1. the core,提供了公共的用户接口和基础代码，定义了基本框架。
2. scaling governors, 实现估计CPU容量的算法。
3. scaling drivers直接和硬件交互，根据scaling governors的请求改变CPU的P-State。

## 2 状态控制和负载控制

```bash
docker run -it --privileged --hostname cpuload --name cpuload ubuntu:20.04 /bin/bash

# 安装modprobe
apt-get install kmod -y
# 安装cpu-power指令,https://manpages.debian.org/testing/linux-cpupower/index.html
apt-get install linux-tools-common linux-tools-$(uname -r)
# cpupower --help
```

`/sys/devices/system/cpu/cpu<N>/cpuidle/`

### 2.1 CPUIdle (C-State)

#### Ubuntu

cpu-idle的控制启用和禁用c-state，不能指定cpu立刻进入某个c-state

1. idle-info

    ```bash
    # --cpu 指定cpu
    cpupower --cpu 0 idle-info
    ```

    结果输出
    - `latency`,Exit latency of the idle state in microseconds.
    - `usage`,Total number of times the hardware has been asked by the given CPU to enter this idle state.

    ```text
    CPUidle driver: intel_idle
    CPUidle governor: menu
    analyzing CPU 0:

    Number of idle states: 4 
    Available idle states: POLL C1 C1E C6
    POLL:
    Flags/Description: CPUIDLE CORE POLL IDLE
    Latency: 0
    Usage: 4338722
    Duration: 32917820
    C1:
    Flags/Description: MWAIT 0x00
    Latency: 2
    Usage: 202782261
    Duration: 9259454371
    C1E:
    Flags/Description: MWAIT 0x01
    Latency: 10
    Usage: 126552375
    Duration: 31762469023
    C6:
    Flags/Description: MWAIT 0x20
    Latency: 133
    Usage: 708330302
    Duration: 3409706493833
    ```

2. idle-set

    ```bash
    # 启用所有的idle-sate
    cpupower --cpu 15 idle-set --enable-all
    # 禁用 POLL 
    cpupower --cpu 15 idle-set --disable POLL
    cpupower -c 15 frequency-info 
    ```

    输出结果中POLL已经被禁用.

    ```text
    CPUidle driver: intel_idle
    CPUidle governor: menu
    analyzing CPU 15:

    Number of idle states: 4
    Available idle states: POLL C1 C1E C6
    POLL (DISABLED) :
    Flags/Description: CPUIDLE CORE POLL IDLE
    Latency: 0
    Usage: 5442095
    Duration: 27783830
    C1:
    Flags/Description: MWAIT 0x00
    Latency: 2
    Usage: 227929586
    Duration: 9066384859
    C1E:
    Flags/Description: MWAIT 0x01
    Latency: 10
    Usage: 144210011
    Duration: 29640193014
    C6:
    Flags/Description: MWAIT 0x20
    Latency: 133
    Usage: 292763623
    Duration: 3842674571060
    ```

<!-- 3. monitor
    这里的值可能不准确
    ```bash
    cpupower --cpu 15 monitor -i 1
    ```
    输出
    ```text
                 | Nehalem                   || Mperf              || Idle_Stats
    PKG|CORE| CPU| C3   | C6   | PC3  | PC6  || C0   | Cx   | Freq || POLL | C1   | C1E  | C6
      1|   1|  15|******|******|******|******||******|******|******||  0.00|  0.00|  0.12| 52.45
    ``` -->

### 2.2 CPUFreq (P-State)

#### Ubuntu

1. frequency-info

    ```bash
    # -c/--cpu 指定cpu编号
    cpupower -c 0 frequency-info 
    ```

    输出中提供了CPU的`maximum transition latency`,`hardware limits`,`available cpufreq governors`,`current CPU frequency`

    ```text
    analyzing CPU 0:
    driver: intel_pstate
    CPUs which run at the same hardware frequency: 0
    CPUs which need to have their frequency coordinated by software: 0
    maximum transition latency:  Cannot determine or is not supported.
    hardware limits: 1000 MHz - 3.20 GHz
    available cpufreq governors: performance powersave
    current policy: frequency should be within 1000 MHz and 3.20 GHz.
                    The governor "powersave" may decide which speed to use
                    within this range.
    current CPU frequency: Unable to call hardware
    current CPU frequency: 2.04 GHz (asserted by call to kernel)
    boost state support:
        Supported: yes
        Active: yes
    ```

2. frequency-set

    ```bash
    # 可用cpufreq-set --help查看flags
    # 设置最小2.0GHz频率,最大2.5GHzx频率
    cpupower --cpu 15 frequency-set --min 2.0GHz --max 2.5GHz 
    # 当前策略选择
    cpupower -c 15 frequency-info 
    ```

    输出结果可见，`current policy`显示了我们的参数值，同时观察了其他CPU，并无变化。

    ```text
    analyzing CPU 15:
    driver: intel_pstate
    CPUs which run at the same hardware frequency: 15
    CPUs which need to have their frequency coordinated by software: 15
    maximum transition latency:  Cannot determine or is not supported.
    hardware limits: 1000 MHz - 3.20 GHz
    available cpufreq governors: performance powersave
    current policy: frequency should be within 2.00 GHz and 2.50 GHz.
                    The governor "performance" may decide which speed to use
                    within this range.
    current CPU frequency: Unable to call hardware
    current CPU frequency: 2.24 GHz (asserted by call to kernel)
    boost state support:
        Supported: yes
        Active: yes
    ```

### 2.3 Workload (Utilization)

基于该工具实现了CPU负载生成:https://github.com/SheepHuan/cpuload

## 3 实验观察

设置了3个实验，这里我们无法设置确定的频率。在`Intel(R) Xeon(R) Gold 5120 CPU`实验的。

1. 设置cpu50,利用率15%,频率在2.4GHz-2.5GHz；cpu52，利用率95%，频率在2.4GHz-2.5GHz。

    ![2.png](http://pic.yanghuan.site/i/2023/05/30/6475b46816d5b.png)
    <center style="">图 3-1</center> 

2. 设置cpu50,利用率15%,频率在2.499GHz-2.5GHz；cpu52，利用率95%，频率在2.499GHz-2.5GHz。

    ![3.png](http://pic.yanghuan.site/i/2023/05/30/6475b46888260.png)
    <center style="">图 3-2</center> 

3. 设置cpu50,利用率15%,频率在0.9GHz-1.0GHz；cpu52，利用率95%，频率在2.4GHz-2.5GHz。

    ![1.png](http://pic.yanghuan.site/i/2023/05/30/6475b467b37cd.png)
    <center style="">图 3-3</center> 

- 图3-1中基本上cpu52的频率稳定在2.5GHz，没有什么波动，但是cpu50的频率在2.4GHz来回波动。说明2.4GHz的频率来说对于利用率15%来说太高了。
- 图3-2中我们进一步限制频率范围cpu50的波动更大了，cpu52和之前没什么区别。
- 图3-3中，我们将cpu50的频率降低后，我们发现cpu50频率能够稳定在1.0GHz了，但是两端出现了大幅度波动，不知道是不是因为还有其他进程调用了还是有其他原因。
- 在图3-1,2,3中，我们发现utilization的数值并不稳定，这个地方还要分析下为什么，检查下负载生成的部分是不是实现的不好（TODO）。

## 4 资料

1. https://docs.kernel.org/admin-guide/pm/working-state.html