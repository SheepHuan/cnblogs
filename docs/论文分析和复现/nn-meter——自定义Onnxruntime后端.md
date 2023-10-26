## 1 nn-Meter 后端结构
nn-Meter文档中给出了自定义后端的流程(https://github.com/microsoft/nn-Meter/blob/main/docs/builder/prepare_backend.md#-build-customized-backend-)。

### 1.1 BaseBackend
首先介绍基类BaseBackend，任何一个Backend都是继承这个类。它的定义和实现保存在了`nn_meter/builder/backends/interface.py`文件中。
重要的成员变量
- `profiler_class`
- `parser_class`
- `self.parser`,由`parser_class`类初始化而来。
- `self.profiler`,由`profiler_class`初始化而来。

重要的成员函数：
- `self.update_configs()`, 负责解析外部的yaml配置文件,更新self.profiler的参数。
- `self.convert_model()`，将Tensorflow Keras的pb模型保存为TFLite格式供后端推理。
- `self.profile()`, 先调用`self.profiler()`运行模型返回profiling内容，再由`self.parser`正则提取延时结果。
- `self.profile_model_file()`, profile指定的模型，将执行结果返回。
- `self.test_connection()`，测试后端的连通性。

### 1.2 BaseProfiler
这个类就是预定义了一个`Profiler.profile()`接口，用于在edge设备上运行模型。
### 1.3 BaseParser
- 定义了`self.parse()`函数用于解析`Profiler.profile()`的运行结果
- 定义了属性`self.results`用于包装延时数据。
## 2 自定义Onnxruntime后端

### 2.1 OnnxruntimeBackend
```Python
# Copyright (c) 2023, Huan Yang
# All rights reserved.

import logging
from ..interface import BaseBackend
logging = logging.getLogger("nn-Meter")


class OnnxruntimeBackend(BaseBackend):
    parser_class = None
    profiler_class = None

    def update_configs(self):
        """update the config parameters for Onnxruntime platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
            'benchmark_model_path': self.configs['BENCHMARK_MODEL_PATH'],
            'serial': self.configs['DEVICE_SERIAL'],
            'dst_kernel_path': self.configs['KERNEL_PATH']
        })

    def convert_model(self, model_path, save_path, input_shape=None):
        """onnx models change nothing
        """
        return model_path

    def test_connection(self):
        """check the status of backend interface connection, ideally including open/close/check_healthy...
        """
        from ppadb.client import Client as AdbClient
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.configs['DEVICE_SERIAL']:
            device = client.device(self.configs['DEVICE_SERIAL'])
        else:
            device = client.devices()[0]
        logging.keyinfo(device.shell("echo hello backend !"))
```

### 2.2 OnnxruntimeProfiler
```Python
# Copyright (c) 2023, Huan Yang
# All rights reserved.

import os
from ..interface import BaseProfiler

class OnnxruntimeProfiler(BaseProfiler):
    use_cpu = None
    use_nnapi = None
    
    
    def __init__(self, dst_kernel_path, benchmark_model_path, graph_path='', dst_graph_path='', serial='', num_threads=1, num_runs=50, warm_ups=10):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        """
        self._serial = serial
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._dst_kernel_path = dst_kernel_path
        self._benchmark_model_path = benchmark_model_path
        self._num_threads = num_threads
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def profile(self, graph_path, preserve = False, clean = True, **kwargs):
        """
        @params:
        preserve: onnx file exists in remote dir. No need to push it again.
        clean: remove onnx file after running.
        """
        model_name = os.path.basename(graph_path)
        remote_graph_path = os.path.join(self._dst_graph_path, model_name)
   
        from ppadb.client import Client as AdbClient
        client = AdbClient(host="127.0.0.1", port=5037)
        if self._serial:
            device = client.device(self._serial)
        else:
            device = client.devices()[0]

        if not preserve:
            device.push(graph_path, remote_graph_path)
        try:
       
            # kernel_cmd = f'--kernel_path={self._dst_kernel_path}' if self._dst_kernel_path else ''
      
            res = device.shell(f' {self._benchmark_model_path}' \
                               f' --num_threads={self._num_threads}' \
                               f' --num_runs={self._num_runs}' \
                               f' --warmup_runs={self._warm_ups}' \
                               f' --graph={remote_graph_path}' \
                               f' --enable_op_profiling=false'
                              )
        except:
            raise
        finally:
            if clean:
                if self._serial:
                    os.system(f"adb -s {self._serial} shell rm {remote_graph_path}")
                    # os.remove(graph_path)
                else:
                    os.system(f"adb shell rm {remote_graph_path}")

        return res
```

### 2.3 OnnxruntimeParser
```Python
# Copyright (c) 2023, Huan Yang
# All rights reserved.

from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
import re

class OnnxruntimeCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content):
        # self.nodes = self._parse_nodes(content)
        self.total_latency = self._parse_total_latency(content)
        return self

    # TODO 重写规则
    def _parse_total_latency(self, content):
        
        pattern = r"\d+\.\d+"  # 匹配浮点数的正则表达式
        matches = re.findall(pattern, content)
        print(matches)  # 输出 ['635.800000', '6.477654']
        total_latency = Latency()
        if matches:
            # convert microseconds to millisecond
            total_latency = Latency(float(matches[1]), float(matches[2]))

        return total_latency
    
    @property
    def latency(self):
        return self.total_latency

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results

```

### 2.4 测试后端可用性

#### 2.4.1 onnxruntime android benchmark
查看博客[《基于onnxruntime c++ api开发benchmark工具》](https://www.cnblogs.com/sheephuan/p/17430937.html),访问密码:`yanghuan`。该工具负责在android获取模型执行的实际latency。

```Python
# 新建一个脚本测试下
from nn_meter.builder.backends.ort.ort_profiler import OnnxruntimeProfiler
from nn_meter.builder.backends.ort.ort_cpu import OnnxruntimeCPULatencyParser
    
if __name__=="__main__":
    profiler = OnnxruntimeProfiler('','/data/local/tmp/hcp/main',dst_graph_path='/mnt/sdcard/ort_models/',serial='3a9c4f5')
    res = profiler.profile("/mnt/sdcard/ort_models/FasterRCNN-12.onnx","/data/local/tmp/hcp/libs",preserve=True,clean=False)
    parser = OnnxruntimeCPULatencyParser()
    print( parser.parse(res).latency)
```
执行结果

![1684996836382.png](http://pic.yanghuan.site/i/2023/05/25/646f02e7017c6.png)

走到这一步基本上已经搞定了。
### 2.5 注册后端
