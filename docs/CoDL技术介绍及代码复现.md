
## 1 CoDL MobiSys2021 技术介绍
CoDL是一个移动设备上并行DL推理框架,它基于Xiaomi MACE框架开发,主要利用手机SoC的CPU和GPU并行加速模型推理过程,而当前的主要的推理框架还是依次只利用一个设备去推理.
### 1.1 挑战和解决方案
1. 减少异构处理器之间的数据共享开销
2. 如何为异构处理器恰当地分配OP

为了充分利用异构处理器来加速Model的每一个OP,论文中提到了两个技术来解决上面的挑战.
1. **混合类型优化的数据共享(hybrid type-friendly data sharing)**,这个技术允许每个处理器能够使用其最高效的数据类型去进行推理,因为文章中设计了实验,其结果表明如果异构处理器共用统一类型的数据结构会导致推理效率并不高效且不合理.为了减少共享的开销,同时还采用了**hybrid-dimension partitioning**和**operator chain**.
2. **非线性和并发感知的延迟预测(non-linearity and concurrency-aware latency prediction)**.通过构建一个轻量且准确的延迟预测器来指导OP切分来确保合理性.

有空再详细介绍介绍CoDL的技术实现
## 2 代码复现
CoDL的代码已经开源在了GitHub上,地址是:https://github.com/csu-eis/CoDL/

### 2.1 CoDL运行流程
![CoDL运行流程(摘自论文).png](http://pic.yanghuan.site/i/2023/05/25/646e3c11d7201.png)
#### 2.1.1 离线阶段(offline)
在离线阶段,CoDL设计了一个轻量的延迟预测器指导在在线阶段的OP切分,它会考虑到data sharing的开销.
#### 2.1.2 在线阶段(online)
在线阶段分成了两个部分,一个是OP切分(operator partitioner),另一个是OP协同执行(operator co-executor).
1. operator partitioner,这主要负责去找对于输入model的优化的OP 切分的计划,同样OP的权重也会被预分配CPU和GPU上,从而避免推理时再去转换.
2. operator co-executor,这主要是就是根据切分计划去对OP执行进行同步,并对不同的处理器采用不同的数据类型.
    


### 2.2 基于CoDL加速模型
废话不多说了,开干.
#### 2.2.1 构建Docker环境和可执行文件

1. 编译镜像
   ```bash
    git clone https://github.com/csu-eis/CoDL.git
    cd CoDL
    # 基于dockerfile 编译出docker
    docker build -t codl/codl -f ./Dockerfile .
    ```
2. 创建环境
   ```bash
   # 这里的{worksapce}设置为自己电脑上的工作区绝对路径
   sudo docker run -it --name codl-u16 --privileged -v /dev/bus/usb:/dev/bus/usb --hostname codl codl/codl:latest  /bin/bash

   # sudo docker run -it --name codl-u16 --privileged -v /dev/bus/usb:/dev/bus/usb --hostname codl codl/codl:latest  /bin/bash
   ```
   这里我们需要注意一个电脑上只能同时开启一个adb server.请使用`adb kill-server`命令关闭其他正在运行的adb sever,确保只有docker里面的adb server正在运行
   ```bash
   #查看一下手机序列号
   adb devices
   #结果
   #root@codl:~/codl-mobile# adb devices
   #List of devices attached
   #3a9c4f5 device
   ```
3. 编译可执行文件并push到手机
   ```bash
   # 确保进入了docker 容器
   cd /root/codl-mobile
   # 编译可执行文件
   bash tools/codl/build_executable_files.sh
   ```
   构建成功如下图  
   ![1684947363647.png](http://pic.yanghuan.site/i/2023/05/25/646e41a53b6e0.png)

    push 到手机
   ```bash
   # 确保pwd还在/root/codl-mobile
   bash tools/codl/push_executable_files.sh 3a9c4f5
   # bash tools/codl/push_executable_files.sh H933CK9N01234567
   # bash tools/codl/push_executable_files.sh 9YS0220110011018
   # 确保push成功
   adb -s 3a9c4f5 shell "ls /data/local/tmp/codl"
   # adb -s 9YS0220110011018 shell "ls /data/local/tmp/codl"
   ```
    push成功如下图  
    ![1684947603851.png](http://pic.yanghuan.site/i/2023/05/25/646e429555b42.png)

#### 2.2.2 构建延迟预测器
1. 收集延迟数据
   ```bash
   # 确保还在docker 容器内
   cd /root/codl-eval-tools/codl-lat-collect-and-eval/
   # 注意,如果我们需要对其他的Soc进行测试,我们需要改写collect_all_op_latency.sh,这里的.sh只支持[sdm855, sdm865,sdm888,kirin990]这几种
   # 注意:/root/codl-eval-tools/codl-lat-collect-and-eval/tools/collect_all_op_latency.sh 修改测试GPU Buffer/Images类型 
   bash tools/collect_all_op_latency.sh sdm865 3a9c4f5
   # bash tools/collect_all_op_latency.sh kirin990 9YS0220110011018
   # bash tools/collect_all_op_latency.sh dimensity8050 H933CK9N01234567
   ```
   数据获取完后,会在产生`~/codl-eval-tools/codl-lat-collect-and-eval/lat_datasets/`目录，里面有一个`sdm865`文件夹，里面包含了一些内容
   ```Text
   |-- t_conv2d_cpu_direct.csv
   |-- t_conv2d_cpu_gemm.csv
   |-- t_conv2d_cpu_winograd.csv
   |-- t_conv2d_cpu_winograd_combined.csv
   |-- t_conv2d_cpu_winograd_gemm.csv
   |-- t_conv2d_gpu_direct.csv
   |-- t_data_sharing.csv
   |-- t_fc_cpu_gemv.csv
   |-- t_fc_gpu_direct.csv
   |-- t_mulayer_conv2d_cpu.csv
   |-- t_mulayer_conv2d_gpu.csv
   |-- t_mulayer_fc_cpu.csv
   |-- t_mulayer_fc_gpu.csv
   |-- t_mulayer_pooling_cpu.csv
   |-- t_mulayer_pooling_gpu.csv
   |-- t_pooling_cpu_direct_max.csv
   |-- t_pooling_gpu_direct_max.csv
   ```
   下面就是开始训练预测器了。
2. 训练延迟预测器
   ```bash
   cd /root/codl-eval-tools/codl-lat-predict

   bash tools/train_and_eval_lat_predictors.sh /root/codl-eval-tools/codl-lat-collect-and-eval/lat_datasets sdm865
   ```
3. 测试CoDL性能
   ```bash
   # 1. 推送模型文件到设备上
   cd /root/codl-eval-tools/codl-lat-collect-and-eval

   bash tools/push_lat_predictors_and_configs.sh \
   sdm865 \
   /root/codl-eval-tools/codl-lat-predict/saved_models \
   /root/codl-eval-tools/codl-lat-collect-and-eval/configs \
   3a9c4f5
   # 2. 测试性能
   cd /root/codl-eval-tools/codl-lat-collect-and-eval
   # /root/codl-eval-tools/codl-lat-collect-and-eval/test/op_chain_adb_run_test.py 中修改测试GPU Buffer/Image类型
   bash tools/eval_codl_and_baselines.sh 3a9c4f5
   ```
   ![1684991657557.png](http://pic.yanghuan.site/i/2023/05/25/646eeeac00482.png)

### 2.3 自定义测试

#### 2.3.1 自定义SoC测试
我们以发哥的天玑8050芯片为例。
1. 进入`/root/codl-eval-tools/codl-lat-collect-and-eval/utils/soc`,新建`dimensity8050.yaml`文件，配置内容如下。

   ```yaml
   global_mem_cache_size: 1048576
   compute_units: 9
   max_work_group_size: 512
   kernel_wave_size: 4
   ```
   因为`/root/codl-eval-tools/codl-lat-collect-and-eval/measure.py`文件在`line:556`行通过`RegistrySoc()`加载所有SoC的opencl配置信息。不过显然这里的信息是要经过测试才能获得的。
2. 在`/root/codl-eval-tools/codl-lat-collect-and-eval/measure.py`理解修改参数

   ```Python
   # 在line:537处添加dimensity8050
   parser.add_argument('--soc', type=str, required=True,
                        choices=['sdm855', 'sdm865', 'sdm888', 'kirin960', 'kirin990','dimensity8050'],
                        default='sdm855', help='SoC')

   ```

3. 逐级剖析，发现`op_data_utils`会根据设备参数提取参数。（具体是啥参数，暂时没搞懂）

   ```Python  
   ...
   op_data_utils = get_dataset_utils(dataset_name) # line:344
   ...
   data_dict = op_data_utils.extract(target_op_param, 
                                    device,
                                    do_data_transform,
                                    lat_list) # line:431
   ...
   ```

4. 考虑如何计算正确的参数。
   `kernel_wave_size`目前还没理解是啥意思。而`global_mem_cache_size`、`compute_units`、`max_work_group_size`参数理论啥可以从libOpenCL.so中提取出来,我在CoDL代码中已经加了获取`max_work_group_size`的代码，我们只需要看下Log就能知道了（代码已提交至CoDL代码库）。
   ```bash 
   # 保证文件存在后，在bash里面运行这个命令试试。
   adb -s H933CK9N01234567 shell "/data/local/tmp/codl/codl_op_run --compute --cpu_affinity_policy=1 --rounds=21 --gpu_memory_type=1 --num_threads=4 --part_dim=4 --part_ratio=0.000000 --cu_hint=1 --op_type=Conv2D --input_shape=\"1,418,418,3\" --weight_shape=\"32,3,3,3\" --strides=\"1,1\""
   ```
   部分结果输出
   ```Text
   I mace/core/runtime/opencl/opencl_runtime.cc:530] Creating OpenCL runtime
   I mace/core/runtime/opencl/opencl_runtime.cc:545] Using platform: ARM Platform, FULL_PROFILE, OpenCL 3.0 v1.r32p1-01eac0.efd03ef21f136842c0935bd4f493fe81, unknown
   I mace/core/runtime/opencl/opencl_runtime.cc:604] OpenCL device: Mali-G77 MC9 r0p1
   I mace/core/runtime/opencl/opencl_runtime.cc:605] OpenCL device version: OpenCL 3.0 v1.r32p1-01eac0.efd03ef21f136842c0935bd4f493fe81
   I mace/core/runtime/opencl/opencl_runtime.cc:606] OpenCL device extensions: cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_3d_image_writes cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_fp16 cl_khr_icd cl_khr_egl_image cl_khr_image2d_from_buffer cl_khr_depth_images cl_khr_subgroups cl_khr_subgroup_extended_types cl_khr_subgroup_non_uniform_vote cl_khr_subgroup_ballot cl_khr_il_program cl_khr_priority_hints cl_khr_create_command_queue cl_khr_spirv_no_integer_wrap_decoration cl_khr_extended_versioning cl_khr_device_uuid cl_arm_core_id cl_arm_printf cl_arm_non_uniform_work_group_size cl_arm_import_memory cl_arm_import_memory_dma_buf cl_arm_import_memory_host cl_arm_import_memory_protected cl_arm_import_memory_android_hardware_buffer cl_arm_integer_dot_product_int8 cl_arm_integer_dot_product_accumulate_int8 cl_arm_integer_dot_product_accumulate_saturate_int8 cl_arm_job_slot_selection cl_arm_scheduling_controls cl_arm_controlled_kernel_termination cl_ext_cxx_for_opencl
   W mace/core/kv_storage.cc:109] Failed to read kv store file: /data/local/tmp/mace_run/interior/mace_cl_compiled_program.bin
   W mace/core/runtime/opencl/opencl_runtime.cc:703] Load OpenCL cached compiled kernel file failed. Please make sure the storage directory exist and you have Write&Read permission
   I mace/core/runtime/opencl/opencl_runtime.cc:767] OpenCLRuntime: global_mem_cacheline_size 64, global_mem_cache_size 1048576, compute_units 9, warp_size 0, max_wgp_size 512, kwg_size 0
   I mace/core/runtime/opencl/opencl_runtime.cc:510] OpenCL memory flags: CL_MEM_READ_WRITE 1, CL_MEM_WRITE_ONLY 2, CL_MEM_READ_ONLY 4
   ```
   `warp_size`和`kwg_size`结果为0，好像是因为它不是从libOpenCL.so中获取的。而是作者自己设置的配置参数。

到此,根据测试结果修改配置文件参数，我们就可以运行天玑8050 SoC进行CoDL的复现测试了。注意Adreno系列的GPU,CoDL用GPU image格式效果较好,对于Mali系列GPU,CoDL用GPU Buffer格式较好。
这个可以在2.2.2中手动指定参数。

#### 2.3.1 自定义模型测试

在CoDL中自定义一个测试模型比较复杂，因为在`/root/codl-mobiletest/codl_run/op_chain_test.cc`文件的`main`函数是`codl_run`脚本的入口函数，里面的模型是作者用C++定义的结构。

```C++
...
int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  print_flags();
  example_yolov2();
  example_posenet();
  example_alexnet();
  example_vgg16();
  example_fast_style_transfer();
  example_retinaface();
  example_mobilenetv1();
  example_mobilenetv2();
  example_resnet50v1();
  example_resnet50v2();
  exmaple_matmul_net();
  example_bert();
  example_op_chain_net();
  example_full_op_chain_search();
  
  return 0;
}
```