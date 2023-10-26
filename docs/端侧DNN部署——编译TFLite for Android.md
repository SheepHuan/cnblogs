## 编译tensorflow lite

1. 下载tensorflow代码,进入lite项目

```bash
git clone -b v2.13.0 https://github.com/tensorflow/tensorflow.git

# 进入工作目录
cd tensorflow/tensorflow/lite
```

2. 基于cmake交叉编译tf lite android库,可以参考[文档](https://www.tensorflow.org/lite/guide/build_cmake)

**在windows上交叉编译android**
```powershell
# 创建build/
mkdir build4android 
cd build4android
rm -Recurse -Force *

# 生成，注意android默认不开启gpu，默认开启nnapi了。
cmake -S .. -B . `
-DBUILD_SHARED_LIBS=ON `
-DTFLITE_ENABLE_GPU=ON `
-DCMAKE_TOOLCHAIN_FILE="C:/Users/sheep/AppData/Local/Android/Sdk/ndk/25.2.9519653/build/cmake/android.toolchain.cmake" `
-DANDROID_ABI=arm64-v8a `
-DANDROID_PLATFORM=android-29 `
-DCMAKE_INSTALL_PREFIX="D:/code/tmp/tensorflow/tensorflow/lite/build4android/install" `
-G="Ninja"

# 编译
cmake --build .
cmake --install .
```
执行完后，`build4android/`目录下会出现`lib/`和`include/`目录。