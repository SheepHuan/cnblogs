



## EFFICIENT GPU KERNELS FOR N:M-SPARSE WEIGHTS IN DEEP LEARNING
### 概述/摘要
N:M稀疏是一种能让模型保持比较高的的精度和及效率的方法。但是由于GPU Kernel上没有很好的适配各种N:M比例的实现，所以在N:M稀疏的应用受到限制。作者提出[`nmSPARSE`](https://github.com/microsoft/SparTA/tree/nmsparse)库在GPU高效地实现了两个N:M系数的基本算子: sparse matrix-vector multiplication (SpMV)，sparse matrix-matrix multiplication (SpMM). 

### 关键挑战

### 核心方法

### 实验结果