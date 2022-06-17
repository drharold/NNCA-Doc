# 神经网络压缩算法文档 (NNCA-Doc: Neural Network Compression Algorithms Documentation)

## 介绍
个人兴趣向。收集、整理一些开源的，用于深度学习 (神经网络) 模型的高级压缩算法的项目和库。

- 后训练量化 (PTQ，post-training quantization)

>后训练量化，也有称作训练后量化或者离线量化。
>
>指对已经训练好的模型 (预训练模型) 进行校正和量化，不需要重训练，属于微调量化中的一种方法。
>
>后训练量化分为后训练动态量化(post-training dynamic/weight qantization)和后训练静态量化(post-training static qantization)，区别在于是否有样本数据进行校正，若没有或者不需要样本数据校正则是后训练动态量化。
>
>需要注意的是，对于low-bit (例如 4 bit, 3 bit) 量化，有部分硬件不支持 low-bit 运行，则需要自行添加相关的硬件适配代码。

>参考资料 (推荐首要看PyTorch)：
>- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/#post-training-dynamicweight-only-quantization)
>- [Paddle-Lite 模型量化](https://paddle-lite.readthedocs.io/zh/develop/user_guides/quant_aware.html#id7)
>- [TensorFlow 训练后量化](https://www.tensorflow.org/lite/performance/post_training_quantization)

- 位移，加法或者其它操作 (shift, add, and other operations)

>指采用位移，加法或者其它操作代替乘法，起到加速的作用(可能需要自行添加相关操作的cuda或其它硬件适配的代码)。
>
>目前有位移操作，加法操作，以及位移加法相结合的操作。

## 后训练量化 (PTQ，post-training quantization)
### 项目
2022
- (ICLR) SQuant: On-the-Fly Data-Free Quantization via Diagonal Hessian Approximation [[PyTorch](https://github.com/clevercool/SQuant)]
- (ICLR) QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization [[PyTorch](https://github.com/wimh966/QDrop)]
- (IJCAI) FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer [[PyTorch](https://github.com/megvii-research/FQ-ViT)]

2021
- (ICLR) BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction [[PyTorch](https://github.com/yhhhli/BRECQ)]
- (NeurIPS) Post-Training Sparsity-Aware Quantization [[PyTorch](https://github.com/gilshm/sparq)]
- ((NeurIPS) PTQ4ViT: Post-Training Quantization Framework for Vision Transformers [[PyTorch](https://github.com/hahnyuan/ptq4vit)]

2020
- (ICML) Up or Down? Adaptive Rounding for Post-Training Quantization [[PyTorch & TensorFlow](https://github.com/quic/aimet)]

2019
- (NeurIPS) Post-training 4-bit quantization of convolution networks for rapid-deployment [[PyTorch](https://github.com/submission2019/cnn-quantization)]
- (ICCV) Data-Free Quantization Through Weight Equalization and Bias Correction [[PyTorch & TensorFlow](https://github.com/quic/aimet)]

2018
- (CVPR) Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference [[TensorFlow](https://github.com/tensorflow/tensorflow/tree/v2.6.4/tensorflow/lite/micro)] [[TensorFlow Lite](https://github.com/tensorflow/tflite-micro)]

### 库
- AIMET [[PyTorch & TensorFlow](https://github.com/quic/aimet)]
- MQBench [[PyTorch](https://github.com/ModelTC/MQBench)]
- NNCF [[PyTorch & TensorFlow](https://github.com/openvinotoolkit/nncf)]

## 位移，加法或者其它操作 (shift, add, and other operations)
### 项目
2021
- (CVPR) DeepShift: Towards Multiplication-Less Neural Networks [[PyTorch](https://github.com/mostafaelhoushi/DeepShift)]

2020
- (CVPR) AdderNet: Do We Really Need Multiplications in Deep Learning? [[PyTorch](https://github.com/huawei-noah/AdderNet)]
- (NeurIPS) ShiftAddNet: A Hardware-Inspired Deep Network [[PyTorch](https://github.com/RICE-EIC/ShiftAddNet)]

2019
- (CVPR) All you need is a few shifts: Designing efficient convolutional neural networks for image classification [[PyTorch (Not offical)](https://github.com/DeadAt0m/ActiveSparseShifts-PyTorch)] [[TensorFlow (Not offical)](https://github.com/Eunhui-Kim/SSPQ)]

2018
- (CVPR) Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions [[PyTorch](https://github.com/alvinwan/shiftresnet-cifar)]

