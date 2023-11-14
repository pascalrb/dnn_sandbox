# dnn_sandbox
DNN Processing Investigation

lenet5_vgg --> mix between LeNet5 and VGG architecture. 
  - 83.8% accuracy in image classificatio on CIFAR-10 dataset.


deep_compression --> implementaiton of the deep compression paper by **Han, et al. (2015)** with a few additional features.
  - Pruning, Quantization, Huffman Coding.
  - 44.7x compression rate while maintaining accuracy at 90.12%. 
