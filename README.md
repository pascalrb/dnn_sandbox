# DNN Sandbox
DNN Processing Investigation <br/><br/>

## Primer on HW-SW codesign for DNN Processing
`Guide_to_SW_Compiler_HW_Opts_for_DL.pdf`


## Standard Computer Vision Model
`lenet5_vgg` --> mix between LeNet5 and VGG architecture. 
  - 83.8% accuracy in image classification on CIFAR-10 dataset. 

## Deep Compression 
`deep_compression` --> implementaiton of the deep compression paper by **Han, et al. (2015)** with a few additional features.
  - Pruning, Quantization, Huffman Coding.
  - 44.7x compression rate while maintaining accuracy at 90.12%. 

## Distributed Learning
`distributed_learning` --> implemention of PyTorch's multiprocessing's Distributed Data Parallel (DDP)
  - 5x learning speedup
