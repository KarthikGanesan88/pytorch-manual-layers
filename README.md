# pytorch-manual-layers

This repo hosts custom versions of conv2d and linear layers written in C++ and CUDA, to work with PyTorch. 

I did this to be able to experiment with modifying the underlying operations in both layers for greater speed (in custom ML hardware). I found that doing this modifications to the PyTorch code was much more difficult. So I started with the CPP version and found it was too slow so wrote the layers in CUDA as well. So this code does not use CUDNN or any other Nvidia functions but rather performs the operations using simple MAC operations. 

Feel free to use this code if it can be of use to you. But *please note*, I am absolutely no expert in PyTorch. So if you have any questions about PyTorch itself, please post it on their discussion forums. 

## Repo description

* mnist_test.py: The main file that loads the baseline network using pytorch code, the custom model using the custom layers as well as a model that uses fake quantization using the Nvidia pytorch-quantization-toolkit. 
* mnist_train.py: A way to train your own network if you want to try other networks. 
* approximate_convolutional_layer.py: Custom implementation of CUDA layer. 
* approximate_fully_connected_layer.py: Custom implementation of linear layer. 
* cpp:
  * cpp_layers.cpp: Definitions of both layers in CPP. 
  * setup_cpp.py: File to compile and load the CPP functions as a library that PyTorch can use. 
* cuda
  * cuda_layers.cu: Definitions of both layers in CUDA C++. 
  * setup_cuda.py: File to compile and load the CUDA functions as a library that PyTorch can use. 
* Data
  * pytorch_mnist.pth: Pre-trained LeNet-5 like CNN for MNIST.
  * pytorch_mnist_quant.pth: Above network but 'fake quantized' for Int8 inference.

## Usage instructions

To use GPU, make sure you have the latest nvidia drivers and cuda version. Without it, you can still use the C++ version but it will be quite slow since I did not implement any vectorization etc. 

1. Install PyTorch, following the instructions [here](https://pytorch.org/get-started/locally/). I used a conda environment, but I would assume using pip would work the same. 
2. Download all the files in this repo.
3. For CPP/CUDA, go to each folder and type in 'python setup_<name>.py install', where <name> is either 'cpp' or 'cuda'. This will compile the associated file and install the library file for pytorch to use. Again, I did all this in my conva env. 
4. Run 'python mnist_test.py'. 

## My system setup
  
* Ubuntu 21.04
* PyTorch version: 1.9.1
* Nvidia 2080ti
  * Driver version: 470.63.01
  * CUDA version: 11.4
  * CUDA toolkit (inside conda): 11.1.74

### Acknowledgements

The custom layer code is heavily based on code from ' AG-X09/Defensive-Approximation', [here](https://github.com/AG-X09/Defensive-Approximation)
The matrix multiplication code in CUDA is based on code from 'lzhengchun/matrix-cuda', [here](https://github.com/lzhengchun/matrix-cuda). 
