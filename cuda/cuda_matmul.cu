#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

/*
*********************************************************************
function name: matmul_cuda
description: dot product of two arbitrarily sized matrices.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  output: output image of size m x k.
  m,n,k: sizes of matrices.
  batch_size: Number of images in each batch.
return: none
Acknowledgement: Original code from 'lzhengchun/matrix-cuda' on github.
link: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
*********************************************************************
*/
__global__ void matmul_cuda(
  const float *image,
  const float *weight,
	const float *bias,
  float *output,
  const int m,
  const int n,
  const int k,
  const int batch_size)
{

    // This code doesn't really get much faster using shared memory, since
    // accesses to the image matrix are all sequential anyway. The first access
    // already caches everything, making shared memory useless.

    int img = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f, product_appx = 0.0f, product = 0.0f;
    if( col < k && row < m && img < batch_size){
        for(int i = 0; i < n; i++){
						sum += image[(img*m*n)+(row*n) + i] * weight[i * k + col];
				}
        output[(img*m*k)+(row*k) + col] = sum + bias[col];
    }
}

/*
*********************************************************************
function name: conv_forward
description: convolutional layer that calls the matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
  b: Number of images in each batch.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor conv_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	int b
) {

  // Create an output of size b X m X k, directly on the GPU.
	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({b, m, k}, options);

  // Use this block size to not exceed 1024 threads across all 3 dimensions.
  // You can also do dimblock(16 x 16 x 4) to use all 1024 threads if your
  // batches are small.
	unsigned int block_size = 8;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;
	unsigned int grid_images = (b + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows, grid_images);
	dim3 dimBlock(block_size, block_size, block_size);

  // This is not the 'pytorch recommended way' of launching this kernel.
  // But it works just fine so I've left it this way since it is easier to debug
  // if there is an issue launching the kernel for example.

	matmul_cuda<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, b
	);

  cudaDeviceSynchronize();
  return output;
}

/*
*********************************************************************
function name: linear_forward
description: linear layer that calls the matmul cuda kernel.
parameters:
  image: Input image of size m X n.
  weight: weight kernel of size n X k.
  bias: bias per output channel.
  m,n,k: sizes of matrices.
return:
  output: output image of size m x k.
*********************************************************************
*/
torch::Tensor linear_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({m,k}, options);

	unsigned int block_size = 32;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(block_size, block_size);

  // Linear layers have a vector input. But to re-use the matmul kernel,
  // just pass in a 'batch' of inputs as an m X n matrix, to be multiplied
  // by the n x k weights, to get 'm' output images.

	matmul_cuda<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, 1 // Pass in b=1 since there is no z-dimension for linear layers
	);

  cudaDeviceSynchronize();
  return output;
}

// Binding to generate the .so file, to call from python.
PYBIND11_MODULE(cuda_layers, m) {
  m.doc() = "Implementation of forward pass of conv and linear layers in CUDA";
  m.def("conv_forward", &conv_forward, "conv_forward (CUDA)");
	m.def("linear_forward", &linear_forward, "linear_forward (CUDA)");
}
