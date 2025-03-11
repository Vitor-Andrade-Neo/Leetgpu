#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size){
    int os = input_size - kernel_size +1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float soma = 0.0;
    if(idx < os){
        for(int j = 0; j < kernel_size; j++){
            soma += input[idx + j] * kernel[j];
        }
        output[idx] = soma;
        soma = 0.0;
    }

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
