#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float sum;
        __shared__ float max;

        if (threadIdx.x == 0) {
        max_val = -INFINITY;
        sum = 0.0f;
    }
    __syncthreads();

    atomicMax((int*)&max_val, __float_as_int(input[idx])); // Convertendo para int para atomicMax funcionar
    __syncthreads();

        if(idx<N){
            output[i] = expf(input[idx]) - max;
            atomicAdd(&sum, output[idx]);
        }
    __syncthreads();
    if (idx < N) {
        output[idx] /= sum;
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
