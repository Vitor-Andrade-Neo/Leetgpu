#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include "solve.h"


#define BLOCK_DIM 1024
#define CORSE_FACTOR 2


__global__ void  SharedMemoryReduction(const float* input, float* output, int size){
     __shared__ float sdata[BLOCK_DIM];

     int idx = blockIdx.x * blockDim.x * CORSE_FACTOR + threadIdx.x;
     int tid = threadIdx.x;
    
     float sum = 0.0f;
     for(unsigned int tile = 0; tile < CORSE_FACTOR; tile++){
        int index = idx + tile * blockDim.x;
        if(index < size){
            sum += input[index];
        }
        __syncthreads();
     }

     sdata[tid] = sum;
     __syncthreads();

     for(unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
     }

     if(tid == 0){
        atomicAdd(output, sdata[0]);
     }


}


void solve(const float* input, float* output, int N) {
    const int size  = N;
    unsigned int blockspergrid = (size + BLOCK_DIM * CORSE_FACTOR - 1)/(CORSE_FACTOR * BLOCK_DIM);
    SharedMemoryReduction<<<blockspergrid, BLOCK_DIM>>>(input, output, size);
    cudaDeviceSynchronize();
}
