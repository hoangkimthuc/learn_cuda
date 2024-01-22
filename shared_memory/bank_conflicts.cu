#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define STRIDE 2

template <typename T>
__global__ void add(T *a) {
    int global_tid = (blockIdx.x*blockDim.x + threadIdx.x);
    int local_tid = (threadIdx.x);
    __shared__ T shared_mem[BLOCK_SIZE*STRIDE]; 
    shared_mem[local_tid*STRIDE] = a[global_tid]; // One bank conflict for coping data from global to shared memory
    __syncthreads();
    a[global_tid] = shared_mem[local_tid*STRIDE] + 1; // One bank conflict for reading data from shared memory
}
template <typename T>
void runTest() {
    T *a_h, *a_d;
    
    // allocate memory on host and device
    a_h = (T *)malloc(sizeof(T)*BLOCK_SIZE*STRIDE);
    cudaMalloc((void **)&a_d, sizeof(T)*BLOCK_SIZE*STRIDE);
    // initialize host array and copy it to CUDA device
    for (int i = 0; i < BLOCK_SIZE*STRIDE; i++) {
        a_h[i] = (T)i;
    }
    cudaMemcpy(a_d, a_h, sizeof(T)*BLOCK_SIZE*STRIDE, cudaMemcpyHostToDevice);
    // launch kernel
    add<<<1,BLOCK_SIZE>>>(a_d);
    // copy results back to host
    cudaMemcpy(a_h, a_d, sizeof(T)*BLOCK_SIZE*STRIDE, cudaMemcpyDeviceToHost);

    // free memory
    free(a_h);
    cudaFree(a_d);
}
int main() {
    runTest<float>();
    return 0;
}