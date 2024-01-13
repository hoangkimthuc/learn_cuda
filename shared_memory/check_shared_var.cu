#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 2
#define GRID_SIZE 4
__global__ void check_shared_var(int* global_var, int* shared_mem_buffer) {
    int global_tid = blockIdx.x*blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    __shared__ int shared_var[BLOCK_SIZE];
    shared_var[local_tid] = global_var[global_tid];
    if (blockIdx.x == 3) {
        shared_mem_buffer[local_tid] = shared_var[local_tid];
    }
}

int main() {
    int* h_global_var;
    int* h_shared_mem_buffer;

    h_global_var = (int*)malloc(sizeof(int)*BLOCK_SIZE*GRID_SIZE);
    h_shared_mem_buffer = (int*)malloc(sizeof(int)*BLOCK_SIZE);
    for (int i=0; i < BLOCK_SIZE*GRID_SIZE; i++) {
        h_global_var[i] = i;
        printf("global_var values: %d\n", h_global_var[i]);
    }
    int* d_global_var;
    int* d_shared_mem_buffer;
    cudaMalloc(&d_global_var, sizeof(int)*BLOCK_SIZE*GRID_SIZE);
    cudaMalloc(&d_shared_mem_buffer, sizeof(int)*BLOCK_SIZE);
    
    cudaMemcpy(d_global_var, h_global_var, sizeof(int)*BLOCK_SIZE*GRID_SIZE, cudaMemcpyHostToDevice);

    check_shared_var<<<GRID_SIZE, BLOCK_SIZE>>>(d_global_var, d_shared_mem_buffer);

    cudaMemcpy(h_shared_mem_buffer, d_shared_mem_buffer, sizeof(int)*BLOCK_SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < BLOCK_SIZE; i++) {
        std::cout << "shared_mem values: " << h_shared_mem_buffer[i] << std::endl;
    }

    cudaFree(h_global_var);
    cudaFree(h_shared_mem_buffer);
}
