#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define STRIDE 2
#define STEP 0
template <typename T>
__global__ void add(T *a) {
    int tid = (blockIdx.x*blockDim.x + threadIdx.x)*STRIDE + STEP;
    // __shared__ T cache[BLOCK_SIZE];
    // cache[threadIdx.x] = a[tid];
    // a[tid] = cache[threadIdx.x] + 1.0f;
    a[tid] = a[tid] + 1.0f;
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