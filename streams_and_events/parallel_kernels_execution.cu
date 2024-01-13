#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void myKernel(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] *= 2.0f;
    }
}

int main() {
    const int size = 1024;
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&stream[i]);
    float* hostPtr1;
    float* hostPtr2;

    cudaMallocHost(&hostPtr1, size * sizeof(float));
    cudaMallocHost(&hostPtr2, size * sizeof(float));
    float* devicePtr1;
    float* devicePtr2;

    cudaMalloc(&devicePtr1, size * sizeof(float));
    cudaMalloc(&devicePtr2, size * sizeof(float));

    // Copy hostPtr to devicePtr asynchronously
    cudaMemcpyAsync(devicePtr1, hostPtr1, size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(devicePtr2, hostPtr2, size * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    // Launch kernel asynchronously on stream[1]
    int blockSize = 32;
    int numBlocks = (size + blockSize - 1) / blockSize;
    myKernel<<<numBlocks, blockSize, 0, stream[0]>>>(devicePtr1, size);
    myKernel<<<numBlocks, blockSize, 0, stream[0]>>>(devicePtr1, size);
    cudaMemcpyAsync(hostPtr1, devicePtr1, size * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
    
    myKernel<<<numBlocks, blockSize, 0, stream[1]>>>(devicePtr2, size);
    // Copy devicePtr back to hostPtr asynchronously   
    cudaMemcpyAsync(hostPtr2, devicePtr2, size * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);
    // Wait for all operations to complete
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    // Print the data
    // for (int i = 0; i < size; ++i) {
    //     cout << hostPtr1[i] << " ";
    //     cout << hostPtr2[i] << " ";

    // cout << endl;
    // }
    // Free memory and streams
    cudaFree(devicePtr1);
    cudaFreeHost(hostPtr1);
    cudaFree(devicePtr2);
    cudaFreeHost(hostPtr2);
    for (int i = 0; i < 2; ++i)
        cudaStreamDestroy(stream[i]);
    return 0;
}