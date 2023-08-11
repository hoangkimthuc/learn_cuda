#include <iostream>
#include <cuda_runtime.h>

#define N 10

__global__ void kernel(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        a[tid] = tid;
        b[tid] = tid * tid;
        c[tid] = 0;
    }
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    kernel<<<(N+255)/256, 256, 0, stream1>>>(d_a, d_b, d_c);
    kernel<<<(N+255)/256, 256, 0, stream2>>>(d_a, d_b, d_c);

    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream2);
    cudaStreamWaitEvent(stream1, event, 0);

    cudaMemcpyAsync(a, d_a, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(b, d_b, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);

    return 0;
}