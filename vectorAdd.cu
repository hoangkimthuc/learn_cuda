#include <iostream>
#include <vector>

__global__ void vectorAddKernel(int* a, int* b, int* c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void vectorAdd(int* a, int* b, int* c, int size)
{
    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

int main()
{
    constexpr int N = 1 << 20;
    constexpr size_t bytes = sizeof(int) * N;

    // Allocate host memory
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate device memory
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    cudaMalloc((void**)&dev_a, bytes);
    cudaMalloc((void**)&dev_b, bytes);
    cudaMalloc((void**)&dev_c, bytes);

    // Copy inputs to device
    cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the kernel on the device
    vectorAdd(dev_a, dev_b, dev_c, N);

    // Copy result back to host
    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Print result
    for (int i = 0; i < 10; ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}