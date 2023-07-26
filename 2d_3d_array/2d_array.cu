#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixAdd(float* d_A, size_t pitchA, float* d_B, size_t pitchB, float* d_C, size_t pitchC, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float* row_A = (float*)((char*)d_A + row * pitchA);
        float* row_B = (float*)((char*)d_B + row * pitchB);
        float* row_C = (float*)((char*)d_C + row * pitchC);

        row_C[col] = row_A[col] + row_B[col];
    }
}

void printMatrix(float* matrix, size_t pitch, int width, int height) {
    for (int row = 0; row < height; ++row) {
        float* row_data = (float*)((char*)matrix + row * pitch);
        for (int col = 0; col < width; ++col) {
            std::cout << row_data[col] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int width = 4;
    int height = 4;
    size_t pitchA, pitchB, pitchC;
    float* d_A;
    float* d_B;
    float* d_C;

    // Allocate memory on the GPU
    cudaMallocPitch((void**)&d_A, &pitchA, width * sizeof(float), height);
    cudaMallocPitch((void**)&d_B, &pitchB, width * sizeof(float), height);
    cudaMallocPitch((void**)&d_C, &pitchC, width * sizeof(float), height);

    // Initialize host matrices
    float h_A[4][4] = { {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16} };
    float h_B[4][4] = { {1, 1, 1, 1},
                        {2, 2, 2, 2},
                        {3, 3, 3, 3},
                        {4, 4, 4, 4} };

    // Copy host matrices to device
    cudaMemcpy2D(d_A, pitchA, h_A, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitchB, h_B, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Perform matrix addition on the GPU
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    matrixAdd<<<gridSize, blockSize>>>(d_A, pitchA, d_B, pitchB, d_C, pitchC, width, height);

    // Copy result matrix from device to host
    float h_C[4][4];
    cudaMemcpy2D(h_C, width * sizeof(float), d_C, pitchC, width * sizeof(float), height, cudaMemcpyDeviceToHost);

    // Print the result matrix
    std::cout << "Result matrix:" << std::endl;
    printMatrix((float*)h_C, width * sizeof(float), width, height);

    // Free the allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}