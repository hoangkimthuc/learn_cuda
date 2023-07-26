#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_kernel(int* A, int* B, int* C, int L, int M, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < L && j < N) {
        int C_idx = i*N + j;
        for (int k = 0; k < M; k++) {
            int A_idx = i*M + k;
            int B_idx = k*N + j;
            C[C_idx] += A[A_idx] * B[B_idx];
        }
    }
}

void print_matrix(int* matrix, int r, int c) {
    for (int row = 0; row < r; row++) {
        for (int col = 0; col < c; col++) {
            std::cout << matrix[(row*c + col)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int L = 2; // the size of the matrices
    int M = 3;
    int N = 4;

    // Allocate memory for matrices A, B, and C in unified memory
    int* A;
    cudaMallocManaged(&A, L*M*sizeof(int));
    int* B;
    cudaMallocManaged(&B, M*N*sizeof(int));
    int* C;
    cudaMallocManaged(&C, L*N*sizeof(int));

    // Initialize matrice A with some values
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < M; j++) {
            A[i*L + j] = i + j;            
        }
    }
   
    std::cout << "Matrix A: " << std::endl;
    print_matrix(A, L, M);   
    
    // Initialize matrice B with some values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[i*M + j] = i - j;
        }
    }
    std::cout << "Matrix B: " << std::endl;
    print_matrix(B, M, N);

    // Launch the kernel with 2D grid
    dim3 block_dim(L,N);
    dim3 numBlocks(1,1);

    matmul_kernel<<<numBlocks, block_dim>>>(A, B, C, L, M, N);
    cudaDeviceSynchronize();
    // Copy matrix C from the device to the host
    // Print output matrix C
    std::cout << "Matrix C: " << std::endl;
    print_matrix(C, L, N);

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}