#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_kernel(int* A, int* B, int* C, int L, int M, int N) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < L && col < N) {
        int C_idx = row*N + col;
        for (int k = 0; k < M; k++) {
            int A_idx = row*M + k;
            int B_idx = k*N + col;
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

    // Allocate memory for matrices A, B, and C on the host
    int* A = new int[L*M];
    int* B = new int[M*N];
    int* C = new int[L*N];

    // Initialize matrice A with some values
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < M; j++) {
            A[i*L + j] = i + j;            
        }
    }
    // int* A = new int[L*M]{0,1,2,3,
    //                     1,2,3,4,
    //                     2,3,4,5,
    //                     3,4,5,6};

    
    // Print input matrix A 
    std::cout << "Matrix A: " << std::endl;
    print_matrix(A, L, M);
   
    
    // Initialize matrice B with some values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[i*M + j] = i - j;
        }
    }
    // int* B = new int[M*N]{0,1,2,3};
    std::cout << "Matrix B: " << std::endl;
    print_matrix(B, M, N);
    
    // Allocate memory for matrices A, B, and C on the device
    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc((void**) &d_A, L*M*sizeof(int));
    cudaMalloc((void**) &d_B, M*N*sizeof(int));
    cudaMalloc((void**) &d_C, L*N*sizeof(int));


    // Copy matrices A and B to the device
    cudaMemcpy(d_A, A, L*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M*N*sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 2D grid
    dim3 block_dim(L,N);
    dim3 numBlocks(1,1);

    matmul_kernel<<<numBlocks, block_dim>>>(d_A, d_B, d_C, L, M, N);

    // Copy matrix C from the device to the host
    cudaMemcpy(C, d_C, L*N*sizeof(int), cudaMemcpyDeviceToHost);    

    // Print output matrix C
    std::cout << "Matrix C: " << std::endl;
    print_matrix(C, L, N);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}