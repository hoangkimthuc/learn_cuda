#include <stdio.h>

#define TILE_SIZE 32

__global__ void matrix_vector_mult(int* A, int* x, int* y, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int sum = 0;
        // Use separate j and k loop variables for clarity
        for (int j = 0, k = 0; j < M; j++, k++) {
            sum += A[i * M + j] * x[k];
            // if (k == M - 1) k = -1;  // wraparound k for next iteration
        }
        y[i] = sum;
    }
}

int main() {
    int N = 3;
    int M = 4;
    int A[N][M], x[M], y[N];
    int *d_A, *d_x, *d_y;
    int size_A = N * M * sizeof(int);
    int size_x = M * sizeof(int);
    int size_y = N * sizeof(int);

    // Initialize input matrices and vectors
    printf("Input Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = i + j;
            printf("%d ", A[i][j]);
        }
        printf("\n");
        y[i] = 0;
    }

    printf("\nInput Vector x:\n");
    for (int j = 0; j < M; j++) {
        x[j] = j;
        printf("%d\n", x[j]);
    }

    // Allocate device memory for input matrices and vectors
    cudaMalloc((void **) &d_A, size_A);
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_y, size_y);

    // Initialize output vector y to zero using cudaMemset
    cudaMemset(d_y, 0, size_y);
    
    // Copy input matrices and vectors from host memory to device memory
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);

    // Compute block dimension and grid dimension
    dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);

    // Launch kernel with specified block and grid dimensions      
    matrix_vector_mult<<<grid_dim, block_dim>>>(d_A, d_x, d_y, N, M);

    // Copy result from device memory to host memory
    cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);
    
    // Print output vector y
    printf("\nOutput Vector y:\n");
    for (int i = 0; i < N; i++) {
        printf("%d\n", y[i]);
    }

    // Verify the result
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < M; j++) {
            sum += A[i][j] * x[j];
        }
        if (y[i] != sum) {
            printf("Error: index %d, expected %d but got %d\n", i, sum, y[i]);
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}