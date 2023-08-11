#include <stdio.h>


__global__ void matrix_vector_mult(int* x, int* y, int* z, int vector_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vector_len) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int N = 10;
    int x[N], y[N], z[N];
    int *d_z, *d_x, *d_y;
    int size = N * sizeof(int);
    

    // Initialize input matrices and vectors
    printf("Input vectors:\n");
    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = i;
        printf("%d %d\n", x[i], y[i]);
    }

    // Allocate device memory for vectors
    cudaMalloc((void **) &d_z, size);
    cudaMalloc((void **) &d_x, size);
    cudaMalloc((void **) &d_y, size);

    // Initialize output vector y to zero using cudaMemset
    cudaMemset(d_z, 0, size);
    
    // Copy input matrices and vectors from host memory to device memory
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    

    // Launch kernel with specified block and grid dimensions      
    matrix_vector_mult<<<1, N>>>(d_x, d_y, d_z, N);

    // Copy result from device memory to host memory
    cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);
    
    // Print output vector y
    printf("\nOutput Vector z:\n");
    for (int i = 0; i < N; i++) {
        printf("%d\n", z[i]);
    }

        // Free device memory
    cudaFree(d_z);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}