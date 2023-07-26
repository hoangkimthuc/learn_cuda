__global__ void matrixMultiply(float *A, float *B, float *C, int n, int tile_size) {
    __shared__ float sA[tile_size][tile_size];
    __shared__ float sB[tile_size][tile_size];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    float sum = 0.0f;

    // Iterate over tiles in A, B
    for (int sub = 0; sub < (n + tile_size - 1) / tile_size; ++sub) {
        if (row < n && sub * tile_size + tx < n) {
            sA[ty][tx] = A[row * n + sub * tile_size + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (col < n && sub * tile_size + ty < n) {
            sB[ty][tx] = B[(sub * tile_size + ty) * n + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < tile_size; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write result to output matrix if within range
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    int n = 1024; // Matrix size
    int tile_size = 32; // Tile size

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    // Allocate and initialize matrices on host
    hA = (float *) malloc(n * n * sizeof(float));
    hB = (float *) malloc(n * n * sizeof(float));
    hC = (float *) malloc(n * n * sizeof(float));

    for (int i = 0; i < n*n; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
        hC[i] = 0.0f;
    }

    // Allocate matrices on device
    cudaMalloc(&dA, n * n * sizeof(float));
    cudaMalloc(&dB, n * n * sizeof(float));
    cudaMalloc(&dC, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dA, hA, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set kernel dimensions
    dim3 dimBlock(tile_size, tile_size, 1);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y, 1);

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, n, tile_size);

    // Copy result matrix from device to host
    cudaMemcpy(hC, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < n*n; i++) {
        if (fabs(hC[i] - (float) n) > 1e-5) {
            printf("Error: hC[%d] = %f, expected %f\n", i, hC[i], (float) n);
            break;
        }
    }

    // Free memory
    free(hA);
    free(hB);
    free(hC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}