#include <stdio.h>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 8

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void async_MatMul(const Matrix A, const Matrix B, Matrix C)
{    
    // cudaStream_t stream[2];
    // for (int i = 0; i < 2; ++i)
    //     cudaStreamCreate(&stream[i]);

    // Load A1 and B1 to device memory
    Matrix d_A1;
    d_A1.width = A.width; d_A1.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A1.elements, size);
    cudaMemcpyAsync(d_A1.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B1;
    d_B1.width = B.width; d_B1.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B1.elements, size);
    cudaMemcpyAsync(d_B1.elements, B.elements, size,
               cudaMemcpyHostToDevice);
    
    // Allocate C1 in device memory
    Matrix d_C1;
    d_C1.width = C.width; d_C1.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C1.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A1, d_B1, d_C1);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A1, d_B1, d_C1);   

    // Read C from device memory
    cudaMemcpyAsync(C.elements, d_C1.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A1.elements);
    cudaFree(d_B1.elements);
    cudaFree(d_C1.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

int main()
{
    Matrix A, B, C;
    A.width = 128;
    A.height = 128;
    B.width = 128;
    B.height = 128;
    C.width = 128;
    C.height = 128;
    A.elements = (float*)malloc(A.width * A.height * sizeof(float));
    B.elements = (float*)malloc(B.width * B.height * sizeof(float));
    C.elements = (float*)malloc(C.width * C.height * sizeof(float));
    for (int i = 0; i < A.width * A.height; i++)
    {
        A.elements[i] = 2;
        B.elements[i] = 2;
    }
    async_MatMul(A, B, C);
}
