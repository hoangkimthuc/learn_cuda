__global__ void kernel_function(float* data, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        data[index] *= 2.0f;
    }
}

int main() {
    float* device_data;
    int size = 1024;
    cudaMalloc(&device_data, size * sizeof(float));
    kernel_function<<<256, 256>>>(device_data, size);
    cudaFree(device_data);
    return 0;
}