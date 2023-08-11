#include <cuda_runtime.h>
#include <iostream>

__global__
void add_tensors(float* tensor1_device, float* tensor2_device, float* out_tensor_device, int num_batches, int num_channels, int height, int width) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % num_channels;
    int b = blockIdx.z / num_channels;
    if (i < height && j < width) {
        int idx = b * num_channels * height * width + c * height * width + i * width + j;
        out_tensor_device[idx] = tensor1_device[idx] + tensor2_device[idx];
    }
}

int main() {
    int num_batches = 2;
    int num_channels = 2;
    int height = 32;
    int width = 32;

    // Allocate memory for the first 4D tensor on the host and initialize it
    float* tensor1_host = new float[num_batches * num_channels * height * width];
    for (int b = 0; b < num_batches; b++) {
        for (int c = 0; c < num_channels; c++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    tensor1_host[b * num_channels * height * width + c * height * width + i * width + j] = (float)(b * c * i * j);
                }
            }
        }
    }

    // Allocate memory for the second 4D tensor on the host and initialize it
    float* tensor2_host = new float[num_batches * num_channels * height * width];
    for (int b = 0; b < num_batches; b++) {
        for (int c = 0; c < num_channels; c++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    tensor2_host[b * num_channels * height * width + c * height * width + i * width + j] = (float)((b + 1) * (c + 1) * (i + 1) * (j + 1));
                }
            }
        }
    }

    // Allocate memory for the two 4D tensors on the device
    float* tensor1_device;
    cudaMalloc((void**) &tensor1_device, num_batches * num_channels * height * width * sizeof(float));
    float* tensor2_device;
    cudaMalloc((void**) &tensor2_device, num_batches * num_channels * height * width * sizeof(float));
    
    // Copy the data from the host to the device
    cudaMemcpy(tensor1_device, tensor1_host, num_batches * num_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor2_device, tensor2_host, num_batches * num_channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for the output 4D tensor on the device
    float* out_tensor_device;
    cudaMalloc((void**) &out_tensor_device, num_batches * num_channels * height * width * sizeof(float));

    // Launch a kernel to add the two 4D tensors element-wise on the device
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y, num_channels * num_batches);
    add_tensors<<<grid_dim, block_dim>>>(tensor1_device, tensor2_device, out_tensor_device, num_batches, num_channels, height, width);

    // Copy the data from the device to the host
    float* out_tensor_host = new float[num_batches * num_channels * height * width];
    cudaMemcpy(out_tensor_host, out_tensor_device, num_batches * num_channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the input and output tensors
    for (int b = 0; b < num_batches; b++) {
        for (int c = 0; c < num_channels; c++) {
            std::cout << "Batch " << b << ", channel " << c << std::endl;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int idx = b * num_channels * height * width + c * height * width + i * width + j;
                    std::cout << tensor1_host[idx] << " + " << tensor2_host[idx] << " = " << out_tensor_host[idx] << std::endl;
                }
            }
        }
    }
}
