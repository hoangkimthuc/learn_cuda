#include <iostream>

__device__ int get_thread_id() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void compute_thread_ids(int num_threads, int* ids) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num_threads) {
        ids[index] = get_thread_id();
    }
}

int main() {
    int threads_per_block = 256;
    int num_blocks = 18;
    int num_threads = threads_per_block * num_blocks;
    int* device_ids;
    cudaMalloc(&device_ids, num_threads * sizeof(int));
    compute_thread_ids<<<num_blocks, threads_per_block>>>(num_threads, device_ids);
    int* host_ids = new int[num_threads];
    cudaMemcpy(host_ids, device_ids, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_threads; i++) {
        std::cout << "Thread " << i << " has ID " << host_ids[i] << std::endl;
    }
    cudaFree(device_ids);
    delete[] host_ids;
    return 0;
}