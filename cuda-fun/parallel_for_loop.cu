#include <cuda.h>
#include <cuda_runtime_api.h>
# include <stdio.h>

// ChatGPT - https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce-cuda-c-and-gpu-programming/c/68fc9aeb-042c-8322-9c37-cf9d17a2b9ab

// GPU implementation
__global__ void increment_gpu(int *a, int N) {
    int i = threadIdx.x;
    if (i < N) {
        a[i] = a[i] + 1;
    }
}

// CPU implementation
void increment_cpu(int *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 1;
    }
}

int main() {
    const int N = 5;
    int h_arr[] = {2,4,6,8,10};

    printf("h_arr: %p\n", h_arr); // first value of h_arr is a point
    printf("h_arr: %d\n", *h_arr); // first value of h_arr
    
    // allocate arrays in device memory
    int* d_arr;

    printf("d_arr before cudaMalloc: %p\n", d_arr); // first value of d_arr is a point
    printf("*d_arr before cudaMalloc: %d\n", *d_arr); // first value of d_arr

    cudaMalloc((void**) &d_arr, N * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(N);

    // increment_cpu(h_arr, N); // on CPU

    increment_gpu<<<grid_size, block_size>>>(d_arr, N);

    // copy back from device to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    for(int i = 0; i < N; i++) {
        printf("Value at %d is %d\n", i, h_arr[i]);
    }
 
    return 0;
}