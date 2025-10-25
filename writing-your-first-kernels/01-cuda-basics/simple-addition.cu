# include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
# include <stdio.h>
#include <vector_types.h>

// ChatGPT - https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce/c/68fc9aeb-042c-8322-9c37-cf9d17a2b9ab

__global__ void add_one(int *a) {
    int i = threadIdx.x; // not really needed here
    a[i] = a[i] + 1;
}

int main() {
    int num = 5; // will copy this to GPU and add one to it
    int* d_num; // device pointer

    // run malloc on the GPU, and whatever address is given, assign it to *d_num
    cudaMalloc((void**) &d_num, sizeof(int));
    
    // pick the value stored at &num and store it at the address d_num
    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);
    printf("address in GPU is %p", d_num);

    add_one<<<1,1>>>(d_num);
    cudaDeviceSynchronize();

    cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(&d_num);

    printf("value incremented %d", num);
    return 0;
}