# include <cuda.h>
# include <stdio.h>

__global__ void vector_add(int* a, int* b, int* c, int N) {
    int i = threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // printf("Vector addition on GPU");
    int h_arr_A[] = {1,2,3,4,5};
    int h_arr_B[] = {2,4,5,6,10};
    int* h_arr_C;

    size_t size = sizeof(h_arr_A) / sizeof(h_arr_A[0]);
    printf("Size of arrays is %zu\n", size);

    // remember that with CUDA, we always work with two memory spaces, Host and Device
    // we can't just use the "host" memory directly on the GPU
    // we must create and manage a device side pointer via a device pointer, the below are the same

    int* d_arr_A;
    int* d_arr_B;
    int* d_arr_C; // device pointer

    // cudaMalloc also expects a pointer to this pointer, &d_arr that is the address of the pointer variable
    // so CUDA can fill it with the GPU memory address
    // so d_arr_A will eventually point to memory in VRAM, which is why we can't really even derefence it here, will lead to a segfault

    // printf("cudaMalloc d_arr_A");
    cudaMalloc((void**) &d_arr_A, size * sizeof(int));
    // printf("cudaMalloc d_arr_B");
    cudaMalloc((void**) &d_arr_B, size * sizeof(int));
    // printf("cudaMalloc d_arr_C");
    cudaMalloc((void**) &d_arr_C, size * sizeof(int));


    // we use cudaMemcpy to copy from device to host, and vice versa
    cudaMemcpy(d_arr_A, h_arr_A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, h_arr_B, size * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid_size(1);
    dim3 block_size(size);

    // printf("Launching kernel vector_add");
    vector_add<<<grid_size, block_size>>>(d_arr_A, d_arr_B, d_arr_C, size);

    cudaDeviceSynchronize();

    h_arr_C = (int*) malloc(size * sizeof(int));

    // printf("Copying back to host");
    cudaMemcpy(h_arr_C, d_arr_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    // printf("Freeing from GPU\n");
    cudaFree(d_arr_A); cudaFree(d_arr_B); cudaFree(d_arr_C);

    for(int i = 0; i < size; i++) {
        printf("Value at %p is %d\n", h_arr_C, h_arr_C[i]);
    }
    return 0;
}