# include <cuda.h>
# include <cuda_runtime.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

__global__ void vector_add(int* a, int* b, int* c, int N) {
    // Calculate global thread index across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// CPU version of vector addition
void vector_add_cpu(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Use a large array size for meaningful benchmarking
    const size_t size = 999999999; // ~1B elements
    printf("Vector addition benchmark (CPU vs GPU)\n");
    printf("Array size: %zu elements\n\n", size);

    // Allocate host memory
    int* h_arr_A = (int*) malloc(size * sizeof(int));
    int* h_arr_B = (int*) malloc(size * sizeof(int));
    int* h_arr_C_gpu = (int*) malloc(size * sizeof(int));
    int* h_arr_C_cpu = (int*) malloc(size * sizeof(int));

    // Initialize arrays with some values
    for (size_t i = 0; i < size; i++) {
        h_arr_A[i] = i % 100;
        h_arr_B[i] = (i * 2) % 100;
    }

    // ========== CPU Benchmark ==========
    printf("Running CPU version...\n");
    clock_t cpu_start = clock();
    vector_add_cpu(h_arr_A, h_arr_B, h_arr_C_cpu, size);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU time: %.6f seconds\n\n", cpu_time);

    // ========== GPU Benchmark ==========
    printf("Running GPU version...\n");
    
    // Allocate device memory
    int* d_arr_A;
    int* d_arr_B;
    int* d_arr_C;

    cudaMalloc((void**) &d_arr_A, size * sizeof(int));
    cudaMalloc((void**) &d_arr_B, size * sizeof(int));
    cudaMalloc((void**) &d_arr_C, size * sizeof(int));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data to device
    cudaMemcpy(d_arr_A, h_arr_A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_B, h_arr_B, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    // Note: For large arrays, we need to use multiple blocks
    // Each block can have up to 1024 threads (on most GPUs)
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_size(num_blocks);
    dim3 block_size(threads_per_block);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    vector_add<<<grid_size, block_size>>>(d_arr_A, d_arr_B, d_arr_C, size);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate GPU time
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    double gpu_time = gpu_time_ms / 1000.0; // Convert to seconds

    // Copy result back to host
    cudaMemcpy(h_arr_C_gpu, d_arr_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_arr_A);
    cudaFree(d_arr_B);
    cudaFree(d_arr_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("GPU time: %.6f seconds\n\n", gpu_time);

    // ========== Results ==========
    printf("========== Benchmark Results ==========\n");
    printf("CPU time: %.6f seconds\n", cpu_time);
    printf("GPU time: %.6f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("========================================\n\n");

    // Verify results match (check first and last few elements)
    printf("Verifying results (first 5 and last 5 elements):\n");
    int verify_count = 0;
    for (int i = 0; i < 5; i++) {
        if (h_arr_C_cpu[i] == h_arr_C_gpu[i]) {
            printf("  [%d] CPU: %d, GPU: %d ✓\n", i, h_arr_C_cpu[i], h_arr_C_gpu[i]);
            verify_count++;
        } else {
            printf("  [%d] CPU: %d, GPU: %d ✗ MISMATCH!\n", i, h_arr_C_cpu[i], h_arr_C_gpu[i]);
        }
    }
    for (int i = size - 5; i < size; i++) {
        if (h_arr_C_cpu[i] == h_arr_C_gpu[i]) {
            printf("  [%d] CPU: %d, GPU: %d ✓\n", i, h_arr_C_cpu[i], h_arr_C_gpu[i]);
            verify_count++;
        } else {
            printf("  [%d] CPU: %d, GPU: %d ✗ MISMATCH!\n", i, h_arr_C_cpu[i], h_arr_C_gpu[i]);
        }
    }

    // Free host memory
    free(h_arr_A);
    free(h_arr_B);
    free(h_arr_C_gpu);
    free(h_arr_C_cpu);

    return 0;
}

