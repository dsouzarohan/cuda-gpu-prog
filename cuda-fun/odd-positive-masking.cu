#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define TILE_SIZE 16

// CPU reference implementation for verification
void cpuOddPositiveMasking(int *input, int *output, int N)
{
    // Step 1: Compute cumulative count of positives
    int *pos_cumsum = (int *)malloc(N * sizeof(int));
    int count = 0;
    for (int i = 0; i < N; i++)
    {
        if (input[i] > 0)
            count++;
        pos_cumsum[i] = count;
    }

    // Step 2: Mask and cumsum
    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        // position i is accumulated if count of positives in input[0:i] (exclusive) is odd
        if (i > 0 && pos_cumsum[i - 1] % 2 == 1)
        {
            sum += input[i];
        }
        output[i] = sum;
    }

    free(pos_cumsum);
}

// Verify GPU result against CPU reference
bool verifyResult(int *gpu_output, int *cpu_output, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (gpu_output[i] != cpu_output[i])
        {
            printf("Mismatch at index %d: GPU=%d, CPU=%d\n", i, gpu_output[i], cpu_output[i]);
            return false;
        }
    }
    return true;
}

// Generate random integers in range [-100, 100]
void initRandomArray(int *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr[i] = (rand() % 201) - 100; // range: -100 to 100
    }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code =%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplyOptimized(float *A, float *B, float *C, int M, int N, int K)
{
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

__global__ void toPositiveOrNegative(int *A, int *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        B[idx] = (A[idx] > 0) ? 1 : 0;
    }
}

__global__ void naiveCumSum(int *A, int *output, int N)
{
    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += A[i];
        output[i] = sum;
    }
}

__global__ void maskAndCumSum(int *input, int *pos_cumsum, int *output, int N)
{
    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        if (i > 0 && pos_cumsum[i - 1] % 2 == 1)
        {
            sum += input[i];
        }
        output[i] = sum;
    }
}

int main()
{
    const int N = 100000;
    size_t size = N * sizeof(int);

    // Seed random number generator
    srand(time(NULL));

    // Host arrays (use malloc for large arrays)
    int *h_input = (int *)malloc(size);
    int *h_output = (int *)malloc(size);
    int *h_cpu_output = (int *)malloc(size);

    // Initialize with random values
    initRandomArray(h_input, N);

    // Device arrays
    int *d_input, *d_pos_indicator, *d_pos_cumsum, *d_output;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pos_indicator, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pos_cumsum, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, size));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Grid and block dimensions for toPositiveOrNegative kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // ==================== GPU Execution with Timing ====================
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Step 1: Convert to positive indicators (1 if > 0, else 0)
    toPositiveOrNegative<<<numBlocks, blockSize>>>(d_input, d_pos_indicator, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Step 2: Cumulative sum of positive indicators
    naiveCumSum<<<1, 1>>>(d_pos_indicator, d_pos_cumsum, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Step 3: Mask and cumsum based on odd positive count
    maskAndCumSum<<<1, 1>>>(d_input, d_pos_cumsum, d_output, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float gpu_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // ==================== CPU Execution with Timing ====================
    clock_t cpu_start = clock();
    cpuOddPositiveMasking(h_input, h_cpu_output, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // ==================== Print Sample Results ====================
    printf("Input (first 10):      [");
    for (int i = 0; i < 10; i++)
    {
        printf("%4d", h_input[i]);
        if (i < 9)
            printf(", ");
    }
    printf(", ...]\n");

    printf("GPU Output (first 10): [");
    for (int i = 0; i < 10; i++)
    {
        printf("%4d", h_output[i]);
        if (i < 9)
            printf(", ");
    }
    printf(", ...]\n");

    printf("CPU Output (first 10): [");
    for (int i = 0; i < 10; i++)
    {
        printf("%4d", h_cpu_output[i]);
        if (i < 9)
            printf(", ");
    }
    printf(", ...]\n");

    // ==================== Verification ====================
    bool correct = verifyResult(h_output, h_cpu_output, N);
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");

    // ==================== Timing Results ====================
    printf("\n--- Timing ---\n");
    printf("GPU Time: %.4f ms\n", gpu_time_ms);
    printf("CPU Time: %.4f ms\n", cpu_time_ms);
    if (gpu_time_ms > 0)
        printf("Speedup:  %.2fx\n", cpu_time_ms / gpu_time_ms);

    // ==================== Memory Usage ====================
    size_t total_device_memory = 4 * size; // d_input, d_pos_indicator, d_pos_cumsum, d_output
    printf("\n--- Memory ---\n");
    printf("N = %d elements\n", N);
    printf("Device memory used: %zu bytes (%.2f KB, %.2f MB)\n", 
           total_device_memory, total_device_memory / 1024.0, total_device_memory / (1024.0 * 1024.0));

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    cudaFree(d_input);
    cudaFree(d_pos_indicator);
    cudaFree(d_pos_cumsum);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_cpu_output);

    return 0;
}