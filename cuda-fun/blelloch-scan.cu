/*
 * Blelloch Parallel Prefix Sum (Exclusive Scan)
 * ==============================================
 * 
 * This implements the work-efficient parallel scan algorithm by Guy Blelloch.
 * It computes the prefix sum in O(n) work and O(log n) parallel steps.
 * 
 * For INCLUSIVE scan: output[i] = input[0] + input[1] + ... + input[i]
 * For EXCLUSIVE scan: output[i] = input[0] + input[1] + ... + input[i-1]  (output[0] = 0)
 * 
 * This implementation does EXCLUSIVE scan, then we convert to inclusive at the end.
 * 
 * ============================================================================
 * ALGORITHM OVERVIEW (for 8 elements)
 * ============================================================================
 * 
 * Input: [3, 1, 7, 0, 4, 1, 6, 3]
 * 
 * PHASE 1: UP-SWEEP (Reduce) - Build partial sums in a tree
 * ---------------------------------------------------------
 * We iteratively sum pairs with increasing stride (1, 2, 4, ...)
 * 
 * Level 0 (stride=1): Add pairs at distance 1
 *   Index:    0    1    2    3    4    5    6    7
 *   Before:  [3,   1,   7,   0,   4,   1,   6,   3]
 *   After:   [3,   4,   7,   7,   4,   5,   6,   9]
 *                  ^         ^         ^         ^
 *               1+3=4     0+7=7     1+4=5     3+6=9
 * 
 * Level 1 (stride=2): Add pairs at distance 2
 *   Before:  [3,   4,   7,   7,   4,   5,   6,   9]
 *   After:   [3,   4,   7,  11,   4,   5,   6,  14]
 *                            ^                   ^
 *                         4+7=11              5+9=14
 * 
 * Level 2 (stride=4): Add pairs at distance 4
 *   Before:  [3,   4,   7,  11,   4,   5,   6,  14]
 *   After:   [3,   4,   7,  11,   4,   5,   6,  25]
 *                                                ^
 *                                           11+14=25 (total sum!)
 * 
 * After up-sweep: last element contains total sum of all elements.
 * 
 * PHASE 2: DOWN-SWEEP - Distribute partial sums back down
 * --------------------------------------------------------
 * Set last element to 0 (identity), then reverse the process.
 * At each step: left child gets parent's value, right child gets left + parent.
 * 
 * Start: Set last to 0
 *   [3,   4,   7,  11,   4,   5,   6,   0]
 *                                       ^
 *                                   was 25, now 0
 * 
 * Level 2 (stride=4): Swap and add at distance 4
 *   Before:  [3,   4,   7,  11,   4,   5,   6,   0]
 *   After:   [3,   4,   7,   0,   4,   5,   6,  11]
 *                            ^                   ^
 *                    left←parent(0)    right←left+parent(11+0)
 * 
 * Level 1 (stride=2): Swap and add at distance 2
 *   Before:  [3,   4,   7,   0,   4,   5,   6,  11]
 *   After:   [3,   0,   7,   4,   4,  11,   6,  16]
 *                  ^         ^         ^         ^
 * 
 * Level 0 (stride=1): Swap and add at distance 1
 *   Before:  [3,   0,   7,   4,   4,  11,   6,  16]
 *   After:   [0,   3,   4,  11,  11,  15,  16,  22]
 *             ^    ^    ^    ^    ^    ^    ^    ^
 * 
 * Result (exclusive scan): [0, 3, 4, 11, 11, 15, 16, 22]
 * 
 * To get INCLUSIVE scan, shift left by 1 and add last input:
 *   Inclusive: [3, 4, 11, 11, 15, 16, 22, 25]
 * 
 * ============================================================================
 * COMPLEXITY
 * ============================================================================
 * - Work: O(n) - same as sequential
 * - Parallel steps: O(log n) - much better than O(n) sequential
 * - Space: O(n) - in-place possible with careful implementation
 * 
 * ============================================================================
 * LIMITATION OF THIS BASIC VERSION
 * ============================================================================
 * This implementation handles ONE BLOCK only (max 1024 elements for shared mem).
 * For larger arrays, you need multi-block scan with block-level sums.
 * We'll keep it simple here for learning.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", 
                file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// CPU reference: inclusive scan
void cpuInclusiveScan(int *input, int *output, int N)
{
    int sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += input[i];
        output[i] = sum;
    }
}

// Generate random integers in range [0, 10]
void initRandomArray(int *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr[i] = rand() % 11; // range: 0 to 10 (small values to avoid overflow)
    }
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

/*
 * Blelloch Scan Kernel (Single Block, Power-of-2 size)
 * 
 * This kernel performs inclusive scan on a single block.
 * N must be a power of 2 and <= 2 * blockDim.x (each thread handles 2 elements).
 */
__global__ void blellochScanKernel(int *d_input, int *d_output, int N)
{
    extern __shared__ int temp[]; // shared memory for scan

    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory (each thread loads 2 elements)
    int ai = tid;
    int bi = tid + (N / 2);
    
    temp[ai] = (ai < N) ? d_input[ai] : 0;
    temp[bi] = (bi < N) ? d_input[bi] : 0;

    // ==================== UP-SWEEP (Reduce) ====================
    // Build the sum tree from leaves to root
    for (int d = N >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Store the total sum, then set last element to 0 for exclusive scan
    __syncthreads();
    int totalSum = temp[N - 1];
    if (tid == 0)
    {
        temp[N - 1] = 0;
    }

    // ==================== DOWN-SWEEP ====================
    // Traverse back down the tree to build the scan
    for (int d = 1; d < N; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // ==================== CONVERT TO INCLUSIVE SCAN ====================
    // Exclusive scan: output[i] = sum of elements before i
    // Inclusive scan: output[i] = sum of elements up to and including i
    // Shift left by 1 and add last element (totalSum) to get inclusive
    if (ai < N)
    {
        // Shift: inclusive[i] = exclusive[i+1] for i < N-1
        //        inclusive[N-1] = totalSum
        if (ai < N - 1)
            d_output[ai] = temp[ai + 1];
        else
            d_output[ai] = totalSum;
    }
    if (bi < N)
    {
        if (bi < N - 1)
            d_output[bi] = temp[bi + 1];
        else
            d_output[bi] = totalSum;
    }
}

int main()
{
    // N must be power of 2 for this simple implementation
    const int N = 1024; // max for single block with 512 threads
    size_t size = N * sizeof(int);

    srand(time(NULL));

    // Host arrays
    int *h_input = (int *)malloc(size);
    int *h_output = (int *)malloc(size);
    int *h_cpu_output = (int *)malloc(size);

    // Initialize with random values
    initRandomArray(h_input, N);

    // Device arrays
    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Kernel config: N/2 threads (each handles 2 elements)
    int threadsPerBlock = N / 2;
    int sharedMemSize = N * sizeof(int);

    // Warm-up run
    blellochScanKernel<<<1, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // ==================== GPU Execution with Timing ====================
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    blellochScanKernel<<<1, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float gpu_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // ==================== CPU Execution with Timing ====================
    clock_t cpu_start = clock();
    cpuInclusiveScan(h_input, h_cpu_output, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // ==================== Print Sample Results ====================
    printf("=== Blelloch Parallel Prefix Sum (Inclusive Scan) ===\n\n");

    printf("Input (first 16):      [");
    for (int i = 0; i < 16; i++)
    {
        printf("%3d", h_input[i]);
        if (i < 15) printf(", ");
    }
    printf(", ...]\n");

    printf("GPU Output (first 16): [");
    for (int i = 0; i < 16; i++)
    {
        printf("%3d", h_output[i]);
        if (i < 15) printf(", ");
    }
    printf(", ...]\n");

    printf("CPU Output (first 16): [");
    for (int i = 0; i < 16; i++)
    {
        printf("%3d", h_cpu_output[i]);
        if (i < 15) printf(", ");
    }
    printf(", ...]\n");

    // ==================== Verification ====================
    bool correct = verifyResult(h_output, h_cpu_output, N);
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");

    // ==================== Timing Results ====================
    printf("\n--- Timing ---\n");
    printf("GPU Time (Blelloch): %.6f ms\n", gpu_time_ms);
    printf("CPU Time:            %.6f ms\n", cpu_time_ms);
    if (gpu_time_ms > 0)
        printf("Speedup:             %.2fx\n", cpu_time_ms / gpu_time_ms);

    // ==================== Memory Usage ====================
    size_t total_device_memory = 2 * size; // d_input, d_output
    printf("\n--- Memory ---\n");
    printf("N = %d elements (power of 2 required)\n", N);
    printf("Shared memory per block: %d bytes\n", sharedMemSize);
    printf("Device memory used: %zu bytes (%.2f KB)\n", 
           total_device_memory, total_device_memory / 1024.0);

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_cpu_output);

    return 0;
}
