#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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

// Functor: convert to positive indicator (1 if > 0, else 0)
struct positive_indicator
{
    __host__ __device__
    int operator()(const int &x) const
    {
        return (x > 0) ? 1 : 0;
    }
};

// Functor: apply mask based on previous cumsum being odd
// Takes (input_value, prev_pos_cumsum) and returns masked value
struct apply_odd_mask
{
    __host__ __device__
    int operator()(const thrust::tuple<int, int> &t) const
    {
        int input_val = thrust::get<0>(t);
        int prev_cumsum = thrust::get<1>(t);
        // If previous cumsum is odd, include this value; else 0
        return (prev_cumsum % 2 == 1) ? input_val : 0;
    }
};

int main()
{
    const int N = 100000;
    size_t size = N * sizeof(int);

    // Seed random number generator
    srand(time(NULL));

    // Host arrays
    int *h_input = (int *)malloc(size);
    int *h_output = (int *)malloc(size);
    int *h_cpu_output = (int *)malloc(size);

    // Initialize with random values
    initRandomArray(h_input, N);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // ==================== GPU Execution with Thrust ====================
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Copy input to device using Thrust
    thrust::device_vector<int> d_input(h_input, h_input + N);
    
    // Step 1: Convert to positive indicators (1 if > 0, else 0)
    thrust::device_vector<int> d_pos_indicator(N);
    thrust::transform(d_input.begin(), d_input.end(), d_pos_indicator.begin(), positive_indicator());

    // Step 2: Cumulative sum of positive indicators (parallel prefix sum!)
    thrust::device_vector<int> d_pos_cumsum(N);
    thrust::inclusive_scan(d_pos_indicator.begin(), d_pos_indicator.end(), d_pos_cumsum.begin());

    // Step 3: Create shifted cumsum (prev_cumsum[i] = cumsum[i-1], with prev_cumsum[0] = 0)
    thrust::device_vector<int> d_prev_cumsum(N);
    d_prev_cumsum[0] = 0;
    thrust::copy(d_pos_cumsum.begin(), d_pos_cumsum.end() - 1, d_prev_cumsum.begin() + 1);

    // Step 4: Apply mask - keep value if prev_cumsum is odd, else 0
    thrust::device_vector<int> d_masked(N);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_input.begin(), d_prev_cumsum.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_input.end(), d_prev_cumsum.end())),
        d_masked.begin(),
        apply_odd_mask()
    );

    // Step 5: Final cumulative sum of masked values
    thrust::device_vector<int> d_output(N);
    thrust::inclusive_scan(d_masked.begin(), d_masked.end(), d_output.begin());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float gpu_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back to host
    thrust::copy(d_output.begin(), d_output.end(), h_output);

    // ==================== CPU Execution with Timing ====================
    clock_t cpu_start = clock();
    cpuOddPositiveMasking(h_input, h_cpu_output, N);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // ==================== Print Sample Results ====================
    printf("=== Odd Positive Masking with Thrust ===\n\n");
    
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
    printf("GPU Time (Thrust): %.4f ms\n", gpu_time_ms);
    printf("CPU Time:          %.4f ms\n", cpu_time_ms);
    if (gpu_time_ms > 0)
        printf("Speedup:           %.2fx\n", cpu_time_ms / gpu_time_ms);

    // ==================== Memory Usage ====================
    // d_input, d_pos_indicator, d_pos_cumsum, d_prev_cumsum, d_masked, d_output = 6 arrays
    size_t total_device_memory = 6 * size;
    printf("\n--- Memory ---\n");
    printf("N = %d elements\n", N);
    printf("Device memory used: %zu bytes (%.2f KB, %.2f MB)\n", 
           total_device_memory, total_device_memory / 1024.0, total_device_memory / (1024.0 * 1024.0));

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_input);
    free(h_output);
    free(h_cpu_output);

    return 0;
}
