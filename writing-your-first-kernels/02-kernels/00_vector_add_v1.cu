#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000 // vector size = 100 million
#define BLOCK_SIZE 256

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

double get_time() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  float *h_arr_A;
  float *h_arr_B;
  float *h_arr_C_CPU;
  float *h_arr_C_GPU;

  size_t size = N * sizeof(float); // define the size_t type for our arrays

  // allocate memory for each of our host arrays
  h_arr_A = (float *)malloc(size);
  h_arr_B = (float *)malloc(size);
  h_arr_C_CPU = (float *)malloc(size);
  h_arr_C_GPU = (float *)malloc(size);

  // initialize vectors using the init_vector function
  init_vector(h_arr_A, N);
  init_vector(h_arr_B, N);

  // define references for our device arrays
  float *d_arr_A;
  float *d_arr_B;
  float *d_arr_C_GPU;

  // allocate memory on VRAM for device arrays using cudaMalloc
  cudaMalloc((void **)&d_arr_A, size);
  cudaMalloc((void **)&d_arr_B, size);
  cudaMalloc((void **)&d_arr_C_GPU, size);

  // copy data from host to device using cudaMemcpy
  cudaMemcpy(d_arr_A, h_arr_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, h_arr_B, size, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // this is basically performing ceil using integer division
  // (N + BLOCK_SIZE - 1) / BLOCK_SIZE is equivalent to ceil(N / BLOCK_SIZE)
  // using integer division

  // TODO: Warm-up runs
  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    vector_add_cpu(h_arr_A, h_arr_B, h_arr_C_CPU, N);
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_arr_A, d_arr_B, d_arr_C_GPU,
                                               N);
    cudaDeviceSynchronize();
  }

  // Benchmarking CPU implementation
  printf("Benchmarking CPU...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_cpu(h_arr_A, h_arr_B, h_arr_C_CPU, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time =
      cpu_total_time / 20.0; // average time for the CPU for 20 vector additions

  // Benchmarking GPU implementation
  printf("Benchmarking GPU...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_arr_A, d_arr_B, d_arr_C_GPU,
                                               N);
    cudaDeviceSynchronize(); //
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
    // printf("Time taken for %d iteration: %f\n", i+1, end_time - start_time);
  }
  double gpu_avg_time =
      gpu_total_time / 20.0; // average time for the GPU for 20 vector additions

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  // Verify results
  cudaMemcpy(h_arr_C_GPU, d_arr_C_GPU, size, cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < N; i++) {
    // fabs is floating absolute
    if (fabs(h_arr_C_CPU[i] - h_arr_C_GPU[i]) > 1e-10) {
      correct = false;
      break;
    }
  }
  printf("Results are correct? %s\n", correct ? "Yes" : "No");

  for (int i = 0; i < 5; i++) {
    printf("h_arr_C_CPU[%d] = %f; h_arr_C_GPU[%d] = %f; abs_diff = %f\n", i,
           h_arr_C_CPU[i], i, h_arr_C_GPU[i],
           fabs(h_arr_C_CPU[i] - h_arr_C_GPU[i]));
  }
  for (int i = N - 5; i < N; i++) {
    printf("h_arr_C_CPU[%d] = %f; h_arr_C_GPU[%d] = %f; abs_diff = %f\n", i,
           h_arr_C_CPU[i], i, h_arr_C_GPU[i],
           fabs(h_arr_C_CPU[i] - h_arr_C_GPU[i]));
  }

  // Free memory
  free(h_arr_A);
  free(h_arr_B);
  free(h_arr_C_CPU);
  free(h_arr_C_GPU);
  cudaFree(d_arr_A);
  cudaFree(d_arr_B);
  cudaFree(d_arr_C_GPU);

  return 0;
}

/*

Built-in CUDA variables
These are automatically provided by CUDA for each thread:

1. blockIdx (block index)
Type: dim3
Meaning: Index of the current block within the grid
In code: blockIdx.x ranges from 0 to 39062 (since num_blocks = 39063)

2. blockDim (block dimensions)
Type: dim3
Meaning: Size of each block (number of threads per block)
In code: blockDim.x = 256 (from BLOCK_SIZE)

3. threadIdx (thread index)
Type: dim3
Meaning: Index of the current thread within its block
In code: threadIdx.x ranges from 0 to 255 (within each block)

How they combine to compute the global index

The formula int i = blockIdx.x * blockDim.x + threadIdx.x; computes the global
thread index.

Concrete examples with above values:

Example 1: First thread in first block
blockIdx.x = 0
threadIdx.x = 0
i = 0 * 256 + 0 = 0 → processes a[0] + b[0]

Example 2: Last thread in first block
blockIdx.x = 0
threadIdx.x = 255
i = 0 * 256 + 255 = 255 → processes a[255] + b[255]

Example 3: First thread in second block
blockIdx.x = 1
threadIdx.x = 0
i = 1 * 256 + 0 = 256 → processes a[256] + b[256]

Example 4: Middle thread in middle block
blockIdx.x = 100
threadIdx.x = 128
i = 100 * 256 + 128 = 25,728 → processes a[25728] + b[25728]

Example 5: Last thread in last block
blockIdx.x = 39062 (last block)
threadIdx.x = 255 (last thread)
i = 39062 * 256 + 255 = 9,999,872 + 255 = 10,000,127

Note: This last example shows why the if (i < n) check is important. With N =
10,000,000, the thread computing i = 10,000,127 is out of bounds, so the guard
prevents accessing invalid memory.

Grid (39063 blocks total):
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ ...     │ Block   │
│         │         │         │         │ 39062   │
│ 256     │ 256     │ 256     │         │ 256     │
│ threads │ threads │ threads │         │ threads │
└─────────┴─────────┴─────────┴─────────┴─────────┘
   ↑         ↑
   │         │
blockIdx.x  blockIdx.x
  = 0        = 1

Within each block:
┌─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │ ... │ 255 │
└─────┴─────┴─────┴─────┴─────┘
  ↑
  │
threadIdx.x

Grid with 5 blocks (each has 256 threads):

Block 0          Block 1          Block 2          Block 3          Block 4
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ 256     │      │ 256     │      │ 256     │      │ 256     │      │ 256     │
│ threads │      │ threads │      │ threads │      │ threads │      │ threads │
└─────────┘      └─────────┘      └─────────┘      └─────────┘      └─────────┘
     ↑                ↑                ↑                ↑                ↑
blockIdx.x=0    blockIdx.x=1    blockIdx.x=2    blockIdx.x=3    blockIdx.x=4
blockDim.x=256  blockDim.x=256  blockDim.x=256  blockDim.x=256  blockDim.x=256

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
  // blockDim.x is ALWAYS 256 for every thread
  // blockIdx.x is DIFFERENT for each block (0, 1, 2, ..., 39062)
  // threadIdx.x is DIFFERENT for each thread within a block (0, 1, 2, ..., 255)

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //        ↑            ↑            ↑
  //        varies      constant     varies
  //        (0-39062)   (256)        (0-255)
}

*/