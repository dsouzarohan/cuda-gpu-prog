#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

// initialises vector with floats of specified size (not actually matrix)
void init_matrix(float *mat, int n) {
  for (int i = 0; i < n; i++) {
    mat[i] = (float)rand() / RAND_MAX;
  }
}

// GPU kernel for matmul
__global__ void mat_mul_gpu(float *A, float *B, float *C, int m, int n, int k) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row
  int j = blockIdx.x * blockDim.x + threadIdx.x; // col

  /*  
  CUDA treats:
  x → fast-changing dimension
  y → slow-changing dimension 
  z → slowest-changing dimension (not being used here)

  This mirrors how we index:
  columns change fastest
  rows change slower
  
  This is why we use y here for "row", that may feel counter-intuitive at first
  row major memory layout: [ row ][ column ]
                             slow     fast

  The innermost dimension (columns) must be mapped to x, because CUDA's x dimension is the fastest-moving parallel index.
  That mapping preserves:
  - coalesced memory access
  - linear memory traversal
  - warp execution efficiency
  - standard row-major indexing
  */

  if (i < m && j < n) { // ensure we are accessing the right threads
    float sum = 0.0;
    for (int p = 0; p < k; p++) {
      float a_ip = A[i * k + p]; // A[i, p]
      float b_pj = B[p * n + j]; // B[p, j]
      sum += a_ip * b_pj;
    }
    C[i * n + j] = sum; // C[i, j]
  }
}

// CPU matrix multiplication
void mat_mul_cpu(float *A, float *B, float *C, int m, int n, int k) {
  /*
   * for i in m:
   *       for j in n:
   *           C[i, j] = sum_p A[i, p] * B[p, j]
   */

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
      for (int p = 0; p < k; p++) {
        float a_ip = A[i * k + p]; // A[i, p]
        float b_pj = B[p * n + j]; // C[p, j]
        sum += a_ip * b_pj;
      }
      C[i * n + j] = sum; // C[i, j]
    }
  }

  /*Basic idea is that when treating matrices as arrays
  * To access an array as a matrix (M,N) with two loops
  * for i -> M // iterates over rows
  *   for j -> N // iterates over columns
  *     mat[i][j] is basically arr[i * N + j]
  * 
  * The column dimension/length always serves as the stride at least in this case
  * This is because when i moves to the next row, it "skips" or "passes" exactly "N" (column) in each pass
  *
  * In matrix multiplication, we use the same two loops
  * But we also need a third that iterates along the common dimension "K" for a matmul of (M,K) and (K, N)
  * For the dot product, we obviously go through row vectors from the first matrix and column vectors from the second matrix
  * So the third loop is

  * for i -> M: // iterates over rows
  *   for j -> N: // iterates over columns
  *     sum = 0
  *     for p -> K: // common dimension in (M,K) @ (K,N)
          // i here still iterates over rows; K is the column stride for matrix A as it is (M,K), p is going across the ith row (along the columns)
          A[i, p] = A[i * K + p]

          // p is used here to iterate over the row, A stays in ith row, but p goes across the jth column (along the rows)
          B[p, j] = B[p * N + j]

          // For A:
          // i is fixed → stay on the same row.
          // p runs from 0 to K-1 → move across the row (columns).
          // Access: A[i * K + p].

          // For B:
          // j is fixed → stay in the same column.
          // p runs from 0 to K-1 → move down the column (across rows).
          // sum += A[i, p] * B[p, j]
        
        C[i, j] = C[i * N + j] = sum
  */
}

// get time
double get_time() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {

  float *h_mat_A;
  float *h_mat_B;
  float *h_mat_CPU_C;
  float *h_mat_GPU_C;

  float *d_mat_A;
  float *d_mat_B;
  float *d_mat_C;

  size_t size_A = (M * K) * sizeof(float);
  size_t size_B = (K * N) * sizeof(float);
  size_t size_C = (M * N) * sizeof(float);

  // Allocate host memory
  h_mat_A = (float *)malloc(size_A);
  h_mat_B = (float *)malloc(size_B);
  h_mat_CPU_C = (float *)malloc(size_C);
  h_mat_GPU_C = (float *)malloc(size_C);

  cudaMalloc((void **)&d_mat_A, size_A);
  cudaMalloc((void **)&d_mat_B, size_B);
  cudaMalloc((void **)&d_mat_C, size_C);

  srand(time(NULL));

  init_matrix(h_mat_A, M * K);
  init_matrix(h_mat_B, K * N);

  cudaMemcpy(d_mat_A, h_mat_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat_B, h_mat_B, size_B, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // threads per block
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Warm up runs
  printf("Performing warm ups...\n\n");
  for (int i = 0; i < 3; i++) {
    mat_mul_cpu(h_mat_A, h_mat_B, h_mat_CPU_C, M, N, K);
    mat_mul_gpu<<<gridDim, blockDim>>>(d_mat_A, d_mat_B, d_mat_C, M, N, K);
    cudaDeviceSynchronize();
  }

  // Benchmarking CPU
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    mat_mul_cpu(h_mat_A, h_mat_B, h_mat_CPU_C, M, N, K);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double avg_cpu_time = cpu_total_time / 20;

  // Benchmarking GPU
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    mat_mul_gpu<<<gridDim, blockDim>>>(d_mat_A, d_mat_B, d_mat_C, M, N, K);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double avg_gpu_time = gpu_total_time / 20;

  // Print results
  printf("CPU average time: %f microseconds\n", (avg_cpu_time * 1e6f));
  printf("GPU average time: %f microseconds\n", (avg_gpu_time * 1e6f));
  printf("Speedup: %fx\n", avg_cpu_time / avg_gpu_time);

  cudaMemcpy(h_mat_GPU_C, d_mat_C, size_C, cudaMemcpyDeviceToHost);
  bool correct = true;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int idx = i * N + j;
      if (fabs(h_mat_CPU_C[idx] - h_mat_GPU_C[idx]) > 1e-5) {
        correct = false;
        break;
      }
    }
  }

  printf("MatMul Results are correct? %s\n", correct ? "Yes" : "No");

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int idx = i * N + j;
      printf("h_mat_CPU_C[%d] = %f h_mat_GPU_C[%d] = %f\n", idx,
             h_mat_CPU_C[idx], idx, h_mat_GPU_C[idx]);
    }
  }

  // Free memory
  free(h_mat_A);
  free(h_mat_B);
  free(h_mat_CPU_C);
  free(h_mat_GPU_C);
  cudaFree(d_mat_A);
  cudaFree(d_mat_B);
  cudaFree(d_mat_C);

  return 0;
}