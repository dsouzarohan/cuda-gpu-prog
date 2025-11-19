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

  if (i < m && j < n) { // ensure we are accessing the right threads
    double sum = 0.0;
    for (int p = 0; p < k; p++) {
      double a_rp = A[i * k + p]; // A[i, p]
      double b_cp = B[p * n + j]; // B[p, j]
      sum += a_rp * b_cp;
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
      double sum = 0.0;
      for (int p = 0; p < k; p++) {
        double a_ip = A[i * k + p]; // A[i, p]
        double b_pj = B[p * n + j]; // C[p, j]
        sum += a_ip * b_pj;
      }
      C[i * n + j] = sum; // C[i, j]
    }
  }
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