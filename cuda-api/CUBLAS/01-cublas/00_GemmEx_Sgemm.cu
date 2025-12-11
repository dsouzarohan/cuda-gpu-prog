// dedicated for small handwritten matrices
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <library_types.h>
#include <stdio.h>

// ChatGPT thread -
// https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce/c/6933d8c2-39fc-8320-8187-c650a71da5fd
// Explains a little about the CUBLASS ecosystem and the difference between
// GemmEx and SGEMM

#define M 3
#define K 4
#define N 2

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUBLAS_CHECK(call)                                                     \
  {                                                                            \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols)                                          \
  for (int i = 0; i < rows; i++) {                                             \
    for (int j = 0; j < cols; j++) {                                           \
      printf("%8.3f ", mat[i * cols + j]);                                     \
    }                                                                          \
    printf("\n");                                                              \
  }                                                                            \
  printf("\n ");

void cpu_matmul(float *A, float *B, float *C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j]; // A[i, k] * B[k, j]
      }
      C[i * N + j] = sum;
    }
  }
}

int main() {
  float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float C_cpu[M * N], C_cublas_s[M * N], C_cublas_gemmex[M * N];

  // CPU matmul
  cpu_matmul(A, B, C_cpu);

  // CUDA setup
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));

  CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // row major A =
  // 1.0 2.0 3.0 4.0
  // 5.0 6.0 7.0 8.0

  // col major A =
  // 1.0 5.0
  // 2.0 6.0
  // 3.0 7.0
  // 4.0 8.0

  // memory layout (row)
  // 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

  // memory layout (col)
  // 1.0 5.0 2.0 6.0 3.0 7.0 4.0 8.0

  // the trick to have CUBLASS compute row ordered matrices -
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  // cuBLAS SGEMM
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_B, N, d_A, K, &beta, d_C, N));
  CUDA_CHECK(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // cuBLAS GemmEx: FP16 inputs, FP32 accumulaet, FP32 output
  half *d_A_gemmex, *d_B_gemmex;
  CUDA_CHECK(cudaMalloc(&d_A_gemmex, M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_B_gemmex, K * N * sizeof(half)));

  // Convert to half precision on CPU
  half A_h[M * K], B_h[K * N];
  for (int i = 0; i < M * K; i++)
    A_h[i] = __float2half(A[i]);
  for (int i = 0; i < K * N; i++)
    B_h[i] = __float2half(B[i]);

  // Copy half precision data to device
  CUDA_CHECK(cudaMemcpy(d_A_gemmex, A_h, M * K * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_gemmex, B_h, K * N * sizeof(half),
                        cudaMemcpyHostToDevice));

  // result matrix C for GemmEx will be float32 on device
  float *d_C_gemmex;
  CUDA_CHECK(cudaMalloc(&d_C_gemmex, M * N * sizeof(float)));

  // scalars for GemmEx
  float alpha_gemmex = 1.0f, beta_gemmex = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_gemmex, d_B_gemmex,
      CUDA_R_16F, N, d_A_gemmex, CUDA_R_16F, K, &beta_gemmex, d_C_gemmex,
      CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Copy result back to host and convert to float
  CUDA_CHECK(cudaMemcpy(&C_cublas_gemmex, d_C_gemmex, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Print results
  printf("Matrix A (%dx%d):\n", M, K);
  PRINT_MATRIX(A, M, K);
  printf("Matrix B (%dx%d):\n", K, N);
  PRINT_MATRIX(B, K, N);
  printf("CPU Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cpu, M, N);
  printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cublas_s, M, N);
  printf("cuBLAS GemmEx Result (%dx%d):\n", M, N);
  PRINT_MATRIX(C_cublas_gemmex, M, N);

  // Clean up
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_A_gemmex));
  CUDA_CHECK(cudaFree(d_B_gemmex));
  CUDA_CHECK(cudaFree(d_C_gemmex));
  CUBLAS_CHECK(cublasDestroy(handle));

  return 0;
}