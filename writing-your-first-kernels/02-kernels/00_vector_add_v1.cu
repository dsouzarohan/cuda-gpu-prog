#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000 // vector size = 100 million
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