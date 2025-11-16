#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000 // vector size = 100 million
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CUDA kernel for vector addition in 1 dimension
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// CUDA kernel for vector addition in 3 dimensions
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny,
                                  int nz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < nx && j < ny && k < nz) {
    int idx = i + j * nx + k * nx * ny;
    if (idx < nx * ny * nz) {
      c[idx] = a[idx] + b[idx];
    }
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
  float *h_arr_C_GPU_1d;
  float *h_arr_C_GPU_3d;

  size_t size = N * sizeof(float); // define the size_t type for our arrays

  // allocate memory for each of our host arrays
  h_arr_A = (float *)malloc(size);
  h_arr_B = (float *)malloc(size);
  h_arr_C_CPU = (float *)malloc(size);
  h_arr_C_GPU_1d = (float *)malloc(size);
  h_arr_C_GPU_3d = (float *)malloc(size);

  // initialize vectors using the init_vector function
  init_vector(h_arr_A, N);
  init_vector(h_arr_B, N);

  // define references for our device arrays
  float *d_arr_A;
  float *d_arr_B;
  float *d_arr_C_1d;
  float *d_arr_C_3d;

  // allocate memory on VRAM for device arrays using cudaMalloc
  cudaMalloc((void **)&d_arr_A, size);
  cudaMalloc((void **)&d_arr_B, size);
  cudaMalloc((void **)&d_arr_C_1d, size);
  cudaMalloc((void **)&d_arr_C_3d, size);

  // copy data from host to device using cudaMemcpy
  cudaMemcpy(d_arr_A, h_arr_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr_B, h_arr_B, size, cudaMemcpyHostToDevice);

  // Define grid and block dimensions for 1D
  int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
  // this is basically performing ceil using integer division
  // (N + BLOCK_SIZE - 1) / BLOCK_SIZE is equivalent to ceil(N / BLOCK_SIZE)
  // using integer division
  dim3 grid_size_1d(num_blocks_1d, 1, 1); // Grid size (how many blocks in grid)
  dim3 block_size_1d(BLOCK_SIZE_1D, 1,
                     1); // Block size (how many threads in block)

  // Defining grid and block dimensions for 3D
  int nx = 100, ny = 100, nz = 10000; // For 100M items, N = 100 * 100 * 10000

  // Block size: how many threads per block in each dimension
  dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);

  // Grid size: how many blocks needed in each dimension (using ceiling
  // division)
  dim3 grid_size_3d((nx + block_size_3d.x - 1) / block_size_3d.x,
                    (ny + block_size_3d.y - 1) / block_size_3d.y,
                    (nz + block_size_3d.z - 1) / block_size_3d.z);

  printf("Block size (threads per block) = (%d, %d, %d)\n", block_size_3d.x,
         block_size_3d.y, block_size_3d.z);
  printf("Grid size (number of blocks) = (%d, %d, %d)\n\n", grid_size_3d.x,
         grid_size_3d.y, grid_size_3d.z);

  // Warm-up runs
  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    vector_add_cpu(h_arr_A, h_arr_B, h_arr_C_CPU, N);
    vector_add_gpu_1d<<<grid_size_1d, block_size_1d>>>(d_arr_A, d_arr_B,
                                                       d_arr_C_1d, N);

    vector_add_gpu_3d<<<grid_size_3d, block_size_3d>>>(d_arr_A, d_arr_B,
                                                       d_arr_C_3d, nx, ny, nz);
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

  // Benchmarking GPU 1D implementation
  printf("Benchmarking GPU 1D...\n");
  double gpu_1d_total_time = 0.0;
  for (int i = 0; i < 100; i++) {
    cudaMemset(d_arr_C_1d, 0, size);
    double start_time = get_time();
    vector_add_gpu_1d<<<grid_size_1d, block_size_1d>>>(d_arr_A, d_arr_B,
                                                       d_arr_C_1d, N);
    cudaDeviceSynchronize(); //
    double end_time = get_time();
    gpu_1d_total_time += end_time - start_time;
    // printf("Time taken for %d iteration: %f\n", i+1, end_time - start_time);
  }
  double gpu_1d_avg_time =
      gpu_1d_total_time /
      100.0; // average time for the GPU for 20 vector additions

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
  printf("1D Speedup: %fx\n", cpu_avg_time / gpu_1d_avg_time);

  // Verify 1D results
  cudaMemcpy(h_arr_C_GPU_1d, d_arr_C_1d, size, cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < N; i++) {
    // fabs is floating absolute
    if (fabs(h_arr_C_GPU_1d[i] - h_arr_C_CPU[i]) > 1e-10) {
      correct = false;
      break;
    }
  }
  printf("1D Results are correct? %s\n", correct ? "Yes" : "No");

  // Benchmarking GPU 3D implementation
  printf("Benchmarking GPU 3D...\n");
  double gpu_3d_total_time = 0.0;
  for (int i = 0; i < 100; i++) {
    cudaMemset(d_arr_C_3d, 0, size);
    double start_time = get_time();
    vector_add_gpu_3d<<<grid_size_3d, block_size_3d>>>(d_arr_A, d_arr_B,
                                                       d_arr_C_3d, nx, ny, nz);
    cudaDeviceSynchronize(); //
    double end_time = get_time();
    gpu_3d_total_time += end_time - start_time;
    // printf("Time taken for %d iteration: %f\n", i+1, end_time - start_time);
  }
  double gpu_3d_avg_time =
      gpu_3d_total_time /
      100.0; // average time for the GPU for 20 vector additions

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
  printf("3D Speedup: %fx\n", cpu_avg_time / gpu_3d_avg_time);

  // Verify 3D results
  cudaMemcpy(h_arr_C_GPU_3d, d_arr_C_3d, size, cudaMemcpyDeviceToHost);

  correct = true;
  for (int i = 0; i < N; i++) {
    // fabs is floating absolute
    if (fabs(h_arr_C_GPU_3d[i] - h_arr_C_CPU[i]) > 1e-10) {
      correct = false;
      break;
    }
  }
  printf("3D Results are correct? %s\n", correct ? "Yes" : "No");

  // Free memory
  free(h_arr_A);
  free(h_arr_B);
  free(h_arr_C_CPU);
  free(h_arr_C_GPU_1d);
  free(h_arr_C_GPU_3d);
  cudaFree(d_arr_A);
  cudaFree(d_arr_B);
  cudaFree(d_arr_C_1d);
  cudaFree(d_arr_C_3d);

  return 0;
}