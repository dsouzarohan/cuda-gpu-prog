#include <cuda_runtime.h>
#include <stdio.h>

// ChatGPT thread:
// https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce-cuda-c-and-gpu-programming/c/692aaf00-c4e0-8323-be90-e63fb6b970d9
// Discussion on CUDA streams and how streams help unblock compute (host and
// device)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(err), cudaGetErrorString(err), func);
  }
}

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    C[idx] = A[idx] + B[idx];
  }
}

int main(void) {
  int numElements = 50000;
  size_t size = sizeof(float) * numElements;
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  cudaStream_t stream1, stream2; // defines our cudaStreams

  // Allocate host memory
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initializing host array
  for (int i = 0; i < numElements; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate device memory
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

  // Create streams
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

  // H2D
  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

  // make sure d_B is copied before launching kernel that uses it
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

  // Launch kernels
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C,
                                                            numElements);

  // copy back result to host async
  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

  // synchronize streams
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

  // Verify result
  for (int i = 0; i < numElements; i++) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED!\n");

  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  free(h_A);
  free(h_B);
  free(h_C);
}