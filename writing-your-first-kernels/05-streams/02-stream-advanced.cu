#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdio.h>

// ChatGPT thread:
// https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce-cuda-c-and-gpu-programming/c/692aaf00-c4e0-8323-be90-e63fb6b970d9
// Discussion on CUDA streams and how streams help unblock compute (host and
// device)
// This also covers events, callbacks, pinned memory and priorities

#define CUDA_CHECK_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(err), cudaGetErrorString(err), func);
  }
}

__global__ void kernel1(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= 2.0f;
  }
}

__global__ void kernel2(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += 1.0f;
  }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status,
                                void *userData) {
  printf("Stream callback: Operation completed\n");
}

int main(void) {
  const int N = 1000000;
  size_t size = N * sizeof(float);
  float *h_data, *d_data;
  cudaStream_t stream1, stream2;
  cudaEvent_t event;
  std::cout << event << std::endl;

  // Allocate host and device memory
  CUDA_CHECK_ERROR(cudaMallocHost(
      (void **)&h_data, size)); // pinned memory, allocates memory on host but
                                // is page-locked and accessible to the device
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_data, size));

  // Initialize the data
  for (int i = 0; i < N; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // Create streams with different priorities
  int leastPriority, greatestPriority;
  CUDA_CHECK_ERROR(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  CUDA_CHECK_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking,
                                                leastPriority));
  CUDA_CHECK_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking,
                                                greatestPriority));

  // Create event
  CUDA_CHECK_ERROR(cudaEventCreate(&event));

  // Asynchronous memory copy H2D and kernel execution in stream1
  CUDA_CHECK_ERROR(
      cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
  kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_data, N);

  // Record event in stream1
  CUDA_CHECK_ERROR(cudaEventRecord(event, stream1));

  // Make stream2 wait for event
  CUDA_CHECK_ERROR(cudaStreamWaitEvent(stream2, event, 0));

  // Execute kernel in stream2
  kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_data, N);

  // Add callback to stream2
  CUDA_CHECK_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));

  // Asynchronous memory copy back to host on stream2
  CUDA_CHECK_ERROR(
      cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));

  // Synchronize streams
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream1));
  CUDA_CHECK_ERROR(cudaStreamSynchronize(stream2));

  // Verify result
  for (int i = 0; i < N; i++) {
    float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
    if (fabs(expected - h_data[i]) > 1e-5) {
      fprintf(stderr,
              "Result verification failed at element %d, expected=%f.6d, "
              "actual=%f.6d",
              i, expected, h_data[i]);
      exit(EXIT_FAILURE);
    }
  }

  printf("TEST PASSED!");

  CUDA_CHECK_ERROR(cudaFree(d_data));
  CUDA_CHECK_ERROR(cudaFreeHost(h_data));
  CUDA_CHECK_ERROR(cudaStreamDestroy(stream1));
  CUDA_CHECK_ERROR(cudaStreamDestroy(stream2));
  CUDA_CHECK_ERROR(cudaEventDestroy(event));
}
