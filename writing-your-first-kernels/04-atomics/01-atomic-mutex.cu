#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>

// ChatGPT thread:
// https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce-cuda-c-and-gpu-programming/c/69240e15-8ba8-8322-90f0-e1382ee947c0
// Discussion on how atomics and mutex with in CUDA, shared between device and
// host

// Our mutex structure
struct Mutex {
  int *lock;
};

// Initialize the mutex
__host__ void initMutex(Mutex *m) {
  cudaMalloc((void **)&m->lock, sizeof(int));
  int initial = 0;
  cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// Acquire the mutex
__device__ void lock(Mutex *m) {
  while (atomicCAS(m->lock, 0, 1) != 0) {
    // Spin-wait
  }
}

// Release the mutex
__device__ void unlock(Mutex *m) { atomicExch(m->lock, 0); }

// Kernel function to demonstrate mutex usage
__global__ void mutexKernel(int *counter, Mutex *m) {
  lock(m);
  // Critical section
  int old = *counter;
  *counter = old + 1;
  unlock(m);
}

int main() {
  //   Mutex m;
  //   initMutex(&m);
  //   Always good to create host and device copies of mutexes as well

  Mutex h_m;       // struct allocated in memory, but not intialised, points to
                   // garbage values
  initMutex(&h_m); // pass a pointer to the Mutex to initMutex
  // entire object h_m now initialised and resides in host memor
  // post initMutex, h_m.lock now points to a memory in VRAM

  Mutex *d_m;
  cudaMalloc((void **)&d_m, sizeof(Mutex));
  cudaMemcpy(d_m, &h_m, sizeof(Mutex), cudaMemcpyHostToDevice);

  int *d_counter;
  cudaMalloc((void **)&d_counter, sizeof(int));
  int initial = 0;
  cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);

  // Launch the kernel
  //   mutexKernel<<<1, 1000>>>(d_counter, &m); Don't do this, this can lead to
  //   memory issues as we are passing the host mutex to the device kernel
  mutexKernel<<<1, 1000>>>(d_counter, d_m);

  int result;
  cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Counter value: %d\n", result);

  cudaFree(h_m.lock);
  cudaFree(d_m);
  cudaFree(d_counter);

  return 0;
}