#include <stdio.h>
#include <cuda.h>

__global__ void pi_leibniz(double *sum, long long n_terms) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (unsigned long long k = idx; k < (unsigned long long)n_terms; k += stride) {
        double sign = (k & 1ull) ? -1.0 : 1.0;   // (-1)^k
        double term = sign / (2.0 * (double)k + 1.0);
        local_sum += term;
    }

    // Accumulate each thread's partial sum into the global sum
    atomicAdd(sum, local_sum);
}

int main() {
    const long long N = 1000000000LL; // 1e9 terms (just an example)

    double *d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    int blockSize = 256;
    int numBlocks = 256; // adjust as you like

    pi_leibniz<<<numBlocks, blockSize>>>(d_sum, N);
    cudaDeviceSynchronize();

    double h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    double pi_est = 4.0 * h_sum;
    printf("Estimate of pi: %.15f\n", pi_est);

    cudaFree(d_sum);
    return 0;
}
