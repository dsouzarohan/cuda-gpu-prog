#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/*
Question 1: The "Sepia Filter" Kernel (Indexing & Memory Layout)
Difficulty: Low-Mid Concept: 1D vs. 2D Indexing, Structs in CUDA, Global Memory Coalescing

Scenario: You are building an image processing pipeline. Images are stored in memory as a linearized 1D array of pixels, but conceptually they are 2D (Width x Height). Each pixel has 3 colors: Red, Green, and Blue.

Your task is to write a kernel that converts an RGB image into a Sepia tone.

The Math: For each pixel, the new color values are calculated as:

New_R = (0.393 * R) + (0.769 * G) + (0.189 * B)
New_G = (0.349 * R) + (0.686 * G) + (0.168 * B)
New_B = (0.272 * R) + (0.534 * G) + (0.131 * B)
(Note: You must clamp the result to a max of 255 if the calculation exceeds it).

Requirements:

Define a struct Pixel { unsigned char r, g, b; }.

Create a host array of N * M pixels (randomly initialized).

Write a kernel sepiaKernel that takes the image dimensions (width, height) and the pixel array.

Constraint: Launch the kernel using a 2D Grid and 2D Block (like you did in vector_add_gpu_3d but in 2D), but access the memory as a 1D array.

Why this helps: This reinforces the logic you learned in vector_add_indexing_notes.md regarding mapping (x, y) coordinates to a linear index i = y * width + x.

ChatGPT discussion conversation - https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce-cuda-c-and-gpu-programming/c/69512558-b0a4-8323-9e50-f98d81deebeb

Gemini conversation - https://gemini.google.com/app/94e1d7f629ce2e89
*/

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE); // Stop execution so you can see the first error
    }
}

#define PRINT_IMAGE(mat, rows, cols)                                                                 \
    for (int i = 0; i < rows; i++)                                                                   \
    {                                                                                                \
        for (int j = 0; j < cols; j++)                                                               \
        {                                                                                            \
            printf("(%d,%d,%d)    ", mat[i * cols + j].r, mat[i * cols + j].g, mat[i * cols + j].b); \
        }                                                                                            \
        printf("\n");                                                                                \
    }                                                                                                \
    printf("\n ");

struct Pixel
{
    unsigned char r, g, b; // unsigned char basically goes from 0 to 255, that is quite memory efficient for RGB intensities
};

__global__ void
sepiaKernel(Pixel *p_arr, int width, int height)
{
    // Since our kernel uses two dimensions
    // Kernel launch also expects a 2D dimension for grids and blocks
    int x = blockDim.x * blockIdx.x + threadIdx.x; // since x is fastest moving dimension, it is used for columns
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    /*
    Threads map to pixels
    Grid = image dimensions
    Block = tile of image
    */

    if (x < width && y < height)
    {
        int idx = y * width + x; // p_arr[y, x]
        unsigned char R = p_arr[idx].r;
        unsigned char G = p_arr[idx].g;
        unsigned char B = p_arr[idx].b;

        p_arr[idx].r = (unsigned char)fminf((0.393 * R) + (0.769 * G) + (0.189 * B), 255.0f);
        p_arr[idx].g = (unsigned char)fminf((0.349 * R) + (0.686 * G) + (0.168 * B), 255.0f);
        p_arr[idx].b = (unsigned char)fminf((0.272 * R) + (0.534 * G) + (0.131 * B), 255.0f);
    }
}

int main()
{
    int M = 32, N = 32;
    size_t image_size = sizeof(Pixel) * (size_t)M * (size_t)N;

    Pixel *h_pixels = (Pixel *)malloc(image_size);
    Pixel *d_pixels = nullptr;

    // Initialize pixels randomly
    for (int i = 0; i < M * N; i++)
    {
        h_pixels[i].r = rand() % 256;
        h_pixels[i].g = rand() % 256;
        h_pixels[i].b = rand() % 256;
    }

    printf("Pixel at h_pixels[0] = {%d, %d, %d} before sepia\n", h_pixels[0].r, h_pixels[0].g, h_pixels[0].b);
    // printf("\nImage after sepia\n");
    // PRINT_IMAGE(h_pixels, M, N);

    // allocate image array memory on device
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pixels, image_size));

    // copy device array
    CHECK_CUDA_ERROR(cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16); // 16 * 16 threads per block to be launched
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,  // N is width, goes across columns
        (M + blockSize.y - 1) / blockSize.y); // M is height, goes across rows

    sepiaKernel<<<gridSize, blockSize>>>(d_pixels, N, M);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check if the launch itself failed
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_pixels, d_pixels, image_size, cudaMemcpyDeviceToHost));

    printf("Pixel at h_pixels[0] = {%d, %d, %d} after sepia\n", h_pixels[0].r, h_pixels[0].g, h_pixels[0].b);

    // printf("\nImage after sepia\n");
    // PRINT_IMAGE(h_pixels, M, N);

    CHECK_CUDA_ERROR(cudaFree(d_pixels));
    free(h_pixels);
    return 0;
}