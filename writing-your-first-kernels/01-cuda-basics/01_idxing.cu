#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector_types.h>

/*
Some helpful diagrams/representations of the thread block cluster architecture
ChatGPT thread - https://chatgpt.com/g/g-p-67d182679c9c8191b3f035753b9537ce/c/68fb70dd-7128-8323-bf71-051e275ab417

Grid
 ├── Block 0  (Shared memory, sync allowed)
 │    ├── Warp 0 (32 threads)
 │    └── Warp 1 (32 threads)
 └── Block 1
      ├── Warp 0 (32 threads)
      └── Warp 1 (32 threads)

Grid
 ├── Block(0,0)
 │    ├── Thread(0,0)
 │    ├── Thread(0,1)
 │    └── Thread(0,2)
 ├── Block(0,1)
 │    ├── Thread(0,0)
 │    ├── Thread(0,1)
 │    └── Thread(0,2)
 └── ...

Grid = 3x3 blocks
Each Block = 16x16 threads

         Grid (3x3)
      ┌───────────────┬───────────────┬───────────────┐
      │ Block(0,0)    │ Block(0,1)    │ Block(0,2)    │
      │ 16x16 threads │ 16x16 threads │ 16x16 threads │
      ├───────────────┼───────────────┼───────────────┤
      │ Block(1,0)    │ Block(1,1)    │ Block(1,2)    │
      │ 16x16 threads │ 16x16 threads │ 16x16 threads │
      ├───────────────┼───────────────┼───────────────┤
      │ Block(2,0)    │ Block(2,1)    │ Block(2,2)    │
      │ 16x16 threads │ 16x16 threads │ 16x16 threads │
      └───────────────┴───────────────┴───────────────┘

SM0 → Block(0,0), Block(0,1)
SM1 → Block(1,0), Block(1,1)
SM2 → Block(2,0), ...

SM
 ├── Warp Scheduler 0
 │     ├── executes 32 threads (1 warp)
 │     └── uses 32 CUDA cores
 ├── Warp Scheduler 1
 │     ├── executes another warp
 │     └── uses another 32 CUDA cores
 └── ...

SM
├── Shared memory + registers
├── Warp schedulers (typically 4)
│     ├── Warp 0 (32 threads)
│     ├── Warp 1 (32 threads)
│     ├── ...
│     └── Warp 47
└── 128 CUDA cores

*/

__global__ void whoami(void) {
  int block_id =
      blockIdx.x +             // apartment number on this floor (points across)
      blockIdx.y * gridDim.x + // floor number in this building (rows high)
      blockIdx.z * gridDim.x *
          gridDim.y; // building number in this city (panes deep)

  int block_offset =
      block_id * // times our apartment number
      blockDim.x * blockDim.y *
      blockDim.z; // total threads per block (people per apartment)

  int thread_offset = threadIdx.x + threadIdx.y * blockDim.x +
                      threadIdx.z * blockDim.x * blockDim.y;

  int id = block_offset +
           thread_offset; // global person id in the entire apartment complex

  printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", id,
         blockIdx.x, blockIdx.y, blockIdx.z, block_id, threadIdx.x, threadIdx.y,
         threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
  const int b_x = 2, b_y = 3, b_z = 4;
  const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so
  // we will get 2 warp of 32 threads per block

  int blocks_per_grid = b_x * b_y * b_z;
  int threads_per_block = t_x * t_y * t_z;

  printf("%d blocks/grid\n", blocks_per_grid);
  printf("%d threads/block\n", threads_per_block);
  printf("%d total threads (threads across grids)\n",
         blocks_per_grid * threads_per_block);

  dim3 blocksPerGrid(b_x, b_y, b_z);   // 3D cube of shape 2*3*4 = 24
  dim3 threadsPerBlock(t_x, t_y, t_z); // 3D cube of shape 4*4*4 = 64

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}

/*
** Logical CUDA view **

Grid (2 x 3 x 4)   // (gridDim.x, gridDim.y, gridDim.z)
├── Block (0,0,0)
│   ├── Thread(0,0,0)
│   ├── Thread(1,0,0)
│   ├── ...
│   └── Thread(3,3,3)   // last thread in this block
├── Block (1,0,0)
│   ├── Thread(0,0,0)
│   ├── ...
│   └── Thread(3,3,3)
├── Block (0,1,0)
│   ├── Thread(0,0,0)
│   ├── ...
│   └── Thread(3,3,3)
└── ...
    └── Block (1,2,3)   // last block in grid
        ├── Thread(0,0,0)
        ├── ...
        └── Thread(3,3,3)

Every block is a 4×4×4 cube of threads.
Every thread runs whoami once with its own (blockIdx, threadIdx).
So that’s the pure programming model hierarchy:
Grid → Blocks → Threads
*/

/*
** How the hardware might map it **
GPU → SMs → Blocks → Warps → Threads
Now put the hardware underneath. My 5070Ti has 70 SMs and needs to run 24 blocks
CUDA runtime might do something like this (exact mapping is up to the scheduler, but this is valid):

GPU
├── SM 0
│   ├── Block (0,0,0)    // block_id = 0
│   │   ├── Warp 0  (threads  0..31 in this block)
│   │   └── Warp 1  (threads 32..63 in this block)
│   └── Block (1,0,0)    // block_id = 1
│       ├── Warp 0
│       └── Warp 1
├── SM 1
│   ├── Block (0,1,0)    // block_id = 2
│   │   ├── Warp 0
│   │   └── Warp 1
│   └── Block (1,1,0)    // block_id = 3
│       ├── Warp 0
│       └── Warp 1
├── SM 2
│   ├── Block (0,2,0)    // block_id = 4
│   └── Block (1,2,0)    // block_id = 5
├── ...
└── SM N
    └── (extra SMs may be idle; you only have 24 blocks to schedule)

Blocks are software units, but each block is assigned wholly to one SM.
Inside an SM, each block’s 64 threads are split into:
Warp 0: thread offsets 0–31
Warp 1: thread offsets 32–63
Each warp is what actually gets scheduled on the SM’s 128 CUDA cores.

So the “full tree” conceptually is:
GPU
└── SM
    └── Blocks
        └── Warps (32 threads each)
            └── Threads (execute your kernel)
*/

/*
** example output from this kernel **
0640 | Block(0 2 1) = 10 | Thread(0 0 0) = 0
0641 | Block(0 2 1) = 10 | Thread(1 0 0) = 1
0642 | Block(0 2 1) = 10 | Thread(2 0 0) = 2
...

We know,
blocksPerGrid = (2,3,4)
block_id(0,2,1) = 10
threadsPerBlock = (4,4,4) = 64

Let's assume that SM 5 is running this block:
GPU
└── SM 5
    └── Block (0,2,1)   // block_id = 10
        ├── Warp 0  (thread_offset 0..31)
        │   ├── ThreadIdx(0,0,0) → thread_offset = 0  → id = 10*64 + 0  = 640
        │   ├── ThreadIdx(1,0,0) → thread_offset = 1  → id = 641
        │   ├── ThreadIdx(2,0,0) → thread_offset = 2  → id = 642
        │   ├── ...
        │   └── ThreadIdx(?, ?, ?) → thread_offset = 31 → id = 10*64 + 31 = 671
        └── Warp 1  (thread_offset 32..63)
            ├── ThreadIdx(?, ?, ?) → thread_offset = 32 → id = 672
            ├── ...
            └── ThreadIdx(3,3,3) → thread_offset = 63 → id = 10*64 + 63 = 703
*/

/*
** Mental model **

Grid (2 x 3 x 4)
├── Block(bx,by,bz)           // 24 of these, each on some SM
│   ├── Warp 0 (threads 0..31)
│   │   ├── Thread(tx,ty,tz)
│   │   └── ...
│   └── Warp 1 (threads 32..63)
│       ├── Thread(tx,ty,tz)
│       └── ...
└── ...

*/