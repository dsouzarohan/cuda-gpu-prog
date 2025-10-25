// size_t = size type for memory allocation
// size_t is an unsigned integer type used to represent sizes of objects and array indexes in bytes
// It is guaranteed to be big enough to contain the size of the biggest object the host system can handle
// the size of int can vary (16-bit, 32-bit, etc.), but size_t always matches the architecture’s address space

// That means:
// On 32-bit systems, size_t can represent up to 4 GB.
// On 64-bit systems, it can represent up to 16 EB.
// This ensures functions like malloc() or sizeof() always return a type that can hold any valid memory size on that machine.

# include <stdio.h>
# include <stdlib.h>

int main() {
    int arr[] = {12,24,36,48,60,72};

    // size_t 
    size_t size = sizeof(arr) / sizeof(arr[0]); // size of the total array in memory / size of each element in memory = length of array
    printf("Size of arr: %zu\n", size); // length of 6, as we have 6 elements
    printf("Size of size_t: %zu\n", sizeof(size_t)); // How many bytes does it take to store a value of type size_t on this machine? on 64 bit, it is 8 bytes
    printf("int size in bytes: %zu\n", sizeof(int)); // int is int32, so 4 bytes

    // printf format specifiers
    // z -> size_t
    // u unsigned int
    // %zu -> size_t

    return 0;
}