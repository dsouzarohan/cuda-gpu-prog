# include <stdio.h>
# include <stdlib.h>

int main() {
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", ptr); // this prints nil
    printf("1. Initial ptr value: %p\n", (void*)ptr); // this also prints nil

    // check for NULL before using
    if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // allocate memory
    ptr = malloc(sizeof(int)); // malloc allocates some memory of size int here, and returns a void pointer to that memory address
    if (ptr == NULL) {
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("Value at ptr %p is %d\n", ptr, *(int*) ptr); // since we never assigned a value at this memory, it was initialized with 0

    return 0;
}