#include <stdio.h> // Standard input/output header file (for printf)

int main() {
    int x = 10;
    int* ptr = &x;

    printf("Adress of x: %p\n", ptr); // ptr holds the memory address to the variable x
    printf("Value of x: %d\n", *ptr); // actual dereferencing from the pointer
}