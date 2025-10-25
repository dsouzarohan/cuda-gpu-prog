#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.14;
    void* vptr; // pointer to no type, needs to type casted when dereferencing

    vptr = &num;
    // printf("Pointer dereferenced: %d\n", *vptr); // this would throw an error: invalid use of void expression

    printf("Integer value is: %d\n", *(int*) vptr);

    vptr = &fnum;
    printf("Float value is: %.2f\n",*(float*) vptr);

    /*
    * Void pointers are used when we don't know the data type of the memory reference
    * fun fact: malloc() returns a void pointer but we see it as a pointer after the cast
    */
}