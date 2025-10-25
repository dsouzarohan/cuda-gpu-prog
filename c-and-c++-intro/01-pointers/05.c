# include <stdio.h>

int main() {
    int arr[] = {10,20,30,40,50};
    int* ptr = arr; // in C, the pointer points to the first element of the array defined
    // so basically after defining int arr[], arr is just the pointer to the first element, as arrays get allocated on contiguous memory

    printf("Position one: %d\n", *ptr);

    for(int i = 0; i < 5; i++) {
        printf("%d is at %p\n", *ptr, ptr);
        ptr++; // C knows how to increment this pointer based on the datatype
    }

    /**
    * The pointer memory address incrememnts by 4 bytes
    * This is because we defined our array as int[] which is basically an array of int32 i.e 4 bytes
    */
}