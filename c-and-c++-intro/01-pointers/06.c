# include <stdio.h>

int main() {
    int arr1[] = {1,2,3,4};
    int arr2[] = {5,6,7,8};

    int* ptr1 = arr1;
    int* ptr2 = arr2;

    int* matrix[] = {ptr1, ptr2};

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            /**
            * Here we iterate over each of our two array pointers in matrix
            * We first print the value and the address
            * We then increment the pointer stored at matrix[i] which is basically the pointer to the first element of the ith array in matrix
            */
            printf("%d. %d is at %p\n", i, *matrix[i], matrix[i]);
            matrix[i]++; // always separate side effects from printf arguments, we can't be sure as to how C will execute the arguments
        }
    }
}