#include <stdio.h>

void modifyInt(int x) {
    printf("Inside modifyInt: &x = %p, x = %d\n", (void*)&x, x);
    x = 99;
}

void modifyPointer(int *p) {
    printf("Inside modifyPointer:\n");
    printf("  &p = %p (address of local pointer)\n", (void*)&p);
    printf("   p = %p (value copied from caller)\n", (void*)p);
    printf("  *p = %d (value at that address)\n", *p);

    *p = 99;   // modifies what p points to
    p = NULL;  // only changes the local copy of the pointer
}

int main() {
    int a = 10;
    printf("Before call to modifyInt: &a = %p, a = %d\n", (void*)&a, a);
    modifyInt(a);
    printf("After call to modifyInt:  &a = %p, a = %d\n", (void*)&a, a);

    int b = 10;
    int *ptr = &b;

    printf("Before call to modifyPointer:\n");
    printf("  &b   = %p\n", (void*)&b);
    printf("  &ptr = %p (address of pointer variable)\n", (void*)&ptr);
    printf("   ptr = %p (value, points to a)\n", (void*)ptr);
    printf("  *ptr = %d\n\n", *ptr);

    modifyPointer(ptr);

    printf("\nAfter call to modifyPointer:\n");
    printf("  &b   = %p\n", (void*)&b);
    printf("   ptr = %p (still same)\n", (void*)ptr);
    printf("  *ptr = %d (was changed through pointer)\n", *ptr);
    return 0;
}
