# include <stdio.h>

int main() {
    // c-style type casting
    float f = 69.69;
    int i = (int) f;
    printf("%.2f typecased to %d\n", f, i);

    // to char type casting, as characters are technically ASCII, and there are interger to character mappings defined in ASCII
    char c = (int) i;
    printf("%d in ASCII is %c\n", i, c);
}