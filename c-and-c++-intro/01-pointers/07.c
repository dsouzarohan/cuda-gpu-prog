#include <stdio.h>
#include <stdlib.h>

void goodAlloc(int** out, size_t bytes) {
    *out = (int*)malloc(bytes);     // writes into caller's pointer
}

void badAlloc(int* out, size_t bytes) {
    out = (int*)malloc(bytes);      // only changes a local copy; caller sees no change
}

int main() {
    int *p = NULL, *q = NULL;

    goodAlloc(&p, 4 * sizeof(int));
    badAlloc(q, 4 * sizeof(int));

    printf("p %s NULL\n", p ? "!=" : "==");  // prints: p != NULL
    printf("q %s NULL\n", q ? "!=" : "==");  // prints: q == NULL

    free(p);
    return 0;
}
