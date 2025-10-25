# include <stdio.h>

typedef struct {
    float x; // 4 bytes
    float y; // 4 bytes
} Point;

int main() {
    Point p = {1.1, 2.5};
    printf("Size of point: %zu\n", sizeof(Point));
}