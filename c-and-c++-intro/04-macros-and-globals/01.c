# include <stdio.h>

// examples for each condition macro
// #if
// #ifdef
// #ifndef
// #elif
// #else
// #endif

#define PI 3.14159 // defines a constant PI, global
#define AREA(r) (PI * r * r) // defines a lambda like function that computes area of a circle, as a macro

// if radius is not defined, this sets a value for radius
// typically used for hyperparameters or globals, creates an integer radius = 7
#ifndef radius
#define radius 7
#endif

// if elif else logic
// we can only use integer constants in #if and #elif
#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif

int main() {
    printf("Area of circle with radius %d: %f\n", radius, AREA(radius));
}
