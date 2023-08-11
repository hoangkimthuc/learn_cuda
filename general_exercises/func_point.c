#include <stdio.h>
int Add(int a, int b)
{
    return a + b;
}

int main()
{
    int (*p)(int, int);
    p = Add;
    int c = p(2,3);
    printf("%d\n", c);
    return 0;
}