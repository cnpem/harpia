#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

/*
    Disjoint set data structure -> Array-based implementation

        let the dijoint set be: 0 1 2 3 4 5 6 7 8 9

        and the corresponding array be of the same size such that: 
                            0 1 2 3 4 5 6 7 8 9 (index i)
                            0 1 2 3 4 5 6 7 8 9 (array value at the given index, array[i])

        then, if we perform a union operation, that is "union(2,1)"

        the array will be: 
                            0 1 2 3 4 5 6 7 8 9 (index i)
                            0 1 1 3 4 5 6 7 8 9 (array value at the given index, array[i])
*/


__device__ __host__ inline int find(int* set, int a)
{

    while (set[a] != a)
    {
        a = set[a];
    }
    
    return a;

}


__device__ __host__ inline void compress(int* set, int a)
{
    set[a] = a;
}


__device__ __host__ inline void inline_Compress(int* set, int a)
{
    int id = a;

    while (set[a] != a)
    {
        a = set[a];
        set[id] = a;
    }
    
}

__host__ inline void union_cpu(int* set, int a, int b)
{
    a = find(set,a);
    b = find(set,b);

    if (a < b)
    {
        set[b] = a;
    }

    else if (a > b)
    {
        set[a] = b;
    }
     

}

__device__ inline void union_gpu(int* set, int a, int b)
{
    bool done = false;

    while (!done)
    {
    
        a = find(set,a);
        b = find(set,b);

        if (a < b)
        {
            int temp = atomicMin(&set[b],a);
            if (temp == b)
            {
                done = true;
            }
            
            b = temp;
        }

        else if (b < a)
        {
            int temp = atomicMin(&set[a],b);
            if (temp == a)
            {
                done = true;
            }
            a=temp;
        }

        else
        {
            done = true;
        }
        
    }
    

}

__device__ inline bool HasBit(int bitmask, int bit)
{
    return (bitmask & (1 << bit)) != 0;
}
