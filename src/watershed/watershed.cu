#include "../../include/watershed/watershed.h"
#include"../../include/common/union_find.h"
#include<iostream>
#include<chrono>

void get_8_neighbors(int* mask, int i, int j, int rows, int cols)
{
    /*
            mask format
            0   1   2
            3   x   4
            5   6   7
    */   
    
    mask[0] = (i-1)*cols + (j-1);
    mask[1] = (i-1)*cols + (j);
    mask[2] = (i-1)*cols + (j+1);
    mask[3] = (i)*cols + (j-1);
    mask[4] = (i)*cols + (j+1);
    mask[5] = (i+1)*cols + (j-1);
    mask[6] = (i+1)*cols + (j);
    mask[7] = (i+1)*cols + (j+1);


    //test if the neighbors are out of bounds and if they are, set then to -1.
    if (i == 0)
    {
        mask[0] = mask[1] = mask[2] = -1; // Top row
    }

    if (i == rows - 1)
    {
        mask[5] = mask[6] = mask[7] = -1; // Bottom row
    }

    if (j == 0)
    {
        mask[0] = mask[3] = mask[5] = -1; // Left column
    }

    if (j == cols - 1)
    {
        mask[2] = mask[4] = mask[7] = -1; // Right column
    }

}

void get_4_neighbors(int* mask, int i, int j, int rows, int cols)
{
    /*
            mask format
                0   
            1   x   2
                3 
    */

    mask[0] = (i-1)*cols + (j);
    mask[1] = (i)*cols + (j-1);
    mask[2] = (i)*cols + (j+1);
    mask[3] = (i+1)*cols + (j);


    //test if the neighbors are out of bounds and if they are, set then to -1.
    if (i==0)
    {
        mask[0] = -1; //up
    }

    if (j == 0)
    {
         mask[1] = -1; // Left
    }

    if (j == cols - 1)
    {
        mask[2] = -1; // Right
    }

    if (i == rows - 1) 
    {
        mask[3] = -1; // Down
    }
    
    
}

void initi(int* data, int* labels, int* states, int* mask, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int p = i * cols + j;
            int min_neighbor_idx = p;
            int min_value = data[p];

            get_8_neighbors(mask, i, j, rows, cols);

            for (int q = 0; q < 8; q++)
            {
                if (mask[q] == -1) continue;
                if (data[mask[q]] < min_value)
                {
                    min_value = data[mask[q]];
                    min_neighbor_idx = mask[q];
                }
            }

            if (min_value < data[p])
            {
                labels[p] = min_neighbor_idx;
                states[p] = 0;  // downhill
            }
            else if (min_value > data[p])
            {
                labels[p] = p;
                states[p] = 1;  // local minima
            }
            else
            {
                if (min_neighbor_idx > p)
                {
                    labels[p] = min_neighbor_idx;
                    states[p] = 2;  // plateau (borrow neighbor)
                }
                else
                {
                    labels[p] = p;
                    states[p] = 3;  // plateau self
                }
            }
        }
    }
}

void plateau(int* data, int* labels, int* states, int* mask, int rows, int cols)
{
    //Variable to track whether a change has been made
    int change = 0;

    while (change==0)
    {
        
        //Reset change flag at the start of each iteration
        change = 1;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                //test for plateau pixel and/or minima.
                if (states[i*cols+j]<2 )
                {
                    continue;
                }

                //computes the neighbors of the plateau pixel (i,j).
                get_8_neighbors(mask,i,j,rows,cols);

                for (int q = 0; q < 8; q++)
                {
                    //checks for borders and minima.
                    if (mask[q]==-1 || states[mask[q]]!=0)
                    {
                        continue;
                    }
                    
                    //if a non-minimal plateua neighbor is found
                    if (data[i*cols+j] == data[mask[q]] )
                    {
                        //Update the label of the plateau pixel to be the label of the neighbor
                        labels[i*cols+j] = mask[q];

                        //Set the new state of the plateau pixel to 0
                        states[i*cols+j] = 0;

                        //Mark that a change occurred
                        change = 0;
                    }
                    
                    

                }
                 
            }
            
        }
        
    }

}


void propagation(int* labels, int rows, int cols, int RR)
{
    bool change = true;
    while (change)
    {
        change = false;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int p = i * cols + j;

                // Try to shortcut RR times
                for (int k = 0; k < RR && labels[p] != labels[labels[p]]; k++)
                {
                    labels[p] = labels[labels[p]];
                    change = true;
                }
            }
        }
    }
}


void merge(int* labels, int* states, int*mask, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (states[i*cols+j]<2)
            {
                continue;
            }

            get_8_neighbors(mask,i,j,rows,cols);

            for (int q = 0; q < 8; q++)
            {
                if (states[mask[q]]<2 || mask[q]==-1)
                {
                    continue;
                }

                union_cpu(labels,i*cols+j,mask[q]);
                
            }
            
            
        }
        
    }

    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            inline_Compress(labels,i*cols+j);
        }
        
    }

    
}

void watershed(int* data, int* labels, int rows, int cols, int iterations)
{
    int* states = (int*)malloc(rows*cols*(sizeof(int)));

    if (!states)
    {
        return;
    }

    int* mask = (int*)malloc(8*sizeof(int));

    if (!mask)
    {
        return;
    }
    //auto start = std::chrono::high_resolution_clock::now();

    initi(data,labels,states,mask,rows,cols);

    plateau(data,labels,states,mask,rows,cols);

    propagation(labels, rows, cols, iterations);

    merge(labels,states,mask,rows,cols);

    free(states);

    free(mask);

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
    

    
}
