#include "omp.h"
#include <iostream>
#include <stdio.h>
using namespace std;
int main()
{
    int ar[100] = {0};
    int i,j;

    #pragma omp parallel for private(i,j)
    for( i=0;i<4;i++)
    {
        
        for( j=0;j<15;j++)
        {
            printf("th id: %d , i: %d j: %d\n ",omp_get_thread_num(),i,j) ;
            ar[j] ++;
        }
        
    }


    for(int  i=0;i<100;i++)
    {
        cout<<ar[i] << " ";
    }
}