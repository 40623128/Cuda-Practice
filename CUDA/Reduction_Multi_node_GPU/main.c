/* main.c */
#include <mpi.h>
#include<time.h>
#include <stdio.h>
void launch_multiply(const float *a, float *b);

int main (int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    const float *a;
    float *b;
    launch_multiply (a, b);
    MPI_Finalize();
    printf("finshed");
    return 1;
}
