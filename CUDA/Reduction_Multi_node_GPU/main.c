/* main.c */
#include <mpi.h>
#include<time.h>
#include <stdio.h>
void launch_multiply(const int num_node, const int num_gpus);
const int N               = (1 <<25 );
const int num_node = 1;
const int num_gpus = 2;

int main (int argc, char **argv)
{
    int world_size, world_rank;
    //MPI 初始化
    MPI_Init (&argc, &argv);

    //得到當前可使用的程序數目
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    //printf("world_size = %d/n",world_size);

    //得到當前的rank(秩)
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    //printf("world_rank = %d/n",world_rank);

    //得到目前處理器的名稱
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    //印出以下資料
    printf("Hello world from processor %s,"
            "rank %d out of %d processors\n",
           processor_name,world_rank,world_size);

    if (world_rank==0){

    launch_multiply (num_node, num_gpus);
    printf("world_rank %d finshed\n",world_rank);
    }
    else{
    launch_multiply (num_node, num_gpus);
    printf("world_rank %d finshed\n",world_rank);
    }
    MPI_Finalize();
    return 0;
}
