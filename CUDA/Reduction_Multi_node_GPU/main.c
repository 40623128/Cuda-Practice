/* main.c */
#include <mpi.h>
#include<time.h>
#include <stdio.h>
#include <stdlib.h>

double *launch_multiply(const int N,const int num_node, const int num_gpus,
                    double** node_host );
const int N               = (1 <<8);
const int num_node = 1;
const int num_gpus = 1;

int main (int argc, char **argv)
{
    //題目生成
    printf("Generating list\n");
    double* problem_list;
    problem_list = malloc(sizeof(double)*N);
    for(int i = 0; i < N; i++)
    {
        problem_list[i] = 1*i;
    }
    printf("problem_list Generating Completed\n");


    //記憶體分配
    double* node_host[num_node][num_gpus];
    for(int i = 0; i < num_node; i++)
    {
        for(int j = 0; j < num_gpus; j++)
        {
            node_host[i][j] = malloc(sizeof(double)*(N/num_gpus/num_node));
        }
    }
    printf("Memory Allocation Completed\n");


    //題目分配
    for(int i = 0; i < num_node; i++)
    {
        for(int j = 0; j < num_gpus; j++)
        {
            for(int k = 0; k < (N/num_gpus/num_node); k++)
            {
                node_host[i][j][k] = problem_list[i*(N/num_node)+(j*(N/num_gpus/num_node)+k)];
            }
        }
    }


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


    if (world_rank==0)
    {
    double *ans1 = launch_multiply(N,num_node, num_gpus,node_host[0]);
    printf("world_rank %d finshed\n",world_rank);
    printf("GPU_anser node 1 = %f \n",ans1[1]);
    double Ans = 0.0;
    int i = 0;
    while((ans1[i]) != 0)
    {
        Ans = Ans + ans1[i];
        i++;
    }
    printf("Ans = %f \n",Ans);
    }
    else
    {
    double *ans2 = launch_multiply (N,num_node, num_gpus,node_host[1]);
    printf("world_rank %d finshed\n",world_rank);
    printf("GPU_anser node 2 = %f \n",ans2[0]);
    }
    MPI_Finalize();
    return 0;
}
