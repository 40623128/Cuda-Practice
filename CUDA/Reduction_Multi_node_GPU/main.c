/* main.c */
#include <mpi.h>
#include<time.h>
#include <stdio.h>
#include <stdlib.h>

double *launch_multiply(const int N,const int num_node,double* node_host,int world_rank);
const int N               = (1 <<20);
const int num_node = 2;

int main (int argc, char **argv)
{
    //題目生成
    printf("Generating list\n");
    double* problem_list;
    problem_list = malloc(sizeof(double)*N);
    for(int i = 0; i < N; i++){
        problem_list[i] = 1.0+i;
    }
    printf("problem_list Generating Completed\n");


    //記憶體分配
    double* node_host[num_node];
    for(int i = 0; i < num_node; i++){
            node_host[i]= malloc(sizeof(double)*(N/num_node));
    }
    printf("Memory Allocation Completed\n");

    //題目分配
    for(int i = 0; i < num_node; i++){
        for(int j = 0; j < (N/num_node); j++){
            node_host[i][j] = problem_list[i*(N/num_node)+j];
        }
    }

    /* MPI 初始化
     * 得到當前可使用的程序數目
     * 得到當前的rank(秩)
     * 得到目前處理器的名稱
     */
    int world_size, world_rank;
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    //start Calculation
    double Ans_rank0 = 0.0;
    double Ans_rank1 = 0.0;
    if (world_rank==0){
        double *ans1 = launch_multiply(N,num_node,node_host[0],world_rank);
        //printf("world_rank %d finshed\n",world_rank);
        Ans_rank0 = ans1[0];

        /*
        for(int i = 0; i < 10; i++){

        printf("GPU_anser node 0 = %f \n",ans1[i]);
        }
        int i = 0;
        while((ans1[i]) != 0){
            Ans_rank0 = Ans_rank0 + ans1[i];
            i++;
        }
        */
        //printf("Ans_rank0 = %f \n",Ans_rank0);


        MPI_Send(&Ans_rank0, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank==1){
        double *ans2 = launch_multiply (N, num_node, node_host[1], world_rank);
        //printf("world_rank %d finshed\n",world_rank);
        //printf("GPU_anser node 1 = %f \n",ans2[0]);
        Ans_rank1 = ans2[0];

        /*
        for(int i = 0; i < 10; i++){
        //printf("GPU_anser node 1 = %f \n",ans2[i]);
        }
        int i = 0;
        while((ans2[i]) != 0){
            Ans_rank1 = Ans_rank1 + ans2[i];
            i++;
        }
        printf("Ans_rank1 = %f \n",Ans_rank1);
        */

        MPI_Recv(&Ans_rank0, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        Ans_rank1 = Ans_rank1 + Ans_rank0;
        printf("final ans = %f \n",Ans_rank1);

        //double real_ans = ((1+N)/2)*N ;
        //printf("real_ans = %f \n",real_ans);
    }
    MPI_Finalize();
    return 0;
}
