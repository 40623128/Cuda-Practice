/* multiply.cu */
#include<iostream>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
const int threadsPerBlock = 128;
const int iters           = 1;


/*
//Reduction 001
__global__ void __multiply__(double *arr, double *out, int N){
    __shared__ double s_data[threadsPerBlock];
    //�C�ӽu�{Ū���@�Ӥ���
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();
    //s = 1 2 4 8 16 32 64
    for(int s = 1; s < blockDim.x; s*=2){
        if(tid % (2*s) == 0 && i + s <N){
            s_data[tid] +=
            s_data[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        out[blockIdx.x] = s_data[0];
        printf("s_data[0] = %f\n",s_data[0]);
    }
}
*/


//Reduction 002
__global__ void __multiply__(double *arr, double *out, int N){
    __shared__ double s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
        printf("s_data[0] = %f\n",s_data[0]);
    }
}






extern "C" double *launch_multiply(const int N ,const int num_node,
                                   double *node_host,int world_rank)
{
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    for(int i=0;i<num_gpus;i++) {
    // Query the device properties.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device id: %d\n", i);
    printf("Device name: %s\n", prop.name);
    //printf("Node %d Device id: %d\n", world_rank, i);
    //printf("Node %d Device name: %s\n", world_rank, prop.name);
    }

    const int Gpu_N = N/num_node/num_gpus;
    const int blocksPerGrid   = (Gpu_N + threadsPerBlock - 1)/threadsPerBlock;

    printf("Node %d\n"
            "N = %d\n"
            "num_node = %d\n"
            "num_gpus = %d\n"
            "node_host = %f\n",
            world_rank,N,num_node,num_gpus,node_host[0]);

    /*  ���s���t
     *      �D�����s���t
     *      ��d���s���t
     */
    float total_time[num_gpus];
    double *r_host[num_gpus],*a_host[num_gpus];
    double *a_device[num_gpus], *r_device[num_gpus];
    for(int i = 0; i < num_gpus; i++){
        cudaMallocHost(&a_host[i], Gpu_N * sizeof(double));
        cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
        cudaMalloc(&a_device[i], Gpu_N * sizeof(double));
        cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));
    }
    printf("Node %d Memory Allocation Completed\n",world_rank);


    //�D�إͦ��P���t
    printf("Node %d Generating list\n",world_rank);
    for(int i = 0; i < num_gpus; i++){
        for(int j=0;j<blocksPerGrid;j++){
            r_host[i][j] = 0.0;
        }
        for(int j=0;j<Gpu_N;j++){
            a_host[i][j] = node_host[i*Gpu_N+j];
            }
    }


    //�w�q��d�y
    cudaStream_t stream[num_gpus];
    for(int i = 0; i < num_gpus; i++){
        //�Ыجy
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    printf("ERROR_cudaStreamCreate = %s\n",cudaGetErrorString(cudaGetLastError()));
    printf("Node %d GPU Stream Define Completed\n",world_rank);

    //�O����]�w(���B)
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaMemcpyAsync(a_device[i], a_host[i], Gpu_N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
    }
    printf("Node %d Memory asynchronous Completed\n",world_rank);

    //�w�q�}�l�M����ƥ�(Event)
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t stop_events[num_gpus];


    //�Ыض}�l�M����ƥ�(Event)
    for(int i = 0; i < num_gpus; i++){
     cudaSetDevice(i);
     cudaEventCreate(&start_events[i]);
     cudaEventCreate(&stop_events[i]);
    }
    printf("Node %d Create Start & Stop Event Completed\n",world_rank);


    printf("Node %d Start Calculation\n",world_rank);
    for(int i = 0; i < num_gpus; i++){
        /* �]�wdevice
        * �����}�l�ƥ�(Event)
        * �B��__multiply__�i��B��
        * ��������ƥ�(Event)
        */
        cudaSetDevice(i);
        cudaEventRecord(start_events[i], stream[i]);
        __multiply__ <<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(a_device[i], r_device[i], Gpu_N);
        cudaEventRecord(stop_events[i], stream[i]);
        cudaDeviceSynchronize();
        printf("Node %d GPU %d ERROR = %s\n",world_rank,i,cudaGetErrorString(cudaGetLastError()));
        cudaEventSynchronize(stop_events[i]);
    }
    printf("Node %d Calculation Completed\n",world_rank);



    /*
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop_events[i]);
    }
    printf("Node %d Calculation time\n",world_rank);
    */

    float elapsedTime[num_gpus];
    //�p��}�l�ƥ�ܼȰ��ƥ�Ҹg�ɶ�
    for(int i = 0; i < num_gpus; i++){
        cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
        total_time[i] = total_time[i] + (elapsedTime[i] / iters);
        //printf("total_time %d = %f\n",i, total_time[i]);
        //printf("elapsedTime %d = %f\n",i, elapsedTime[i]);
    }


    for(int i = 0; i < num_gpus; i++){
        if (i ==0){
            total_time[i] = total_time[i];
        }
        else{
        total_time[i] = total_time[i-1] + total_time[i];
        }
        }


    printf("Node %d Event Destroy\n",world_rank);
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }

    printf("Node %d Share Memory form Device to Host\n",world_rank);
    //��ƥ���d�O����ǿ�ܥD���O����
    for(int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        //cudaMemcpy(r_host[i], r_device[i],blocksPerGrid * sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(r_host[i], r_device[i],blocksPerGrid * sizeof(double),cudaMemcpyDeviceToHost,stream[i]);
        printf("ERROR_cudaMemcpyAsync Node %d GPU %d ERROR = %s\n",world_rank,i,cudaGetErrorString(cudaGetLastError()));
        printf("Node %d r_host[%d][0] = %f\n",world_rank,i, r_host[i][0]);
    }

    printf("Node %d Free Memory\n",world_rank);
    //����O����
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaFree(r_device[i]);
        //cudaFreeHost(r_host[i]);
        cudaFree(a_device[i]);
        //cudaFreeHost(a_host[i]);
    }


    for(int i = 0; i < num_gpus; i++){
        /*
        printf("Node %d GPU %d Elapse time for The __multiply__ : %f ms\n",
        world_rank,i, total_time[i]);
        */
        total_time[i] = 0.0 ;
        elapsedTime[i] = 0.0 ;
    }

    for(int i = 0; i < num_gpus; i++){
        for(int j = 0; j < blocksPerGrid; j++){
            if (i == 0 && j == 0){
            r_host[0][0] = r_host[i][j];
            }
            else if (r_host[i][j] != 0){
            r_host[0][0] = r_host[0][0] + r_host[i][j];
            }
            printf("Node %d r_host[%d][%d] = %f\n", world_rank, i, j, r_host[i][j]);
            printf("Node %d Ans [%d][%d] = %f\n", world_rank, i, j, r_host[0][0]);
        }
    }
    printf("r_host[0][0] = %f\n",r_host[0][0]);

    return *r_host;
}
