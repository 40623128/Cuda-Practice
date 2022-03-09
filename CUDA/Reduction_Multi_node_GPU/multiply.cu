/* multiply.cu */
#include<iostream>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>
const int threadsPerBlock = 128;
const int iters           = 1;

/*
//Reduction 001
__global__ void __multiply__(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
    //每個線程讀取一個元素
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
    }
}
*/

//Reduction 002
__global__ void __multiply__(double* arr, double* out, int N){
    __shared__ float s_data[threadsPerBlock];
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
    }
}






extern "C" double *launch_multiply(const int N ,const int num_node, const int num_gpus,double** a_host)
{
    const int Gpu_N = N/num_node/num_gpus;
    const int blocksPerGrid   = (Gpu_N + threadsPerBlock - 1)/threadsPerBlock;


    //data check
    for(int i = 0; i < 1; i++)
    {
        printf("a_host %d = %f\n",
           i,a_host[0][i]);
    }
    printf("num_node = %d\n"
            "num_gpus = %d\n"
            "a_host = %f\n",
           num_node,num_gpus,*a_host[2]);


    float total_time[num_gpus];
    double *r_host[num_gpus];
    double *a_device[num_gpus], *r_device[num_gpus];
    //內存分配
    for(int i = 0; i < num_gpus; i++){
        //主機內存分配
        //cudaMallocHost(&a_host[i], Gpu_N * sizeof(double));
        cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
        //顯卡內存分配
        cudaMalloc(&a_device[i], Gpu_N * sizeof(double));
        cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));
    }
    printf("Memory Allocation Completed\n");


    //題目生成
    printf("Generating list\n");
    for(int i = 0; i < num_gpus; i++){
        for(int j=0;j<blocksPerGrid;j++){
            r_host[i][j] = 0.0;
            //printf("i = %d ;j = %d\n",i,j);
        }
        printf("r_host %d Generating Completed\n",i);
    }


    //定義顯卡流
    cudaStream_t stream[num_gpus];
    for(int i = 0; i < num_gpus; i++){
        //創建流
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    printf("GPU Stream Define Completed\n");


    //記憶體設定(異步)
    for(int i = 0; i < num_gpus; i++){
        //創建流
        cudaSetDevice(i);
        cudaMemcpyAsync(a_device[i], a_host[i], Gpu_N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
    }
    printf("Memory asynchronous Completed\n");

    //定義開始和停止事件(Event)
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t stop_events[num_gpus];

    //創建開始和停止事件(Event)
    for(int i = 0; i < num_gpus; i++){
     cudaSetDevice(i);
     cudaEventCreate(&start_events[i]);
     cudaEventCreate(&stop_events[i]);
    }
    printf("Create Start & Stop Event Completed\n");


    printf("Start Calculation\n");
    for(int j=0;j<iters;j++){
        for(int i = 0; i < num_gpus; i++){
            cudaSetDevice(i);
            // In cudaEventRecord, ommit stream or set it to 0 to record
            // in the default stream. It must be the same stream as
            // where the kernel is launched.
            //紀錄開始事件(Event)
            cudaEventRecord(start_events[i], stream[0]);
            //運用Kernel1進行運算
            __multiply__ <<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(a_device[i], r_device[i], Gpu_N);
            //紀錄停止事件(Event)
            cudaEventRecord(stop_events[i], stream[0]);
        }
    }
    printf("Calculation Completed\n");


    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop_events[i]);
    }
    printf("Calculation time\n");


    float elapsedTime[num_gpus];
    //計算開始事件至暫停事件所經時間
    for(int i = 0; i < num_gpus; i++){
        cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
        total_time[i] = total_time[i] + (elapsedTime[i] / iters);
        printf("total_time %d = %f\n",i, total_time[i]);
        printf("elapsedTime %d = %f\n",i, elapsedTime[i]);
    }


    for(int i = 0; i < num_gpus; i++){
        if (i ==0){
            total_time[i] = total_time[i];
        }
        else{
        total_time[i] = total_time[i-1] + total_time[i];
        }
        }


    printf("Event Destroy\n");
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }


    printf("Share Memory form Device to Host\n");
    //資料由顯卡記憶體傳輸至主機記憶體
    for(int i = 0; i < num_gpus; i++)
    {
        //創建流
        cudaSetDevice(i);
        cudaMemcpy(r_host[i], r_device[i],blocksPerGrid * sizeof(double),
                   cudaMemcpyDeviceToHost);

        for(int j = 0; j < 256; j++)
        {
            printf("r_host %d %d = %f\n",i,j,r_host[i][j]);
        }
    }

    printf("Free Memory\n");
    //釋放記憶體
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaFree(r_device[i]);
        //cudaFreeHost(r_host[i]);
        cudaFree(a_device[i]);
        //cudaFreeHost(a_host[i]);
    }


    for(int i = 0; i < num_gpus; i++){
        printf("GPU %d Elapse time for The Kernal 1 : %f ms\n",i, total_time[i]);
        total_time[i] = 0.0 ;
        elapsedTime[i] = 0.0 ;
    }
    return *r_host;
}
