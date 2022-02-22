/* multiply.cu */
#include<iostream>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>


using std::cout;using std::endl;

const int num_gpus = 8;
const int threadsPerBlock = 128;
const int N               = (1 <<28 )/num_gpus;
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 1;

/*
__global__ void __multiply__ (const float *a, float *b)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    b[i] *= a[i];
}
*/
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

extern "C" void launch_multiply(const float *a, const float *b)
{
    float total_time[num_gpus];
    double* a_host[num_gpus], *r_host[num_gpus];
    double* a_device[num_gpus], *r_device[num_gpus];


    //內存分配
    for(int i = 0; i < num_gpus; i++){
        //主機內存分配
        cudaMallocHost(&a_host[i], N * sizeof(double));
        cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
        //顯卡內存分配
        cudaMalloc(&a_device[i], N * sizeof(double));
        cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));
    }
    cout << "Memory Allocation Completed" << endl;



    //題目生成
    cout << "Generating list" << endl;
    for(int i = 0; i < num_gpus; i++){
        for(int j=0;j<N;j++){
            a_host[i][j] = 1.0;
            //cout <<"i =" <<i<< "; j =" <<j<< endl;
        }
        cout << "a_host "<< i <<" Generating Completed" << endl;
        for(int j=0;j<blocksPerGrid;j++){
            r_host[i][j] = 0.0;
            //cout <<"i =" <<i<< "; j =" <<j<< endl;
        }
        cout << "r_host "<< i <<" Generating Completed" << endl;
    }



    //定義顯卡流
    cudaStream_t stream[num_gpus];
    for(int i = 0; i < num_gpus; i++){
        //創建流
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    cout << "GPU Stream Define Completed" << endl;



    //記憶體設定(異步)
    for(int i = 0; i < num_gpus; i++){
        //創建流
        cudaSetDevice(i);
        cudaMemcpyAsync(a_device[i], a_host[i], N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
    }
    cout << "Memory asynchronous Completed" << endl;

    //定義開始和停止事件(Event)
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t stop_events[num_gpus];

    //創建開始和停止事件(Event)
    for(int i = 0; i < num_gpus; i++){
     cudaSetDevice(i);
     cudaEventCreate(&start_events[i]);
     cudaEventCreate(&stop_events[i]);
    }
    cout << "Create Start & Stop Event Completed" << endl;
    cout << "Start Calculation" << endl;


    for(int j=0;j<iters;j++){
        for(int i = 0; i < num_gpus; i++){
            cudaSetDevice(i);
            // In cudaEventRecord, ommit stream or set it to 0 to record
            // in the default stream. It must be the same stream as
            // where the kernel is launched.
            //紀錄開始事件(Event)
            cudaEventRecord(start_events[i], stream[0]);
            //運用Kernel1進行運算
            __multiply__ <<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(a_device[i], r_device[i], N);
            //紀錄停止事件(Event)
            cudaEventRecord(stop_events[i], stream[0]);
        }
    }
    cout << "Calculation Completed" << endl;


    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop_events[i]);
    }
    cout << "Calculation time" << endl;
    float elapsedTime[num_gpus];
    //計算開始事件至暫停事件所經時間
    for(int i = 0; i < num_gpus; i++){
        cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
        total_time[i] = total_time[i] + (elapsedTime[i] / iters);
        cout << "total_time "<< i << "= " << total_time[i] << endl;
        cout << "elapsedTime "<< i << "= " << elapsedTime[i] << endl;
    }

    for(int i = 0; i < num_gpus; i++){
        if (i ==0){
            total_time[i] = total_time[i];
        }
        else{
        total_time[i] = total_time[i-1] + total_time[i];
        }
        }


    cout << "Event Destroy" << endl;
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }

    cout << "Share Memory form Device to Host" << endl;
    //資料由顯卡記憶體傳輸至主機記憶體
    for(int i = 0; i < num_gpus; i++){
        //創建流
        cudaSetDevice(i);
        cudaMemcpy(r_host[i], r_device[i],
                   blocksPerGrid * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }

    cout << "Free Memory" << endl;
    //釋放記憶體
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaFree(r_device[i]);
        cudaFreeHost(r_host[i]);
        cudaFree(a_device[i]);
        cudaFreeHost(a_host[i]);
    }
    for(int i = 0; i < num_gpus; i++){
        cout << "GPU "<< i <<" Elapse time for The Kernal 1 :"<< total_time[i] << " ms" << endl;
        total_time[i] = 0.0 ;
        elapsedTime[i] = 0.0 ;
    }
}
