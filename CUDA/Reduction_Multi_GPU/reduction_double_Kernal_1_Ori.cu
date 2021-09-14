#include<iostream>
#include<time.h>
using namespace std;

//const int num_GPUs = 2;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
//const int kernal_number = 7;
//kernel1
__global__ void kernel1(double* arr, double* out, int N){
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

int main(){
    double* a_A_host, *r_A_host;
    double* a_A_device, *r_A_device;
    float total_time = 0.0;
    //主機內存分配
    cudaMallocHost(&a_A_host, N * sizeof(double));
    cudaMallocHost(&r_A_host, blocksPerGrid * sizeof(double));
    //顯卡內存分配
    cudaMalloc(&a_A_device, N * sizeof(double));
    cudaMalloc(&r_A_device, blocksPerGrid * sizeof(double));
    //題目生成
    for(int i=0;i<N;i++){
        a_A_host[i] = 1;
    }
    for(int i=0;i<blocksPerGrid;i++){
        r_A_host[i] = 0.0;
    }
    //定義顯卡流
    cudaStream_t streamA;
    //創建流
    cudaSetDevice(0);
    cudaStreamCreate(&streamA);

    //記憶體設定(異步)
    cudaMemcpyAsync(a_A_device, a_A_host, N * sizeof(double), cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(r_A_device, r_A_host, blocksPerGrid * sizeof(double), cudaMemcpyHostToDevice, streamA);
    //定義與創建開始和停止事件(Event)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //紀錄開始事件(Event)
    cudaEventRecord(start, 0);
    //運用Kernel1進行運算

    for(int i=0;i<iters;i++){
        kernel1<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(a_A_device, r_A_device, N);
    }

    //紀錄停止事件(Event)
    cudaEventRecord(stop, 0);
    //等待停止事件(Event)完成
    cudaEventSynchronize(stop);
    float elapsedTime;
    //計算開始事件至暫停事件所經時間
    cudaEventElapsedTime(&elapsedTime, start, stop);
    total_time = total_time + (elapsedTime / iters);
    //事件移除
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //主機與設備間記憶體複製
    cudaMemcpy(r_A_host, r_A_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
    //varifyOutput(r_host, a_host, N);
    //釋放記憶體
    cudaFree(r_A_device);
    cudaFree(a_A_device);
    cudaFreeHost(r_A_host);
    cudaFreeHost(a_A_host);
    cout << "GPU Elapse time for The Kernal 1" <<" :"<< total_time << " ms" << endl;
    total_time = 0.0 ;
    return 0;
}
