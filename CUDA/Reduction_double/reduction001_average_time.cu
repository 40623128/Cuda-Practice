#include<iostream>
#include<time.h>
using namespace std;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
//計算平均時間之執行次數
const int times_of_average = 10;

//第一種Reduction
__global__ void kernel1(double* arr, double* out, int N){
    __shared__ float s_data[threadsPerBlock];
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
    double* a_host, *r_host;
    double* a_device, *r_device;
    float total_time = 0.0;

    for(int j=0; j<times_of_average; j++){
        //主機內存分配
        cudaMallocHost(&a_host, N * sizeof(double));
        cudaMallocHost(&r_host, blocksPerGrid * sizeof(double));
        //顯卡內存分配
        cudaMalloc(&a_device, N * sizeof(double));
        cudaMalloc(&r_device, blocksPerGrid * sizeof(double));
        //題目生成
        for(int i=0;i<N;i++){
            a_host[i] = 1;
        }
        for(int i=0;i<blocksPerGrid;i++){
            r_host[i] = 0.0;
        }
        //定義顯卡流
        cudaStream_t stream;
        //創建流
        cudaStreamCreate(&stream);

        //記憶體設定(異步)
        cudaMemcpyAsync(a_device, a_host, N * sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(double), cudaMemcpyHostToDevice, stream);

        //定義與創建開始和停止事件(Event)
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //紀錄開始事件(Event)
        cudaEventRecord(start, 0);
        //運用Kernel1進行運算
        for(int i=0;i<iters;i++){
            kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
        }
        //紀錄停止事件(Event)
        cudaEventRecord(stop, 0);
        //等待停止事件(Event)完成
        cudaEventSynchronize(stop);
        float elapsedTime;
        //計算開始事件至暫停事件所經時間
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cout << "GPU Elapse time "<<j<<" : " << elapsedTime / iters << " ms" << endl;
        total_time = total_time + (elapsedTime / iters);
        //事件移除
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //主機與設備間記憶體複製
        cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
        //varifyOutput(r_host, a_host, N);
        //釋放記憶體
        cudaFree(r_device);
        cudaFree(a_device);
        cudaFreeHost(r_host);
        cudaFreeHost(a_host);
        //return 0;
}

cout << "GPU Elapse average time for " << times_of_average <<" times:"<< total_time/times_of_average << " ms" << endl;
return 0;
}
