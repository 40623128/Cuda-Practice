#include<iostream>
#include<time.h>
using namespace std;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<30);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
//計算平均時間之執行次數
const int times_of_average = 10;

//第一種Reduction
__global__ void kernel2(float* arr, float* out, int N) {
    __shared__ float s_data[threadsPerBlock];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    s_data[tid] = arr[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
        s_data[index] += s_data[index + s];
    }
    __syncthreads();
}

    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = s_data[0];
}

int main(){
    float* a_host, *r_host;
    float* a_device, *r_device;
    float total_time = 0.0;

    for(int j=0; j<times_of_average; j++){
        //主機內存分配
        cudaMallocHost(&a_host, N * sizeof(float));
        cudaMallocHost(&r_host, blocksPerGrid * sizeof(float));
        //顯卡內存分配
        cudaMalloc(&a_device, N * sizeof(float));
        cudaMalloc(&r_device, blocksPerGrid * sizeof(float));
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
        cudaMemcpyAsync(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice, stream);

        //定義與創建開始和停止事件(Event)
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //紀錄開始事件(Event)
        cudaEventRecord(start, 0);
        //運用Kernel1進行運算
        for(int i=0;i<iters;i++){
            kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
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
        cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
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
