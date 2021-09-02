#include<iostream>
#include<time.h>
using namespace std;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<30);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;

template<unsigned int blockSize>
__device__ void warpRecude2(volatile float* s_data, int tid){ // volatile 很重要，保證s_data從相應的內從單元取出，這裡指gpu內存
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if 是防止blockSize小於64，比如blockSize為16，那會直接至
    if(blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if(blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if(blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if(blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if(blockSize >= 2) s_data[tid] += s_data[tid + 1];
}


template<unsigned int blockSize>
__global__ void reduce(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一輪迭代，有一半的線程是閒置的，現在把一個block的大小縮小一半
    unsigned int gridSize = blockSize*2*gridDim.x;
    s_data[tid] = 0;

    while (i<N){
        s_data[tid] += arr[i] + arr[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if(blockSize >= 1024){
        if(tid < 512){
            s_data[tid] += s_data[tid+512];
        }
        __syncthreads();
    }
    if(blockSize >= 512){
        if(tid < 256){
            s_data[tid] += s_data[tid+256];
        }
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128){
            s_data[tid] += s_data[tid+128];
        }
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64){
            s_data[tid] += s_data[tid+64];
        }
        __syncthreads();
    }

    if(tid < 32){
        warpRecude2<blockSize>(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

void kernel6(float* arr, float* out, int N, cudaStream_t &stream){   // 展開所有循環，去除循環
    switch(threadsPerBlock){
        case 1024:
            reduce<1024><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 512:
            reduce<512><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 256:
            reduce<256><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 128:
            reduce<128><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 64:
            reduce<64><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 32:
            reduce<32><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 16:
            reduce<16><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 8:
            reduce<8><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 4:
            reduce<4><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 2:
            reduce<2><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 1:
            reduce<1><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
    }
}

int main(){
    float* a_host, *r_host;
    float* a_device, *r_device;
    float total_time = 0.0;

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
        //kernel6<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
        kernel6(a_device, r_device, N, stream);
    }
    //紀錄停止事件(Event)
    cudaEventRecord(stop, 0);
    //等待停止事件(Event)完成
    cudaEventSynchronize(stop);
    float elapsedTime;
    //計算開始事件至暫停事件所經時間
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU Elapse time: " << elapsedTime / iters << " ms" << endl;
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
    return 0;
}
