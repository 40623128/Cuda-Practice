#include<iostream>
#include<time.h>
using namespace std;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
const int kernal_number = 7;

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

//kernel2
__global__ void kernel2(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N){
        s_data[tid] = arr[i];
    }
    __syncthreads();
    // s = 1 2 4 8 16 32 64
    for(int s = 1; s < blockDim.x; s*=2){
        int index = tid * 2 * s;
        if((index + s) < blockDim.x && (blockIdx.x * blockDim.x + index + s) < N){
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

//kernel3
__global__ void kernel3(double* arr, double* out, int N){
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
    }
}

//kernel4
__global__ void kernel4(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一?迭代，有一半的?程是idle的，?在把一?block的大小?小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???行原?的第一?迭代，后面代?不用?
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

//kernel5
__device__ void warpRecude(volatile double* s_data, int tid){ // volatile ??字很重要，保?s_data?相?的?存?元取出，?里??指gpu?存
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}
__global__ void kernel5(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一?迭代，有一半的?程是idle的，?在把一?block的大小?小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???行原?的第一?迭代，后面代?不用?
    }else{
        s_data[tid] = 0;
    }
    __syncthreads();

    for(int s = blockDim.x/2; s > 32; s>>=1){
        if(tid < s && i + s < N){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32){
        warpRecude(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}

//kernel6
template<unsigned int blockSize>
__device__ void warpRecude06(volatile double* s_data, int tid){ // volatile ??字很重要，保?s_data?相?的?存?元取出，?里??指gpu?存
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if 是防止blockSize小于64，比如blockSize?16，那么?直接到下面
    if(blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if(blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if(blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if(blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if(blockSize >= 2) s_data[tid] += s_data[tid + 1];
}
template<unsigned int blockSize>
__global__ void reduce06(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3的第一?迭代，有一半的?程是idle的，?在把一?block的大小?小一半
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???行原?的第一?迭代，后面代?不用?
    }else{
        s_data[tid] = 0;
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
        warpRecude06<blockSize>(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}
void kernel6(double* arr, double* out, int N, cudaStream_t &stream){   // 展?所有的循?，去除循?
    switch(threadsPerBlock){
        case 1024:
            reduce06<1024><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 512:
            reduce06<512><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 256:
            reduce06<256><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 128:
            reduce06<128><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 64:
            reduce06<64><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 32:
            reduce06<32><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 16:
            reduce06<16><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 8:
            reduce06<8><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 4:
            reduce06<4><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 2:
            reduce06<2><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 1:
            reduce06<1><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
    }
}


//kernel7
template<unsigned int blockSize>
__device__ void warpRecude07(volatile double* s_data, int tid){ // volatile 很重要，保證s_data從相應的內從單元取出，這裡指gpu內存
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if 是防止blockSize小於64，比如blockSize為16，那會直接至
    if(blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if(blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if(blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if(blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if(blockSize >= 2) s_data[tid] += s_data[tid + 1];
}
template<unsigned int blockSize>
__global__ void reduce07(double* arr, double* out, int N){
    __shared__ double s_data[threadsPerBlock];
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
        warpRecude07<blockSize>(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}
void kernel7(double* arr, double* out, int N, cudaStream_t &stream){   // 展開所有循環，去除循環
    switch(threadsPerBlock){
        case 1024:
            reduce07<1024><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 512:
            reduce07<512><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 256:
            reduce07<256><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 128:
            reduce07<128><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 64:
            reduce07<64><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 32:
            reduce07<32><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 16:
            reduce07<16><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 8:
            reduce07<8><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 4:
            reduce07<4><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 2:
            reduce07<2><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
        case 1:
            reduce07<1><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(arr, out, N);break;
    }
}


int main(){
    double* a_host, *r_host;
    double* a_device, *r_device;
    float total_time = 0.0;
    for(int k=0; k<kernal_number; k++){
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
            if (k == 0){
            for(int i=0;i<iters;i++){
                kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
            }
            }
            if (k == 1){
            for(int i=0;i<iters;i++){
                kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
            }
            }
            if (k == 2){
            for(int i=0;i<iters;i++){
                kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
            }
            }
            if (k == 3){
            for(int i=0;i<iters;i++){
                kernel4<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
            }
            }
            if (k == 4){
            for(int i=0;i<iters;i++){
                kernel5<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
            }
            }
            if (k == 5){
            for(int i=0;i<iters;i++){
                kernel6(a_device, r_device, N, stream);
            }
            }
            if (k == 6){
            for(int i=0;i<iters;i++){
                kernel7(a_device, r_device, N, stream);
            }
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
            cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
            //varifyOutput(r_host, a_host, N);
            //釋放記憶體
            cudaFree(r_device);
            cudaFree(a_device);
            cudaFreeHost(r_host);
            cudaFreeHost(a_host);
            cout << "The Kernal" << k+1 <<" times:"<< endl;
            cout << "GPU Elapse time for The Kernal " << k+1 <<" :"<< total_time << " ms" << endl;
            total_time = 0.0 ;
    }
return 0;
}
