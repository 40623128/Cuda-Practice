#include<iostream>
#include<time.h>
using namespace std;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iter = 100;
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
	double* a_UnifiedMemory, *r_UnifiedMemory;
	//主機&顯卡 統一內存分配
	cudaMallocManaged(&a_UnifiedMemory, N*sizeof(double));
	cudaMallocManaged(&r_UnifiedMemory, blocksPerGrid*sizeof(double));
	
	//題目生成
	for(int i=0;i<N;i++){
		a_UnifiedMemory[i] = 1;
	}
	for(int i=0;i<blocksPerGrid;i++){
		r_UnifiedMemory[i] = 0.0;
	}
	
	//定義顯卡流
	cudaStream_t stream;
	//創建流
	cudaStreamCreate(&stream);

	//運用Kernel1進行運算
	for(int i=0; i<iter; i++){
		kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_UnifiedMemory, r_UnifiedMemory, N);
	}
	cudaDeviceSynchronize();
	
	cout << "Ans = " << r_UnifiedMemory[0] <<" "<< endl;
	
	//釋放記憶體
	cudaFree(a_UnifiedMemory);
	cudaFree(r_UnifiedMemory);
	//return 0;
	return 0;
}
