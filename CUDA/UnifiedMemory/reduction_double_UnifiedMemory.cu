#include<iostream>
#include<time.h>
using namespace std;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
//�ۥ[�������Ӽ�(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iter = 100;
//kernel1
__global__ void kernel1(double* arr, double* out, int N){
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
    }
}

int main(){
	double* a_UnifiedMemory, *r_UnifiedMemory;
	//�D��&��d �Τ@���s���t
	cudaMallocManaged(&a_UnifiedMemory, N*sizeof(double));
	cudaMallocManaged(&r_UnifiedMemory, blocksPerGrid*sizeof(double));
	
	//�D�إͦ�
	for(int i=0;i<N;i++){
		a_UnifiedMemory[i] = 1;
	}
	for(int i=0;i<blocksPerGrid;i++){
		r_UnifiedMemory[i] = 0.0;
	}
	
	//�w�q��d�y
	cudaStream_t stream;
	//�Ыجy
	cudaStreamCreate(&stream);

	//�B��Kernel1�i��B��
	for(int i=0; i<iter; i++){
		kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_UnifiedMemory, r_UnifiedMemory, N);
	}
	cudaDeviceSynchronize();
	
	cout << "Ans = " << r_UnifiedMemory[0] <<" "<< endl;
	
	//����O����
	cudaFree(a_UnifiedMemory);
	cudaFree(r_UnifiedMemory);
	//return 0;
	return 0;
}
