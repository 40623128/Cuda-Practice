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
    double* a_host, *r_host;
    double* a_device, *r_device;
	//�D�����s���t
	cudaMallocHost(&a_host, N * sizeof(double));
	cudaMallocHost(&r_host, blocksPerGrid * sizeof(double));
	//��d���s���t
	cudaMalloc(&a_device, N * sizeof(double));
	cudaMalloc(&r_device, blocksPerGrid * sizeof(double));
	//�D�إͦ�
	for(int i=0;i<N;i++){
		a_host[i] = 1;
	}
	for(int i=0;i<blocksPerGrid;i++){
		r_host[i] = 0.0;
	}
	
	//�w�q��d�y
	cudaStream_t stream;
	//�Ыجy
	cudaStreamCreate(&stream);

	//�O����]�w(���B)
	cudaMemcpyAsync(a_device, a_host, N * sizeof(double), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(double), cudaMemcpyHostToDevice, stream);

	//�B��Kernel1�i��B��
	for(int i=0; i<iter; i++){
		kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
	}
	cudaDeviceSynchronize();
	
	//�D���P�]�ƶ��O����ƻs
	cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
	cout << "Ans = " << r_host[0] <<" "<< endl;
	
	//����O����
	cudaFree(r_device);
	cudaFree(a_device);
	cudaFreeHost(r_host);
	cudaFreeHost(a_host);
	return 0;
}
