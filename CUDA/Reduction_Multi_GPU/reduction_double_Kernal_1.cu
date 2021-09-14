#include<iostream>
#include<time.h>
using namespace std;

//const int num_GPUs = 2;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
//�ۥ[�������Ӽ�(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
//const int kernal_number = 7;
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
    double* a_A_host, *r_A_host,*a_B_host,*r_B_host;
    double* a_A_device, *r_A_device, *a_B_device, *r_B_device;
    float total_time = 0.0;
    //�D�����s���t
    cudaMallocHost(&a_A_host, N * sizeof(double));
    cudaMallocHost(&a_B_host, N * sizeof(double));
    cudaMallocHost(&r_A_host, blocksPerGrid * sizeof(double));
    cudaMallocHost(&r_B_host, blocksPerGrid * sizeof(double));
    //��d���s���t
    cudaMalloc(&a_A_device, N * sizeof(double));
    cudaMalloc(&r_A_device, blocksPerGrid * sizeof(double));
    cudaMalloc(&a_B_device, N * sizeof(double));
    cudaMalloc(&r_B_device, blocksPerGrid * sizeof(double));
    //�D�إͦ�
    for(int i=0;i<N/2;i++){
        a_A_host[i] = 1;
        a_B_host[i] = 1;
    }
    for(int i=0;i<blocksPerGrid;i++){
        r_A_host[i] = 0.0;
        r_B_host[i] = 0.0;
    }
    //�w�q��d�y
    cudaStream_t streamA, streamB;
    //�Ыجy
    cudaSetDevice(0);
    cudaStreamCreate(&streamA);
    cudaSetDevice(1);
    cudaStreamCreate(&streamB);

    //�O����]�w(���B)
    cudaMemcpyAsync(a_A_device, a_A_host, N * sizeof(double), cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(r_A_device, r_A_host, blocksPerGrid * sizeof(double), cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(a_B_device, a_B_host, N * sizeof(double), cudaMemcpyHostToDevice, streamB);
    cudaMemcpyAsync(r_B_device, r_B_host, blocksPerGrid * sizeof(double), cudaMemcpyHostToDevice, streamB);
    //�w�q�P�Ыض}�l�M����ƥ�(Event)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //�����}�l�ƥ�(Event)
    cudaEventRecord(start, 0);
    //�B��Kernel1�i��B��

    for(int i=0;i<iters;i++){
        kernel1<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(a_A_device, r_A_device, N/2);
        kernel1<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(a_B_device, r_B_device, N/2);
    }

    //��������ƥ�(Event)
    cudaEventRecord(stop, 0);
    //���ݰ���ƥ�(Event)����
    cudaEventSynchronize(stop);
    float elapsedTime;
    //�p��}�l�ƥ�ܼȰ��ƥ�Ҹg�ɶ�
    cudaEventElapsedTime(&elapsedTime, start, stop);
    total_time = total_time + (elapsedTime / iters);
    //�ƥ󲾰�
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //�D���P�]�ƶ��O����ƻs
    cudaMemcpy(r_A_host, r_A_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(r_B_host, r_B_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
    //varifyOutput(r_host, a_host, N);
    //����O����
    cudaFree(r_A_device);
    cudaFree(a_A_device);
    cudaFree(r_B_device);
    cudaFree(a_B_device);
    cudaFreeHost(r_A_host);
    cudaFreeHost(a_A_host);
    cudaFreeHost(r_B_host);
    cudaFreeHost(a_B_host);
    cout << "GPU Elapse time for The Kernal 1" <<" :"<< total_time << " ms" << endl;
    total_time = 0.0 ;
    return 0;
}
