#include<iostream>
#include<time.h>
using namespace std;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
const int N               = (1 <<30);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
//�p�⥭���ɶ������榸��
const int times_of_average = 10;

//�ĤT��Reduction
__global__ void kernel3(float* arr, float* out, int N){
    __shared__ float s_data[threadsPerBlock];
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

int main(){
    float* a_host, *r_host;
    float* a_device, *r_device;
    float total_time = 0.0;
    for(int j=0;j<times_of_average;j++){
        //�D�����s���t
        cudaMallocHost(&a_host, N * sizeof(float));
        cudaMallocHost(&r_host, blocksPerGrid * sizeof(float));
        //��d���s���t
        cudaMalloc(&a_device, N * sizeof(float));
        cudaMalloc(&r_device, blocksPerGrid * sizeof(float));
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
        cudaMemcpyAsync(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(r_device, r_host, blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice, stream);

        //�w�q�P�Ыض}�l�M����ƥ�(Event)
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //�����}�l�ƥ�(Event)
        cudaEventRecord(start, 0);
        //�B��Kernel1�i��B��
        for(int i=0;i<iters;i++){
            kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
        }
        //��������ƥ�(Event)
        cudaEventRecord(stop, 0);
        //���ݰ���ƥ�(Event)����
        cudaEventSynchronize(stop);
        float elapsedTime;
        //�p��}�l�ƥ�ܼȰ��ƥ�Ҹg�ɶ�
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cout << "GPU Elapse time "<<j<<" : " << elapsedTime / iters << " ms" << endl;
        total_time = total_time + (elapsedTime / iters);
        //�ƥ󲾰�
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        //�D���P�]�ƶ��O����ƻs
        cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
        //varifyOutput(r_host, a_host, N);
        //����O����
        cudaFree(r_device);
        cudaFree(a_device);
        cudaFreeHost(r_host);
        cudaFreeHost(a_host);
        //return 0;
}

cout << "GPU Elapse average time for " << times_of_average <<" times:"<< total_time/times_of_average << " ms" << endl;
return 0;
}
