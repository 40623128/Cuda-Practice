#include<iostream>
#include<time.h>
using namespace std;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
//�ۥ[�������Ӽ�(2^30-3)
const int N               = (1 <<30);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;

template<unsigned int blockSize>
__device__ void warpRecude2(volatile float* s_data, int tid){ // volatile �ܭ��n�A�O��s_data�q���������q�椸���X�A�o�̫�gpu���s
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if �O����blockSize�p��64�A��pblockSize��16�A���|������
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
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3���Ĥ@�����N�A���@�b���u�{�O���m���A�{�b��@��block���j�p�Y�p�@�b
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

void kernel6(float* arr, float* out, int N, cudaStream_t &stream){   // �i�}�Ҧ��`���A�h���`��
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
        //kernel6<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a_device, r_device, N);
        kernel6(a_device, r_device, N, stream);
    }
    //��������ƥ�(Event)
    cudaEventRecord(stop, 0);
    //���ݰ���ƥ�(Event)����
    cudaEventSynchronize(stop);
    float elapsedTime;
    //�p��}�l�ƥ�ܼȰ��ƥ�Ҹg�ɶ�
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU Elapse time: " << elapsedTime / iters << " ms" << endl;
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
    return 0;
}
