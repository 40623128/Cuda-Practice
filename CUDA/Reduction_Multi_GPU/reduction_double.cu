#include<iostream>
#include<time.h>
using namespace std;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
//�ۥ[�������Ӽ�(2^30-3)
const int N               = (1 <<28);
const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
const int iters           = 100;
const int kernal_number = 7;

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
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3���Ĥ@?���N�A���@�b��?�{�Oidle���A?�b��@?block���j�p?�p�@�b
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???���?���Ĥ@?���N�A�Z���N?����?
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
__device__ void warpRecude(volatile double* s_data, int tid){ // volatile ??�r�ܭ��n�A�O?s_data?��?��?�s?�����X�A?��??��gpu?�s
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
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3���Ĥ@?���N�A���@�b��?�{�Oidle���A?�b��@?block���j�p?�p�@�b
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???���?���Ĥ@?���N�A�Z���N?����?
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
__device__ void warpRecude06(volatile double* s_data, int tid){ // volatile ??�r�ܭ��n�A�O?s_data?��?��?�s?�����X�A?��??��gpu?�s
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if �O����blockSize�p�_64�A��pblockSize?16�A���\?������U��
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
    unsigned int i = threadIdx.x + blockIdx.x * (blockDim.x * 2); // 3���Ĥ@?���N�A���@�b��?�{�Oidle���A?�b��@?block���j�p?�p�@�b
    if(i < N){
        s_data[tid] = arr[i] + arr[i + blockDim.x];  // ???���?���Ĥ@?���N�A�Z���N?����?
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
void kernel6(double* arr, double* out, int N, cudaStream_t &stream){   // �i?�Ҧ����`?�A�h���`?
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
__device__ void warpRecude07(volatile double* s_data, int tid){ // volatile �ܭ��n�A�O��s_data�q���������q�椸���X�A�o�̫�gpu���s
    if(blockSize >= 64) s_data[tid] += s_data[tid + 32];   // if �O����blockSize�p��64�A��pblockSize��16�A���|������
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
        warpRecude07<blockSize>(s_data, tid);
    }

    if(tid == 0){
        out[blockIdx.x] = s_data[0];
    }
}
void kernel7(double* arr, double* out, int N, cudaStream_t &stream){   // �i�}�Ҧ��`���A�h���`��
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

            //�w�q�P�Ыض}�l�M����ƥ�(Event)
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            //�����}�l�ƥ�(Event)
            cudaEventRecord(start, 0);
            //�B��Kernel1�i��B��
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
            cudaMemcpy(r_host, r_device, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
            //varifyOutput(r_host, a_host, N);
            //����O����
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
