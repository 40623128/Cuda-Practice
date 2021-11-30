#include<iostream>
#include<time.h>
using namespace std;

const int num_gpus = 2;
//�g���թ�RTX3070�̨έȬ�128�A���۬�256�C
const int threadsPerBlock = 128;
//�ۥ[�������Ӽ�(2^30-3)
const int N               = (1 <<28 );
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
    float total_time[num_gpus];
    double* a_host[num_gpus], *r_host[num_gpus];
    double* a_device[num_gpus], *r_device[num_gpus];

    for(int i = 0; i < num_gpus; i++){
        //�D�����s���t
        cudaMallocHost(&a_host[i], N * sizeof(double));
        cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
        //��d���s���t
        cudaMalloc(&a_device[i], N * sizeof(double));
        cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));
        cout << i << endl;
    }
    cout << "���s���t����" << endl;


    //�D�إͦ�
    for(int i = 0; i < num_gpus; i++){
        cout << "�D�ض}�l�ͦ�" << endl;
        for(int j=0;j<N;j++){
            a_host[i][j] = 1.0;
            //cout <<"i =" <<i<< "; j =" <<j<< endl;
        }
        cout << "a_host�ͦ�����" << endl;
        for(int j=0;j<blocksPerGrid;j++){
            r_host[i][j] = 0.0;
            //cout <<"i =" <<i<< "; j =" <<j<< endl;
        }
        cout << "r_host�ͦ�����" << endl;
    }

    //�w�q��d�y
    cudaStream_t stream[num_gpus];
    for(int i = 0; i < num_gpus; i++){
        //�Ыجy
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    cout << "��d�y�w�q����" << endl;

    //�O����]�w(���B)
    for(int i = 0; i < num_gpus; i++){
        //�Ыجy
        cudaSetDevice(i);
        cudaMemcpyAsync(a_device[i], a_host[i], N * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
                                       cudaMemcpyHostToDevice, stream[i]);
    }
    cout << "�O����]�w(���B)����" << endl;

    //�w�q�}�l�M����ƥ�(Event)
    cudaEvent_t start_events[num_gpus];
    cudaEvent_t stop_events[num_gpus];

    //�Ыض}�l�M����ƥ�(Event)
    for(int i = 0; i < num_gpus; i++){
     cudaSetDevice(i);
     cudaEventCreate(&start_events[i]);
     cudaEventCreate(&stop_events[i]);
    }
    cout << "�Ыض}�l�M����ƥ󧹦�" << endl;

    for(int i=0;i<iters;i++){
        for(int i = 0; i < num_gpus; i++){
            cudaSetDevice(i);
            // In cudaEventRecord, ommit stream or set it to 0 to record
            // in the default stream. It must be the same stream as
            // where the kernel is launched.
            //�����}�l�ƥ�(Event)
            cudaEventRecord(start_events[i], stream[i]);
            //�B��Kernel1�i��B��
            kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(a_device[i], r_device[i], N);
            //��������ƥ�(Event)
            cudaEventRecord(stop_events[i], stream[i]);
        }
    }
    cout << "�B�⧹��" << endl;

    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop_events[i]);
    }
    cout << "�p��g�L�ɶ�" << endl;
    float elapsedTime[num_gpus];
    //�p��}�l�ƥ�ܼȰ��ƥ�Ҹg�ɶ�
    for(int i = 0; i < num_gpus; i++){
        cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
        total_time[i] = total_time[i] + (elapsedTime[i] / iters);
    }

    cout << "�ƥ󲾰�" << endl;
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }

    cout << "��ƥ���d�O����ǿ�ܥD���O����" << endl;
    //��ƥ���d�O����ǿ�ܥD���O����
    for(int i = 0; i < num_gpus; i++){
        //�Ыجy
        cudaSetDevice(i);
        cudaMemcpy(r_host[i], r_device[i], blocksPerGrid * sizeof(double),
                                  cudaMemcpyDeviceToHost);
    }

    cout << "����O����" << endl;
    //����O����
    for(int i = 0; i < num_gpus; i++){
        cudaSetDevice(i);
        cudaFree(r_device[i]);
        cudaFreeHost(r_host[i]);
        cudaFree(a_device[i]);
        cudaFreeHost(a_host[i]);
    }

    for(int i = 0; i < num_gpus; i++){
        cout << "GPU "<< i <<" Elapse time for The Kernal 1 :"<< total_time[i] << " ms" << endl;
        total_time[i] = 0.0 ;
    }
    return 0;
}
