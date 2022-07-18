#include <iostream>
#include <time.h>
#include <stdio.h>
#include <mpi.h>

using namespace std;

//const int num_gpus = 2;
//經測試於RTX3070最佳值為128，接著為256。
const int threadsPerBlock = 128;
//相加之元素個數(2^30-3)
const int Total_N               = (1 <<10 );
const int iters           = 1;
//const int kernal_number = 7;
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


int main(int argc, char *argv[]) {
	
	/************************
	*MPI初始化           	*
	*得到當前的rank(秩)  	*
	*得到目前處理器的名稱	*
	************************/
	int world_size, world_rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name,&name_len);

	int num_gpus;
	int device;
	cudaError_t err;
	cudaGetDeviceCount(&num_gpus);
	printf("node_%d_num_gpus = %d\n", world_rank,num_gpus);
	//err = cudaGetDevice(&device);
	//printf("cudaGetDevice = %d\n",err);

	const int N = Total_N/num_gpus;
	const int blocksPerGrid   = (N + threadsPerBlock - 1)/threadsPerBlock;
	float total_time[num_gpus];
	double* a_host[num_gpus], *r_host[num_gpus];
	double* a_device[num_gpus], *r_device[num_gpus];
	for(int i = 0; i < num_gpus; i++){
		//主機內存分配
		cudaMallocHost(&a_host[i], N * sizeof(double));
		cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
		//顯卡內存分配
	}
	//cout << "Memory Allocation Completed" << endl;


	//題目生成
	//cout << "Generating list" << endl;
	for(int i = 0; i < num_gpus; i++){
		for(int j=0;j<N;j++){
			a_host[i][j] = 1.0+i*N+j;
		}
		//cout << "a_host "<< i <<" Generating Completed" << endl;
		for(int j=0;j<blocksPerGrid;j++){
			r_host[i][j] = 0.0;
		}
		//cout << "r_host "<< i <<" Generating Completed" << endl;
	}


	//定義開始和停止事件(Event)
	cudaStream_t stream[num_gpus];
	cudaEvent_t start_events[num_gpus];
	cudaEvent_t stop_events[num_gpus];
	float elapsedTime[num_gpus];

	//定義顯卡流
	for(int i = 0; i < num_gpus; i++){
		err = cudaSetDevice(i);
		//printf("GPU %d Set Device = %d\n",i,err);
		err = cudaGetDevice(&device);
		//printf("GPU %d Get Device = %d\n",i,err);
		//定義顯卡流
		cudaStreamCreate(&stream[i]);
		//printf("GPU %d Stream Define Completed\n",i);

		cudaMalloc(&a_device[i], N * sizeof(double));
		cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));



		//記憶體設定(異步)
		cudaMemcpyAsync(a_device[i], a_host[i], N * sizeof(double),
						cudaMemcpyHostToDevice,stream[i]);
		cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
						cudaMemcpyHostToDevice,stream[i]);
		//printf("Mem ERROR GPU %d = %s\n",i,cudaGetErrorString(cudaGetLastError()));
		//printf("Memory asynchronous Completed\n");

		//創建開始和停止事件(Event)
		cudaEventCreate(&start_events[i]);
		cudaEventCreate(&stop_events[i]);
		//printf("Create Start & Stop Event Completed\n");

		//printf("Start Calculation\n");
		cudaEventRecord(start_events[i],stream[i]);
		//運用Kernel1進行運算
		kernel1<<<blocksPerGrid, threadsPerBlock, 0,stream[i]>>>(a_device[i], r_device[i], N);
		//紀錄停止事件(Event)
		cudaEventRecord(stop_events[i],stream[i]);
		cudaDeviceSynchronize();
		cudaEventSynchronize(stop_events[i]);
		printf("node_%d_GPU_%d ERROR = %s\n", world_rank, i, cudaGetErrorString(cudaGetLastError()));
		//printf("GPU %d Calculation Completed\n",i);

		//計算開始事件至暫停事件所經時間
		//printf("GPU %d Calculation time\n",i);
		cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
		total_time[i] = total_time[i] + (elapsedTime[i] / iters);
		//cout << "total_time "<< i << "= " << total_time[i] << endl;
		//cout << "elapsedTime "<< i << "= " << elapsedTime[i] << endl;
		if (i ==0){
			total_time[i] = total_time[i];
		}
		else{
			total_time[i] = total_time[i-1] + total_time[i];
		}

		//cout << "Event Destroy" << endl;
		cudaEventDestroy(start_events[i]);
		cudaEventDestroy(stop_events[i]);

		//資料由顯卡記憶體傳輸至主機記憶體
		//cout << "Share Memory form Device to Host" << endl;
		cudaMemcpy(r_host[i], r_device[i], blocksPerGrid * sizeof(double),
								  cudaMemcpyDeviceToHost);
		//printf("node_%d_GPU_%d Elapse time for The Kernal 1 : %f ms\n", world_rank, i, total_time[i]);
		total_time[i] = 0.0 ;
		elapsedTime[i] = 0.0 ;

		for(int j = 0; j < blocksPerGrid; j++){
			if (i == 0 && j == 0){
			r_host[0][0] = r_host[i][j];
			}
			else if (r_host[i][j] != 0){
			r_host[0][0] = r_host[0][0] + r_host[i][j];
			}
			//printf("r_host[%d][%d] = %f\n", i, j, r_host[i][j]);
			//printf("Ans [%d][%d] = %f\n", i, j, r_host[0][0]);
		}
		//printf("r_host[0][0] = %f\n",r_host[0][0]);

		//記憶體釋放
		cudaFree(r_device[i]);
		cudaFreeHost(r_host[i]);
		cudaFree(a_device[i]);
		cudaFreeHost(a_host[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("node_%d_Finshed\n",world_rank);
	MPI_Finalize();
	return 0;
}


