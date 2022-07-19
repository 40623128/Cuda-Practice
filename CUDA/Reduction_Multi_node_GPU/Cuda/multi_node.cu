#include <iostream>
#include <time.h>
#include <stdio.h>
#include <mpi.h>

using namespace std;

const int threadsPerBlock = 128;
const int Total_N         = (1<<10);
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
	*MPI初始化              *
	*得到當前的process 數目 *
	*得到當前的rank(秩)  	*
	*得到目前處理器的名稱	*
	************************/
	int world_size, world_rank;
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	/*************************
	*num_gpus   單節點GPU數量*
	*total_gpus 共使用GPU數量*
	*device     各GPU編號    *
	*cudaError_t error_code  *
	*************************/
	int num_gpus;
	int total_gpus;
	int device;
	cudaError_t err;
	cudaGetDeviceCount(&num_gpus);
	//num_gpus = 2;
	total_gpus = num_gpus*world_size;
	printf("node_%d_num_gpus = %d\n", world_rank,num_gpus);
	printf("total_gpus = %d\n", total_gpus);
	
	const int Node_N = Total_N/world_size;
	const int GPU_N = Node_N/num_gpus;
	const int blocksPerGrid   = (GPU_N + threadsPerBlock - 1)/threadsPerBlock;
	float total_time[total_gpus];
	double* a_host[total_gpus], *r_host[total_gpus];
	double* a_device[total_gpus], *r_device[total_gpus];
	double* All_host;
	
	//題目生成
	if(world_rank == 0){
		cudaMallocHost(&All_host, Total_N * sizeof(double));
		for(int i=0;i<Total_N;i++){
			All_host[i] = 1.0;
		}
	}
	//node同步
	//MPI_Barrier(MPI_COMM_WORLD);
	double *node_host = (double *)malloc(sizeof(double) * Node_N);
	//題目分配各節點
	MPI_Scatter(All_host, Node_N, MPI_DOUBLE,
				node_host, Node_N, MPI_DOUBLE,
				0, MPI_COMM_WORLD);
	printf("MPI_Scatter Finshed\n");
	for(int i = 0; i < num_gpus; i++){
		//主機內存分配
		cudaMallocHost(&a_host[i], GPU_N * sizeof(double));
		cudaMallocHost(&r_host[i], blocksPerGrid * sizeof(double));
	}
	//printf("host Memory location Finshed\n");
	for(int i = 0; i < num_gpus; i++){
		for(int j = 0; j < GPU_N; j++){
			a_host[i][j] = node_host[i*GPU_N+j];
			//printf("a_host[%d][%d] = %f\n", i, j , a_host[i][j]);
			}
	}
	//printf("a_host location Finshed\n");
	//printf("Memory Allocation Completed\n");

	//定義開始和停止事件(Event)
	cudaStream_t stream[num_gpus];
	cudaEvent_t start_events[num_gpus];
	cudaEvent_t stop_events[num_gpus];
	float elapsedTime[num_gpus];
	
	//定義顯卡流
	for(int i = 0; i < num_gpus; i++){
		err = cudaSetDevice(i);
		err = cudaGetDevice(&device);
		cudaStreamCreate(&stream[i]);
		//printf("GPU %d Set Device = %d\n",i,err);
		//printf("GPU %d Get Device = %d\n",i,err);
		//printf("GPU %d Stream Define Completed\n",i);
		
		cudaMalloc(&a_device[i], GPU_N * sizeof(double));
		cudaMalloc(&r_device[i], blocksPerGrid * sizeof(double));



		//記憶體複製(異步)
		cudaMemcpyAsync(a_device[i], a_host[world_rank*num_gpus+i], GPU_N * sizeof(double),
						cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(r_device[i], r_host[i], blocksPerGrid * sizeof(double),
						cudaMemcpyHostToDevice, stream[i]);
		//printf("Mem ERROR GPU %d = %s\n",i,cudaGetErrorString(cudaGetLastError()));
		//printf("Memory asynchronous Completed\n");

		//創建開始和停止事件(Event)
		cudaEventCreate(&start_events[i]);
		cudaEventCreate(&stop_events[i]);
		//printf("Create Start & Stop Event Completed\n");

		//printf("Start Calculation\n");
		cudaEventRecord(start_events[i],stream[i]);
		//運用Kernel1進行運算
		kernel1<<<blocksPerGrid, threadsPerBlock, 0,stream[i]>>>(a_device[i], r_device[i], GPU_N);
		//紀錄停止事件(Event)
		cudaEventRecord(stop_events[i],stream[i]);
		cudaDeviceSynchronize();
		cudaEventSynchronize(stop_events[i]);
		//printf("node_%d_GPU_%d ERROR = %s\n", world_rank, i, cudaGetErrorString(cudaGetLastError()));
		//printf("GPU %d Calculation Completed\n",i);

		//計算開始事件至暫停事件所經時間
		//printf("GPU %d Calculation time\n",i);
		cudaEventElapsedTime(&elapsedTime[i], start_events[i], stop_events[i]);
		total_time[i] = total_time[i] + (elapsedTime[i] / iters);
		//printf("total_time %d = %f Calculation time\n", i, total_time[i]);
		//printf("elapsedTime %d = %f Calculation time\n", i, elapsedTime[i]);
		
		if (i ==0){
			total_time[i] = total_time[i];
		}
		else{
			total_time[i] = total_time[i-1] + total_time[i];
		}

		//printf("Event Destroy\n");
		cudaEventDestroy(start_events[i]);
		cudaEventDestroy(stop_events[i]);

		//資料由顯卡記憶體傳輸至主機記憶體
		//printf("Share Memory form Device to Host\n");
		cudaMemcpy(r_host[i], r_device[i], blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);
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
		//printf("node_%d_GPU_%d = %f\n", world_rank, i, r_host[0][0]);
		//printf("r_host[0][0] = %f\n", r_host[0][0]);
	}
	
	double *All_Ans = (double *)malloc(sizeof(double) * world_size);;
	MPI_Gather(&r_host[0][0], 1, MPI_DOUBLE, All_Ans, 1, MPI_DOUBLE, 0,
           MPI_COMM_WORLD);
	if (world_rank == 0){
		double final_Ans;
		for(int i = 0; i < world_size; i++){
		final_Ans += All_Ans[i];
		}
		printf("final_Ans = %f\n", final_Ans);
	}
	for(int i = 0; i < num_gpus; i++){
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


