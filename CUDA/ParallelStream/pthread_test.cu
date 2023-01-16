const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
	int num_gpus;
    int device;
    cudaGetDeviceCount(&num_gpus);
    cudaGetDevice(&device);
	/*
    printf("num_gpus = %d\n",num_gpus);
    printf("cudaGetDevice = %d\n",ret);
	*/
	
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];
	
	int GPU_N = N / num_gpus;
	
	
	for (int GPU = 0; GPU < num_gpus; GPU++)
	{
		cudaSetDevice(GPU);
		cudaStream_t streams[num_streams];
		for (int i = 0; i < num_streams; i++) 
		{
			cudaStreamCreate(&streams[i]);
		}
	}
	
	
	
	for (int GPU = 0; GPU < num_gpus; GPU++)
	{
		cudaSetDevice(GPU);
		for (int i = 0; i < num_streams; i++) 
		{
			cudaMalloc(&data[i], GPU_N * sizeof(float));
		}
	}
	
	
	
	for (int GPU = 0; GPU < num_gpus; GPU++)
	{
		cudaSetDevice(GPU);
		for (int i = 0; i < num_streams; i++) 
		{
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
		}
	}
    cudaDeviceReset();

    return 0;
}