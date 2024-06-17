#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define MASK_N 2
#define MASK_X 3
#define MASK_Y 3
#define WHITE  255
#define BLACK  0

unsigned char *image_s;     // source image array
unsigned char *image_t;     // target image array
unsigned char *d_image_s;     // source image array
unsigned char *d_image_t;     // target image array
FILE *fp_s = NULL;                 // source file handler
FILE *fp_t = NULL;                 // target file handler

unsigned int   width, height;      // image width, image height
unsigned int   rgb_raw_data_offset;// RGB raw data offset
unsigned char  bit_per_pixel;      // bit per pixel
unsigned short byte_per_pixel;     // byte per pixel
cudaError_t cudaStatus;
// bitmap header
unsigned char header[54] = {
   0x42,        // identity : B
   0x4d,        // identity : M
   0, 0, 0, 0,  // file size
   0, 0,        // reserved1
   0, 0,        // reserved2
   54, 0, 0, 0, // RGB data offset
   40, 0, 0, 0, // struct BITMAPINFOHEADER size
   0, 0, 0, 0,  // bmp width
   0, 0, 0, 0,  // bmp height
   1, 0,        // planes
   24, 0,       // bit per pixel
   0, 0, 0, 0,  // compression
   0, 0, 0, 0,  // data size
   0, 0, 0, 0,  // h resolution
   0, 0, 0, 0,  // v resolution 
   0, 0, 0, 0,  // used colors
   0, 0, 0, 0   // important colors
};

// sobel mask
__constant__ int d_mask[MASK_N][MASK_X][MASK_Y] = {
  {{-1,-2,-1},
   {0 , 0, 0},
   {1 , 2, 1}},

  {{-1, 0, 1},
   {-2, 0, 2},
   {-1, 0, 1}}
};
 
int read_bmp(const char *fname_s) {
	fp_s = fopen(fname_s, "rb");
	if (fp_s == NULL) {
	printf("fopen fp_s error\n");
	return -1;
	}

	// move offset to 10 to find rgb raw data offset
	fseek(fp_s, 10, SEEK_SET);
	fread(&rgb_raw_data_offset, sizeof(unsigned int), 1, fp_s);

	// move offset to 18 to get width & height;
	fseek(fp_s, 18, SEEK_SET); 
	fread(&width,  sizeof(unsigned int), 1, fp_s);
	fread(&height, sizeof(unsigned int), 1, fp_s);

	// get bit per pixel
	fseek(fp_s, 28, SEEK_SET); 
	fread(&bit_per_pixel, sizeof(unsigned short), 1, fp_s);
	byte_per_pixel = bit_per_pixel / 8;

	// move offset to rgb_raw_data_offset to get RGB raw data
	fseek(fp_s, rgb_raw_data_offset, SEEK_SET);
	
	cudaStatus = cudaMallocHost((void**)&image_s, width * height * byte_per_pixel * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMallocHost((void**)&image_t, width * height * byte_per_pixel * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	
	fread(image_s, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_s);

	return 0;
}

__global__ void sobel_kernel(unsigned char *image_s, unsigned char *image_t, int width, int height, double threshold) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	unsigned char R, G, B;
	double val[MASK_N] = {0.0};
	int adjustX, adjustY, xBound, yBound;
	double total;
	
	__shared__ double shared_mask[MASK_N][MASK_X][MASK_Y];

    // Load mask into shared memory
    if (threadIdx.x < MASK_X && threadIdx.y < MASK_Y) {
        for (int i = 0; i < MASK_N; ++i) {
            shared_mask[i][threadIdx.x][threadIdx.y] = d_mask[i][threadIdx.x][threadIdx.y];
        }
    }
    __syncthreads(); // Ensure all threads have loaded the mask
	
	for (int i = 0; i < MASK_N; ++i) {
		adjustX = (MASK_X % 2) ? 1 : 0;
		adjustY = (MASK_Y % 2) ? 1 : 0;
		xBound = MASK_X / 2;
		yBound = MASK_Y / 2;
			   
		val[i] = 0.0;
		for (int v = -yBound; v < yBound + adjustY; ++v) {
			for (int u = -xBound; u < xBound + adjustX; ++u) {
				int px = x + u;
				int py = y + v;
				if (px >= 0 && px < width && py >= 0 && py < height) {
					R = image_s[3 * (width * py + px) + 2];
					G = image_s[3 * (width * py + px) + 1];
					B = image_s[3 * (width * py + px) + 0];
					val[i] += (R + G + B) / 3 * shared_mask[i][u + xBound][v + yBound];
				}
			}
	   }
   }

	total = sqrt(val[0] * val[0] + val[1] * val[1]);

	if (total > threshold) {
        image_t[3 * (width * y + x) + 2] = 0;
        image_t[3 * (width * y + x) + 1] = 0;
        image_t[3 * (width * y + x) + 0] = 0;
    } else {
        image_t[3 * (width * y + x) + 2] = 255;
        image_t[3 * (width * y + x) + 1] = 255;
        image_t[3 * (width * y + x) + 0] = 255;
    }
}

int write_bmp(const char *fname_t) {
    unsigned int file_size; // file size
   
    fp_t = fopen(fname_t, "wb");
    if (fp_t == NULL) {
        printf("fopen fname_t error\n");
        return -1;
    }
         
    // file size  
    file_size = width * height * byte_per_pixel + rgb_raw_data_offset;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8)  & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;
       
    // width
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8)  & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;
       
    // height
    header[22] = height &0x000000ff;
    header[23] = (height >> 8)  & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;
       
    // bit per pixel
    header[28] = bit_per_pixel;
     
    // write header
    fwrite(header, sizeof(unsigned char), rgb_raw_data_offset, fp_t);
    
    // write image
    fwrite(image_t, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_t);
       
    fclose(fp_s);
    fclose(fp_t);
       
    return 0;
}


double wallclock(void) {
	struct timeval tv;
	struct timezone tz;
	double t;

	gettimeofday(&tv, &tz);

	t = (double)tv.tv_sec*1000;
	t += ((double)tv.tv_usec)/1000.0;

	return t;
}// millisecond


int main() {
	double t1, t2, t3, t4;
	
	read_bmp("../lena_large.BMP"); // 24 bit gray level image

	cudaMalloc(&d_image_s, width * height * 3 * sizeof(unsigned char));
	cudaMalloc(&d_image_t, width * height * 3 * sizeof(unsigned char));
	// Copy input data from host to device
	t1 = wallclock();
	cudaMemcpy(d_image_s, image_s, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	t2 = wallclock();
	// Define grid and block dimensions
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	
	// Launch kernel
	sobel_kernel<<<gridSize, blockSize>>>(d_image_s, d_image_t, width, height, 30.0);
	cudaDeviceSynchronize();
	t3 = wallclock();

	// Copy result back to host
	cudaMemcpy(image_t, d_image_t, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	t4 = wallclock();
	
	// Write output image
	write_bmp("lena_sobel.bmp");
	printf("Host To Device Elapsed time = %f(ms)\n", t2 - t1);
	printf("Kernel Elapsed time = %f(ms)\n", t3 - t2);
	printf("Device To Host Elapsed time = %f(ms)\n", t4 - t3);
	printf("Total Elapsed time = %f(ms) \n", t4 - t1);
	
	// Free GPU memory
	cudaFree(d_image_s);
	cudaFree(d_image_t);

	// Free CPU memory
	cudaFreeHost(image_s);
	cudaFreeHost(image_t);
}
