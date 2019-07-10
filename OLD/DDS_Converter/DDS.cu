#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <math.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct DdsPixelFormat {
    uint32_t dw_size;
    uint32_t dw_flags;
    uint32_t dw_four_cc;
    uint32_t dw_rgb_bit_count;
    uint32_t dw_r_bit_mask;
    uint32_t dw_g_bit_mask;
    uint32_t dw_b_bit_mask;
    uint32_t dw_a_bit_mask;
} DdsPixelFormat;

typedef struct DdsHeader {
    uint32_t dw_magic;
    uint32_t dw_size;
    uint32_t dw_flags;
    uint32_t dw_height;
    uint32_t dw_width;
    uint32_t dw_pitch_or_linear_size;
    uint32_t dw_depth;
    uint32_t dw_mip_map_count;
    uint32_t dw_reserved1[11];
    DdsPixelFormat ddspf;
    uint32_t dw_caps;
    uint32_t dw_caps2;
    uint32_t dw_caps3;
    uint32_t dw_caps4;
    uint32_t dw_reserved2;
} DdsHeader;

// CUDA Kernel function to make grayscale image on GPU
__global__
void ConvertRgbaImageToDxt1(int width, int height, uint8_t *pixels, uint8_t *dxt)
{
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//	printf("index: %d , stride: %d, area: %d\n", index,stride, area);
//	for (int i = index; i < area; i= i+stride)
	int i, j;
    uint32_t offset;
    uint32_t current_pos = 0;
    uint8_t block[64];
    uint8_t color_min[3];
    uint8_t color_max[3];
    uint8_t inset[3];
    uint8_t colors[16];
    uint8_t indices[16];

	for (int i = 0; i < img_w*img_h; i++)
	{
		dxt[3*i] = pixels[4*i]*.3;
		dxt[3*i +1] = pixels[4*i +1]*.59;
		dxt[3*i +2] = pixels[4*i +2]*.11;
	}
}

int main(void)
{
	int img_w = 512;
	int img_h = 512;

	uint8_t *gpu_pixels;
	uint8_t *gpu_dxt;

   // Allocate Unified Memory -- accessible from CPU or GPU

//	cudaMallocManaged(&pixels,(img_w*img_h*4)*sizeof(uint8_t));
//	cudaMallocManaged(&gray,(img_w*img_h*3)*sizeof(uint8_t));
	uint8_t *cpu_dxt = new uint8_t[img_w *img_h *3];
	uint8_t *cpu_pixels = new uint8_t[img_w * img_h * 4];
	for (int i =0; i<img_w * img_h;i++)
	{
		cpu_pixels[4*i] = 255;
		cpu_pixels[4*i+1] = 0;
		cpu_pixels[4*i+2] = 0;
	} 
	//malloc gpu arrays
	//cpu_dxt = (uint8_t*)malloc((img_w*img_h*3)*sizeof(uint8_t));
	//cpu_pixels = (uint8_t*)malloc((img_w*img_h*4)*sizeof(uint8_t));
	
	//cuda malloc cpu arrays	
	cudaMalloc((void**)&gpu_pixels, (img_w*img_h*4));
	cudaMalloc((void**)&gpu_dxt, (img_w*img_h*3));

	// cudaMemCpy host2Device
	cudaMemcpy(gpu_pixels, cpu_pixels, (img_w*img_h*4), cudaMemcpyHostToDevice);

 // Run kernel
//	int blockSize = 256;
//	int numBlocks = ((img_w*img_h*4) + blockSize - 1) / blockSize;
//	int grid_size = (img_w*img_h*4)/ blockSize;
	
// Run kernel on 256*4  elements on the GPU

	ConvertRgbaImageToDxt1<<<1,1>>>(img_w, img_h, gpu_pixels,gpu_dxt);
//	makeGray<<<numBlocks,blockSize>>>((img_w*img_h),gpu_pixels,gpu_dxt);
 
	cudaMemcpy(cpu_dxt, gpu_dxt, (img_w*img_h*3), cudaMemcpyDeviceToHost);

 // Check gray values
	for (int i = 0; i < (img_w *img_h *3); i++)
	{
		printf("gray: %d\n",cpu_dxt[i]);
	}

  // Free memory
	free(cpu_pixels);
	free(cpu_dxt);
	cudaFree(gpu_pixels);
	cudaFree(gpu_dxt);
	return 0;
}
