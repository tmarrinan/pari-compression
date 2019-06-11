#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// CUDA Kernel function to make grayscale image on GPU

__global__ void RGBTOGrayScale(uint8_t *inRGBImage, uint8_t *outGrayImage, int srcW, int srcH )

{
	
// using tiles of shared memory to sore image data for current block
 
	uint8_t shInImage[4*4*4]; // threads per block is 4*4 and each pixel has 4 attributes R, G, B, A.

	uint8_t shoutImage[4*4*3];

	int tid = blockIdx.x * blockDim.x + threadIdx.x; //taking index along srcW in source image.
	//printf("tid:  %d %d %d \n", blockIdx.x, blockDim.x, threadIdx.x);
	//int count;
	int x_offset = ((tid % (srcW/4))*4);
	int y_offset = (4*( tid/(srcW/4)));
	//printf(" tile id %d - %d\n", tid, (srcW*srcH)/16);
	//printf("x_offset, y offset %d - %d\n", x_offset, y_offset);

	//memcpy(buffer, pixels, size)
	if( tid < (srcW*srcH)/16)
	{
		printf("COMPUTE tile id %d - %d\n", tid, (srcW*srcH)/16);

		int pixel_x_offset =  x_offset *4;
		int pixel_y_offset = y_offset *4;

		int px =pixel_y_offset * srcW*4 + pixel_x_offset*4;
		memcpy(&shInImage[0], &inRGBImage[px], sizeof(char)*16);
		px = (pixel_y_offset +1) *srcW*4 + pixel_x_offset*4;
		memcpy(&shInImage[16], &inRGBImage[px],sizeof(char) *16);
		px = (pixel_y_offset +2) *srcW*4 + pixel_x_offset*4;
		memcpy(&shInImage[32], &inRGBImage[px],sizeof(char) *16);
		px = (pixel_y_offset +3) *srcW*4 + pixel_x_offset*4;
		memcpy(&shInImage[48], &inRGBImage[px],sizeof(char) *16);
	
		int gray;
		for(int i=0; i<16; i++)
		{
			gray = shInImage[i*4] * 0.3 + shInImage[i*4+1] * .59 + shInImage[i*4+2] * .11;
			shoutImage[i*3] = gray;
			shoutImage[i*3+1] = gray;
			shoutImage[i*3+2] = gray;
			//shoutImage[i*4+3] = 255;
			//printf("shout image: %d\n", shoutImage[i*3]);
		}
		x_offset = (tid %((srcW*3)/4));
		y_offset = (tid /((srcW*3)/4));
		
		pixel_x_offset =  x_offset *4;
		pixel_y_offset = y_offset *4;

		px =pixel_y_offset * srcW*3 + pixel_x_offset*3;
		memcpy(&outGrayImage[px], &shoutImage[0], sizeof(char)*12);
		px =(pixel_y_offset+1) * srcW*3 + pixel_x_offset*3;
		memcpy(&outGrayImage[px],&shoutImage[12], sizeof(char) *12);
		px =(pixel_y_offset+2) * srcW*3 + pixel_x_offset*3;
		memcpy(&outGrayImage[px],&shoutImage[24], sizeof(char) *12);
		px =(pixel_y_offset+3) * srcW*3 + pixel_x_offset*3;
		memcpy(&outGrayImage[px],&shoutImage[36], sizeof(char) *12);
		
	}
	/*else
	{
		printf("SKIP tile id %d - %d\n", tid, (srcW*srcH)/16);
	}*/
}

void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);

int main(/*int argc, char **argv*/)
{/*
  if (argc < 2)
    {
        fprintf(stderr, "Error: please specify image to convert\n");
        exit(1);
    }

    // Use first command line parameter as input file (e.g. jpeg, png, ...)
	std::string filename = argv[1];

    // Load image into memory (force image format to RGBA)
	int img_w, img_h, img_c;	
	uint8_t *CPUinput = stbi_load(filename.c_str(), &img_w, &img_h, &img_c, STBI_rgb_alpha);
	uint8_t *CPUoutput = new uint8_t[img_w *img_h*3];
*/
	const int img_w = 4;
	const int img_h= 4;
	uint8_t *CPUoutput = new uint8_t[img_w *img_h*3];
	uint8_t *CPUinput = new uint8_t[img_w *img_h *4];

	memset(CPUinput, 0, img_w *img_h *4);
	memset(CPUoutput, 0, img_w *img_h*3);

	for(int i=0; i< (img_w *img_h)/2; i++)
	{
		CPUinput[4*i] = 1;
		CPUinput[4*i+1] = 10;
		CPUinput[4*i+2] = 100; 

	}
	for(int i=(img_w *img_h)/2; i< (img_w *img_h); i++)
	{
		CPUinput[4*i] = 2;
		CPUinput[4*i+1] = 20;
		CPUinput[4*i+2] = 200; 
	
	}

/*	for(int i=0; i< (img_w/64) *(img_h/64); i++)
	{
		printf("PIXELS RGB: %d %d %d\n",CPUinput[4*i], CPUinput[4*i+1],CPUinput[4*i+2]);
	}*/

	uint8_t *GPU_input;
	uint8_t *GPU_output;

	cudaMalloc((void**)&GPU_input, (img_w*img_h*4));
	cudaMalloc((void**)&GPU_output, (img_w*img_h*3));

	// cudaMemCpy host2Device
	cudaMemcpy(GPU_input, CPUinput, (img_w*img_h*4), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = ((img_w *img_h / 16) + blockSize - 1) / blockSize;
    
    RGBTOGrayScale<<<numBlocks, blockSize>>>(GPU_input,GPU_output, img_w,img_h);

	//Copy the results back to CPU
	cudaMemcpy(CPUoutput, GPU_output, (img_w*img_h*3), cudaMemcpyDeviceToHost);	
	SavePPM("decoded.ppm", img_w, img_h, CPUoutput);
	for(int i=0; i<img_w*img_h; i++)
	{
		printf("GPU RGB: %d %d %d \n",CPUoutput[3*i], CPUoutput[3*i+1],CPUoutput[3*i+2]);
	}

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output);


}


void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P6\n%u %u\n255\n", width, height);
	fwrite(pixels, width * height * 3, 1, fp);
	fclose(fp);
}
