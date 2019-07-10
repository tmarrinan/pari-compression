#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TILE_W	16 //(4*4)
#define TILE_H	16 //(4*4)
#define R	2 //filter radius
#define D	(R*2 +1) //filter diameter
#define S	(D*D) //filter size
#define BLOCK_W (TILE_W + (2*R))
#define BLOCK_H (TILE_H + (2*R))

// CUDA Kernel function to make grayscale image on GPU
texture<unsigned char, cudaTextureType2D> tex8u;

//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void box_filter_kernel_8u_c1(unsigned char* output, unsigned char * input, const int width, const int height, const size_t pitch, const int fWidth, const int fHeight)
{
	__shared__ int smem [BLOCK_W*BLOCK_H];
    int x /*xIndex*/ = blockIdx.x * TILE_W/*blockDim.x*/ + threadIdx.x - R;
    int y /*yIndex*/ = blockIdx.y * TILE_H/*blockDim.y*/ + threadIdx.y -R;
	// calmp to edge of image
	x = max(0,x);
	x = min(x, width -1);
	y = max(y,0);
	y = min(y, height -1);

	unsigned int index = y*width +x;
	unsigned int bindex = threadIdx.y*blockDim.y + threadIdx.x;

	//each thread copies its pixel of the block to shared memory
	smem[bindex] = input[index];

	//Make sure the current thread is inside the image bounds
	if((threadIdx.x >= R ) && (threadIdx.x < (BLOCK_W -R)) &&
		(threadIdx.y >= R) && (threadIdx.y < (BLOCK_H-R)))
	{
	float sum =0;
	//Sum the window pixels
        for(int dy= -R; dy<=R; dy++)
        {
            for(int dx=-R; dx<=R; dx++)
            {
                float i = smem[bindex + (dy*blockDim.x) + dx];
		sum+=i;
		
            }
        }
	output[index] = sum/S;
/*
    const int filter_offset_x = fWidth/2;
    const int filter_offset_y = fHeight/2;

    float output_value = 0.0f;

    //Make sure the current thread is inside the image bounds
    if(xIndex<width && yIndex<height)
    {
        //Sum the window pixels
        for(int i= -filter_offset_x; i<=filter_offset_x; i++)
        {
            for(int j=-filter_offset_y; j<=filter_offset_y; j++)
            {
                //No need to worry about Out-Of-Range access. tex2D automatically handles it.
                output_value += tex2D(tex8u,xIndex + i,yIndex + j);
            }
        }

        //Average the output value
        //output_value /= (fWidth * fHeight);

        //Write the averaged value to the output.
        //Transform 2D index to 1D index, because image is actually in linear memory
        int index = yIndex * pitch + xIndex;

        output[index] = static_cast<unsigned char>(output_value);
    */
	}
}

void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);

void box_filter_8u_c1(unsigned char* CPUinput, unsigned char* CPUoutput, const int width, const int height, const int widthStep, const int filterWidth, const int filterHeight)
{

    /*
     * 2D memory is allocated as strided linear memory on GPU.
     * The terminologies "Pitch", "WidthStep", and "Stride" are exactly the same thing.
     * It is the size of a row in bytes.
     * It is not necessary that width = widthStep.
     * Total bytes occupied by the image = widthStep x height.
     */

    //Declare GPU pointer
    unsigned char *GPU_input, *GPU_output;

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch = 0;
   /* cudaMallocPitch<unsigned char>(&GPU_input,&gpu_image_pitch,width,height);
    cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,width,height);
    cudaMemcpy2D(GPU_input,gpu_image_pitch,CPUinput,widthStep,width,height,cudaMemcpyHostToDevice);
    cudaBindTexture2D(NULL,tex8u,GPU_input,width,height,gpu_image_pitch);
    tex8u.addressMode[0] = tex8u.addressMode[1] = cudaAddressModeBorder;

*/
	cudaMalloc((void**)&GPU_input, (width*height*4));
	cudaMalloc((void**)&GPU_output, (width*height*4));

	// cudaMemCpy host2Device
	cudaMemcpy(GPU_input, CPUinput, (width*height*4), cudaMemcpyHostToDevice);
    dim3 block_size(16,16);

    /*
     * Specify the grid size for the GPU.
     * Make it generalized, so that the size of grid changes according to the input image size
     */

    dim3 grid_size;
    grid_size.x = (width + block_size.x - 1)/block_size.x;  /*< Greater than or equal to image width */
    grid_size.y = (height + block_size.y - 1)/block_size.y; /*< Greater than or equal to image height */

    //Launch the kernel
    box_filter_kernel_8u_c1<<<grid_size,block_size>>>(GPU_output,GPU_input,width*4,height*4,gpu_image_pitch,filterWidth,filterHeight);

    //Copy the results back to CPU
    //cudaMemcpy2D(CPUoutput,widthStep,GPU_output,gpu_image_pitch,width,height,cudaMemcpyDeviceToHost);
	cudaMemcpy(CPUoutput, GPU_output, (width*height), cudaMemcpyDeviceToHost);	
	//SavePPM("decoded.ppm", width, height, CPUoutput);
	for(int i=0; i<(width*height); i++)
	{
		printf("GPU RGB: %d %d %d\n",CPUoutput[4*i], CPUoutput[4*i+1],CPUoutput[4*i+2]);
	}
    //Release the texture
    cudaUnbindTexture(tex8u);

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output);
}
int main(/*int argc, char **argv*/)
{
    /*if (argc < 2)
    {
        fprintf(stderr, "Error: please specify image to convert\n");
        exit(1);
    }

    // Use first command line parameter as input file (e.g. jpeg, png, ...)
	std::string filename = argv[1];

    // Load image into memory (force image format to RGBA)
	int img_w, img_h, img_c;	
	uint8_t *cpu_pixels = stbi_load(filename.c_str(), &img_w, &img_h, &img_c, STBI_rgb_alpha);*/
	const int img_w =8;
	const int img_h= 8;
	unsigned char *cpu_Dds = new uint8_t[(img_w *img_h)/2];
	unsigned char *cpu_pixels = new uint8_t[img_w *img_h *4];
	for(int i=0; i< (img_w *img_h)/2; i++)
	{
		cpu_pixels[4*i] = 1;
		cpu_pixels[4*i+1] = 10;
		cpu_pixels[4*i+2] = 100; 

	}
	for(int i=(img_w *img_h)/2; i< (img_w *img_h); i++)
	{
		cpu_pixels[4*i] = 2;
		cpu_pixels[4*i+1] = 20;
		cpu_pixels[4*i+2] = 200; 
	
	}
	for(int i=0; i< (img_w *img_h); i++)
	{
		printf("PIXELS RGB: %d %d %d\n",cpu_pixels[4*i], cpu_pixels[4*i+1],cpu_pixels[4*i+2]);
	}

	box_filter_8u_c1(cpu_pixels, cpu_Dds, img_w, img_h, img_w, 4, 4);



}


void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P6\n%u %u\n255\n", width, height);
	fwrite(pixels, width * height * 3, 1, fp);
	fclose(fp);
}
