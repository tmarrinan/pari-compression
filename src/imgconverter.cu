#include "imgconverter.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <tr1/functional>
/*static uint8_t *rgba_gpu_input;
static uint8_t *gpu_temp;
static uint8_t *gpu_sizes;
static uint8_t *gpu_output;*/
static uint32_t *final_size;
static int img_w;
static int img_h;
typedef int mytype;

using namespace thrust::placeholders;
__global__ void RgbaToGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height);
__global__ void RgbaToTileGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height);
__global__ void RgbaToDxt1Kernel(uint8_t *rgba, uint8_t *dxt1, int width, int height);
__global__ void RgbaToTrleKernel(uint8_t *rgba, uint8_t *trle_tmp, uint8_t *trle_size, int width, int height);
__global__ void FinalizeTrleKernel(uint8_t *trle_tmp, uint8_t *trle, uint8_t *trle_size, int width, int height, uint32_t *final_size);
__device__ void ExtractTile4x4(uint32_t offset, uint8_t *pixels, int width, uint8_t out_tile[64]);
__device__ void ExtractTile16x16(uint32_t offset, uint8_t *pixels, int width, uint8_t out_tile[1024]);
__device__ void GetMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__device__ uint16_t ColorTo565(uint8_t color[3]);
__device__ uint32_t ColorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
__device__ uint32_t ColorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__device__ uint32_t ColorFromRgba(uint8_t *rgba, uint32_t offset);
__device__ void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__device__ void WriteUint32(uint8_t *buffer, uint32_t offset, uint32_t value);
__device__ uint32_t ColorFromRgbaWhole(uint8_t *rgba, uint32_t offset);
struct grayscaleFunctor
{
	const uint8_t *rgba;
        uint8_t *result;
	size_t size;       
	grayscaleFunctor( thrust::device_vector<uint8_t> const& rgba_input_t, thrust::device_vector<uint8_t>& gpu_output)
        {
          rgba = thrust::raw_pointer_cast(rgba_input_t.data());
          result = thrust::raw_pointer_cast(gpu_output.data());
         size = gpu_output.size();
        } 
	__host__ __device__
	void operator()(int x)
	{
		if(x < size)
		{
			uint8_t red = rgba[4 * x + 0];
			uint8_t green = rgba[4 * x + 1];
			uint8_t blue = rgba[4 * x + 2];
			result[x] = (uint8_t)(0.299 * red + 0.587 * green + 0.114 * blue);
		}
	}
};

struct dxt1Functor
{
        uint8_t *rgba;
        uint8_t *dxt1;
        int width;
	size_t size;
        dxt1Functor(int width_input, thrust::device_vector<uint8_t>& rgba_input, thrust::device_vector<uint8_t>& dxt1_output)
        {
          rgba = thrust::raw_pointer_cast(rgba_input.data());
          dxt1 = thrust::raw_pointer_cast(dxt1_output.data());
          width = width_input;
	  size = rgba_input.size() /16;
        }
        __device__
        void operator()(int x)
        {
		if (x < size)
   		 {
				
        		uint8_t tile[64];
       			uint8_t color_min[3];
        		uint8_t color_max[3];

      			int tile_x = x % (width / 4);
        		int tile_y = x / (width / 4);
        		int px_x = tile_x * 4;
        		int px_y = tile_y * 4;

        		uint32_t offset = (px_y * width * 4) + (px_x * 4);
        		uint32_t write_pos = (tile_y * (width / 4) * 8) + (tile_x * 8);

       			ExtractTile4x4(offset, rgba, width, tile);
        		GetMinMaxColors(tile, color_min, color_max);
        		//printf("%u",ColorTo565(color_max));
			WriteUint16(dxt1, write_pos, ColorTo565(color_max));
       			WriteUint16(dxt1, write_pos + 2, ColorTo565(color_min));
       			WriteUint32(dxt1, write_pos + 4, ColorIndices(tile, color_min, color_max));
   		 	
		}
                
        }
};
struct trleFunctor
{
        uint8_t *rgba;
        int width;
	int height;
	uint8_t *trle_tmp;
       	uint8_t *trle_size;
        size_t size_img;
	uint8_t *num_runs;
        trleFunctor(thrust::device_vector<uint8_t>& rgba_input, thrust::device_vector<uint8_t>& tmp_input, thrust::device_vector<uint8_t>& size_input, int width_input, int height_input, thrust::device_vector<uint8_t>& num_runs_input)
        {
          rgba = thrust::raw_pointer_cast(rgba_input.data());
          width = width_input;
	  height = height_input;
	  trle_tmp = thrust::raw_pointer_cast(tmp_input.data());
	  trle_size = thrust::raw_pointer_cast(size_input.data()); 
	  num_runs = thrust::raw_pointer_cast(num_runs_input.data());
        }
        __device__
        void operator()(int tid)
        {
		//one thread per pixel
		if (tid < width*height)
  		{
        		//x,y of whole image
			int x = tid  % width;
        		int y = tid / width;

			//x,y within tile
			int tile_x = x%16;
			int tile_y = y%16;
			
			//tile id
			int tile_id_x = x / 16;
                        int tile_id_y = y / 16;
        		int tile_id = tile_id_y * (width/16) + tile_id_x;
			uint32_t color;
			uint32_t color_prev;
			uint32_t prev;
			color = ColorFromRgba(rgba,tid);
			
			//first in tile
			if(tile_x == 0 && tile_y == 0 ) 
			{
				num_runs[tid] = 1;
			}

			else
			{
				prev = tid-1;
	
				//on new row
				if (tile_x == 0)
				{
					prev = (tid-width) + 15;
				}	
				
				//if previous color (within tile) is same as current
				color_prev = ColorFromRgba(rgba, prev);

				//printf("color: %u, color prev: %u, tid: %d, x-1: %d \n", color, color_prev, tid, prev);
				
				//make so a block is consequtive
				num_runs[tile_id *256 + tile_y *16 + tile_x] = (uint8_t)(color_prev != color);
				if((tile_id *256 + tile_y *16 + tile_x) <257)
				{
					printf("id: %d, tile id: %d, in block: %d, prev: %d, color:%u, color_prev: %u\n", tid, tile_id, tile_id *256 + tile_y *16 + tile_x, prev, color, color_prev);
				}
			}
		}
        }
};

struct finalizeTrleFunctor
{
        uint8_t *trle_tmp;
        uint8_t *trle;
	uint8_t *trle_size;
        int width;
	int height;
	uint32_t *final_size;
        size_t img_size;
        finalizeTrleFunctor(uint8_t *temp, uint8_t *trle_output, uint8_t *size, int width_input, int height_input, uint32_t *final_size_input)
        {
          trle_tmp = temp;
          trle = trle_output;
	  trle_size = size;
          width = width_input;
	  height = height_input;
	  final_size = final_size_input;
          //img_size = final_size.size();
        }
        __device__
        void operator()(int x)
        {
    		if (x < width*height)
    		{
        		int tile_x = x % (width / 16);
        		int tile_y = x / (width / 16);
        		int px_x = tile_x * 16;
        		int px_y = tile_y * 16;

        		int i;
        		uint32_t offset = (px_y * width * 4) + (px_x * 4);
        		uint32_t write_pos = 0;
        		for (i = 0; i < x; i++)
        		{
            			write_pos += trle_size[i] * 4;
        		}

        		memcpy(trle + write_pos, trle_tmp + offset, trle_size[x] * 4);
        		printf("<<<kernel %d>>> write pos %u, size: %u\n", x, write_pos, trle_size[x]);
        		if (x == (width * height / 256) - 1)
        		{
            			printf("<<<kernel>>> write pos %u, size: %u\n", write_pos, trle_size[x]);
            			*final_size = write_pos + (trle_size[x] * 4);
        		}
    		}
        }
};
__global__ void RgbaToGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < width * height)
    {
        uint8_t red = rgba[4 * tid + 0];
        uint8_t green = rgba[4 * tid + 1];
        uint8_t blue = rgba[4 * tid + 2];
        gray[tid] = (uint8_t)(0.299 * red + 0.587 * green + 0.114 * blue);
    }
}

__global__ void RgbaToTileGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < width * height / 16)
    {
        int tile_x = tid % (width / 4);
        int tile_y = tid / (width / 4);
        int px_x = tile_x * 4;
        int px_y = tile_y * 4;
        
        int i, j, idx;
        for (j = px_y; j < px_y + 4; j++)
        {
            for (i = px_x; i < px_x + 4; i++)
            {
                idx = j * width + i;
                uint8_t red = rgba[4 * idx + 0];
                uint8_t green = rgba[4 * idx + 1];
                uint8_t blue = rgba[4 * idx + 2];
                gray[idx] = (uint8_t)(0.299 * red + 0.587 * green + 0.114 * blue);
            }
        }
    }
}

__global__ void RgbaToDxt1Kernel(uint8_t *rgba, uint8_t *dxt1, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < width * height / 16)
    {
        uint8_t tile[64];
        uint8_t color_min[3];
        uint8_t color_max[3];
        
        int tile_x = tid % (width / 4);
        int tile_y = tid / (width / 4);
        int px_x = tile_x * 4;
        int px_y = tile_y * 4;
        
        uint32_t offset = (px_y * width * 4) + (px_x * 4);
        uint32_t write_pos = (tile_y * (width / 4) * 8) + (tile_x * 8);
        
        ExtractTile4x4(offset, rgba, width, tile);
        GetMinMaxColors(tile, color_min, color_max);
        WriteUint16(dxt1, write_pos, ColorTo565(color_max));
        WriteUint16(dxt1, write_pos + 2, ColorTo565(color_min));
        WriteUint32(dxt1, write_pos + 4, ColorIndices(tile, color_min, color_max));
    }   
}

__global__ void RgbaToTrleKernel(uint8_t *rgba, uint8_t *trle_tmp, uint8_t *trle_size, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < width * height / 256)
    {
        uint8_t tile[1024];
        
        int tile_x = tid % (width / 16);
        int tile_y = tid / (width / 16);
        int px_x = tile_x * 16;
        int px_y = tile_y * 16;
        
        uint32_t offset = (px_y * width * 4) + (px_x * 4);
        uint32_t write_pos = offset;
        
        ExtractTile16x16(offset, rgba, width, tile);
        
        int tile_px = 0;
        uint32_t color;
        uint32_t color_next;
        uint8_t count;
        uint8_t size = 0;
        while (tile_px < 1024)
        {
            color = ColorFromRgba(tile, tile_px);
            count = 1;
            if (tile_px < 1020)
            {
                tile_px += 4;
                color_next = ColorFromRgba(tile, tile_px);
                while (color_next == color)
                {
                    tile_px += 4;
                    count++;
                    color_next = ColorFromRgba(tile, tile_px);
                }
            }
            trle_tmp[write_pos + 0] = count;
            trle_tmp[write_pos + 1] = tile[tile_px];
            trle_tmp[write_pos + 2] = tile[tile_px + 1];
            trle_tmp[write_pos + 3] = tile[tile_px + 2];
            write_pos += 4;
            size++;
        }
        trle_size[tid] = size;
        printf("bye trle\n");
    }
}

__global__ void FinalizeTrleKernel(uint8_t *trle_tmp, uint8_t *trle, uint8_t *trle_size, int width, int height, uint32_t *final_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < width * height / 256)
    {
        int tile_x = tid % (width / 16);
        int tile_y = tid / (width / 16);
        int px_x = tile_x * 16;
        int px_y = tile_y * 16;
        
        int i;
        uint32_t offset = (px_y * width * 4) + (px_x * 4);
        uint32_t write_pos = 0;
        for (i = 0; i < tid; i++)
        {
            write_pos += trle_size[i] * 4;
        }
        
        memcpy(trle + write_pos, trle_tmp + offset, trle_size[tid] * 4);
        printf("<<<kernel %d>>> write pos %u, size: %u\n", tid, write_pos, trle_size[tid]);
        if (tid == (width * height / 256) - 1)
        {
            printf("<<<kernel>>> write pos %u, size: %u\n", write_pos, trle_size[tid]);
            *final_size = write_pos + (trle_size[tid] * 4);
        }
    }
}

__device__ void ExtractTile4x4(uint32_t offset, uint8_t *pixels, int width, uint8_t out_tile[64])
{
    int i, j;
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 16; i++)
        {
            out_tile[j * 16 + i] = pixels[offset + i];
        }
        offset += width * 4;
    }
}

__device__ void ExtractTile16x16(uint32_t offset, uint8_t *pixels, int width, uint8_t out_tile[1024])
{
    int i, j;
    for (j = 0; j < 16; j++)
    {
        for (i = 0; i < 64; i++)
        {
            out_tile[j * 64 + i] = pixels[offset + i];
        }
        offset += width * 4;
    }
}

__device__ void GetMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3])
{
    uint8_t inset[3];
    memset(color_min, 255, 3);
    memset(color_max, 0, 3);
    
    int i;
    for (i = 0; i < 16; i++)
    {
        color_min[0] = min(color_min[0], tile[i * 4 + 0]);
        color_min[1] = min(color_min[1], tile[i * 4 + 1]);
        color_min[2] = min(color_min[2], tile[i * 4 + 2]);
        color_max[0] = max(color_max[0], tile[i * 4 + 0]);
        color_max[1] = max(color_max[1], tile[i * 4 + 1]);
        color_max[2] = max(color_max[2], tile[i * 4 + 2]);
    }
    
    inset[0] = (color_max[0] - color_min[0]) >> 4;
    inset[1] = (color_max[1] - color_min[1]) >> 4;
    inset[2] = (color_max[2] - color_min[2]) >> 4;
    
    color_min[0] = min(color_min[0] + inset[0], 255);
    color_min[1] = min(color_min[1] + inset[1], 255);
    color_min[2] = min(color_min[2] + inset[2], 255);
    color_max[0] = max(color_max[0] - inset[0], 0);
    color_max[1] = max(color_max[1] - inset[1], 0);
    color_max[2] = max(color_max[2] - inset[2], 0);
}

__device__ uint16_t ColorTo565(uint8_t color[3])
{
    return ((color[0] >> 3) << 11) | ((color[1] >> 2) << 5) | (color[2] >> 3);
}

__device__ uint32_t ColorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset)
{
    int dx = tile[t_offset + 0] - colors[c_offset + 0];
    int dy = tile[t_offset + 1] - colors[c_offset + 1];
    int dz = tile[t_offset + 2] - colors[c_offset + 2];
    
    return (dx*dx) + (dy*dy) + (dz*dz);
}

__device__ uint32_t ColorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3])
{
    uint8_t colors[16];
    uint8_t indices[16];
    int i, j;
    uint8_t C565_5_MASK = 0xF8;   // 0xFF minus last three bits
    uint8_t C565_6_MASK = 0xFC;   // 0xFF minus last two bits
    
    colors[0] = (color_max[0] & C565_5_MASK) | (color_max[0] >> 5);
    colors[1] = (color_max[1] & C565_6_MASK) | (color_max[1] >> 6);
    colors[2] = (color_max[2] & C565_5_MASK) | (color_max[2] >> 5);
    colors[4] = (color_min[0] & C565_5_MASK) | (color_min[0] >> 5);
    colors[5] = (color_min[1] & C565_6_MASK) | (color_min[1] >> 6);
    colors[6] = (color_min[2] & C565_5_MASK) | (color_min[2] >> 5);
    colors[8] = (2 * colors[0] + colors[4]) / 3;
    colors[9] = (2 * colors[1] + colors[5]) / 3;
    colors[10] = (2 * colors[2] + colors[6]) / 3;
    colors[12] = (colors[0] + 2 * colors[4]) / 3;
    colors[13] = (colors[1] + 2 * colors[5]) / 3;
    colors[14] = (colors[2] + 2 * colors[6]) / 3;
    
    uint32_t dist, min_dist;
    for (i = 0; i < 16; i++)
    {
        min_dist = 195076;  // 255 * 255 * 3 + 1
        for (j = 0; j < 4; j++)
        {
            dist = ColorDistance(tile, i * 4, colors, j * 4);
            if (dist < min_dist)
            {
                min_dist = dist;
                indices[i] = j;
            }
        }
    }
    
    uint32_t result = 0;
    for (i = 0; i < 16; i++)
    {
        result |= indices[i] << (i * 2);
    }
    return result;
}

__device__ uint32_t ColorFromRgba(uint8_t *rgba, uint32_t offset)
{
    
    uint32_t result = rgba[offset * 4] << 24;
    result |= rgba[offset * 4 + 1] << 16;
    result |= rgba[offset * 4 + 2] << 8;
    result |= rgba[offset * 4 + 3];
    return result;
}
__device__ uint32_t ColorFromRgbaWhole(uint8_t *rgba, uint32_t offset)
{

    uint32_t result = rgba[offset] << 24;
    result |= rgba[offset + 1] << 16;
    result |= rgba[offset + 2] << 8;
    result |= rgba[offset + 3];
    return result;
}
__device__ void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value)
{
   //printf("%u\n", value& 0xFF);
   buffer[offset + 0] = value & 0xFF;
   buffer[offset + 1] = (value >> 8) & 0xFF;
}

__device__ void WriteUint32(uint8_t *buffer, uint32_t offset, uint32_t value)
{
    buffer[offset + 0] = value & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
}


void InitImageConverter(int width, int height)
{
	//width is wrong here so divide by 4
    img_w = width/4;
    img_h = height/4;
    //cudaMalloc((void**)&rgba_gpu_input, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_temp, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_sizes, img_w * img_h / 256);
    //cudaMalloc((void**)&gpu_output, img_w * img_h * 4);
    cudaMalloc((void**)&final_size, sizeof(uint32_t));

}
/*
void RgbaToGrayscale(uint8_t *rgba, uint8_t *gray)
{
    // intialize all variables 
    uint8_t *gray_device;

    thrust::device_vector<uint8_t> rgba_input_t(rgba, rgba+img_w*img_h*4 );		
    thrust::device_vector<uint8_t> gpu_output_t (img_w*img_h);
    thrust::counting_iterator<size_t> it(0);    
    //thrust for each n to specify number of threads and number of input and output vectors
    thrust::for_each_n(thrust::device, it, gpu_output_t.size(), grayscaleFunctor(rgba_input_t,gpu_output_t));
    
    // tile-based way (4x4 tiles of pixels per thread)
    //int num_blocks = ((img_w * img_h / 16) + block_size - 1) / block_size;
    //RgbaToTileGrayscaleKernel<<<num_blocks, block_size>>>(rgba_input_t, gpu_output, img_w, img_h);
    
    //copy back to host, Thrust can get rid of memcpy's
    //thrust::copy(gpu_output_t.begin(), gpu_output_t.end(), gray.begin());
    gray_device = thrust::raw_pointer_cast(&gpu_output_t[0]); 
    cudaMemcpy(gray, gray_device, img_w * img_h, cudaMemcpyDeviceToHost);
}

void RgbaToDxt1(uint8_t *rgba, uint8_t *dxt1)
{
    uint8_t *dxt1_device;
    thrust::device_vector<uint8_t> rgba_input(rgba, rgba+img_w*img_h*4);
    thrust::device_vector<uint8_t> dxt1_output(img_w*img_h /2);
    thrust::counting_iterator<size_t> it(0);
    
    thrust::for_each_n(thrust::device, it, rgba_input.size()/16, dxt1Functor(img_w, rgba_input, dxt1_output));
    dxt1_device = thrust::raw_pointer_cast(&dxt1_output[0]);
    cudaMemcpy(dxt1, dxt1_device, img_w * img_h / 2, cudaMemcpyDeviceToHost);
   

    //cudaMemcpy(rgba_gpu_input, rgba, img_w * img_h * 4, cudaMemcpyHostToDevice);
    //int block_size = 256;
    //int num_blocks = ((img_w * img_h / 16) + block_size - 1) / block_size;
    //RgbaToDxt1Kernel<<<num_blocks, block_size>>>(rgba_gpu_input, gpu_output, img_w, img_h);
    //cudaMemcpy(dxt1, gpu_output, img_w * img_h / 2, cudaMemcpyDeviceToHost);
}*/

void RgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size)
{
    uint32_t *trle_device;
    uint8_t *num_runs_cast;
    
    thrust::device_vector<uint8_t> rgba_input(rgba, rgba+img_w*img_h*4);
    thrust::device_vector<uint8_t> gpu_temp(img_w * img_h / 256);
    thrust::device_vector<uint8_t> gpu_sizes(img_w*img_h*4);
    thrust::device_vector<uint8_t> num_runs(img_w*img_h);
    thrust::device_vector<uint8_t> total_runs(img_w*img_h/256);
    thrust::counting_iterator<size_t> it(0);
   
    thrust::for_each_n(thrust::device, it, rgba_input.size()/4, trleFunctor(rgba_input, gpu_temp, gpu_sizes, img_w, img_h,num_runs));
    /*thrust::copy_n(num_runs.begin(), 256, std::ostream_iterator<mytype>(std::cout, ","));
     std::cout << std::endl;*/

     //reduce to sum how many runs there are per tile, segmented reduction (reduece by key)
    //reduce by keys, make a transform iterator, divide by k the number of pixels per tile
   const int N = (img_w*img_h) /256;
   const int K = 256;
   thrust::device_vector<mytype> sums(N);
   thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<mytype>(0), _1/K), thrust::make_transform_iterator(thrust::counting_iterator<mytype>(N*K), _1/K), num_runs.begin(), thrust::discard_iterator<mytype>(), sums.begin());
   thrust::copy_n(sums.begin(), 3, std::ostream_iterator<mytype>(std::cout, ","));
   std::cout << std::endl;

   // use prefix sum on all those sizes, (add everything before you) which gives us the offset into our final (again doing a rduece to get the total size)
   //prcompute the offset and runline per block before doing the rle

    //what to put for size of gpu_temp here?
    //thrust::for_each_n(thrust::device, it, gpu_output.size(), finalizeTrleFunctor(gpu_temp, gpu_output, gpu_sizes, img_w, img_h, final_size));
    //trle_device = thrust::raw_pointer_cast(&trle_output[0]);

    //cudaMemcpy(buffer_size, final_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(trle, gpu_output, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //printf("TRLE Size: %u\n", *buffer_size);
    
   /*

    cudaMemcpy(rgba_gpu_input, rgba, img_w * img_h * 4, cudaMemcpyHostToDevice);
    int block_size = 256;
    int num_blocks = ((img_w * img_h / 256) + block_size - 1) / block_size;
    //printf("%d\n",img_w*img_h);
    RgbaToTrleKernel<<<num_blocks, block_size>>>(rgba_gpu_input, gpu_temp, gpu_sizes, img_w, img_h);
    cudaDeviceSynchronize();
    FinalizeTrleKernel<<<num_blocks, block_size>>>(gpu_temp, gpu_output, gpu_sizes, img_w, img_h, final_size);
    cudaMemcpy(buffer_size, final_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(trle, gpu_output, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("TRLE Size: %u\n", *buffer_size);
    //cudaMemcpy(dxt1, gpu_output, img_w * img_h / 2, cudaMemcpyDeviceToHost);*/

}

void FinalizeImageConverter()
{
    //cudaFree(rgba_gpu_input);
   // cudaFree(gpu_output);
    cudaDeviceSynchronize();
}

