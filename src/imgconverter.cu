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
#include <math.h>

/*static uint8_t *rgba_gpu_input;
static uint8_t *gpu_temp;
static uint8_t *gpu_sizes;
static uint8_t *gpu_output;*/
static uint32_t *final_size;
static int img_w;
static int img_h;


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
	//uint8_t *trle_tmp;
       	//uint8_t *trle_size;
       // size_t size_img;
	uint8_t *num_runs;
        trleFunctor(thrust::device_vector<uint8_t>& rgba_input, int width_input, int height_input, thrust::device_vector<uint8_t>& num_runs_input)
        {
          rgba = thrust::raw_pointer_cast(rgba_input.data());
          width = width_input;
	  height = height_input;
	  //trle_tmp = thrust::raw_pointer_cast(tmp_input.data());
	  //trle_size = thrust::raw_pointer_cast(size_input.data()); 
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
		
			
			//first in tile
			if(tile_x == 0 && tile_y == 0 ) 
			{
				//printf("tid: %d, tile: %d, in tile:  %d\n",tid, tile_id,tile_id *256 + tile_y *16 + tile_x );
				num_runs[tile_id *256 + tile_y *16 + tile_x] = 1;
			}

			else
			{
				prev = tid-1;
	
				//on new row
				if (tile_x == 0)
				{
					prev = (tid-width) + 15;
				}	
				
				//if previous color (withIn tile) is same as current
				color = ColorFromRgba(rgba,tid);
				color_prev = ColorFromRgba(rgba, prev);

				
				//make so a block is consequtive
				num_runs[tile_id *256 + tile_y *16 + tile_x] = (uint8_t)(color_prev != color);
			}
		}
        }
};

struct finalizeTrleFunctor
{
	/* num_runs = 1 and 0 array, mapping to one pixel in image, 0 is run 1 is new
	   sums = total runs per tile
	   total_runs = prefix sum array of all runs before current
	*/
        uint8_t *rgba;
        uint8_t *output;
	uint16_t *sums;
	uint8_t *total_runs;
        int width;
	int height;
	uint8_t *num_runs;

        finalizeTrleFunctor(thrust::device_vector<uint8_t>& rgba_input,thrust::device_vector<uint8_t>& trle_output, thrust::device_vector<uint8_t>& num_runs_input, thrust::device_vector<uint16_t>& sums_input, thrust::device_vector<uint8_t>& runs_input, int width_input, int height_input)
        {
           rgba = thrust::raw_pointer_cast(rgba_input.data());
	   output = thrust::raw_pointer_cast(trle_output.data());
           sums = thrust::raw_pointer_cast(sums_input.data());
	   total_runs = thrust::raw_pointer_cast(runs_input.data());
	   width = width_input;
	   height = height_input;
 	   num_runs = thrust::raw_pointer_cast(num_runs_input.data());

        }
        __device__
        void operator()(int tid)
        {
		//one per tile
    		if (tid < width*height /256)
    		{
			
        		int tile_x = tid % (width / 16);
        		int tile_y = tid / (width / 16);
        		int px_x = tile_x * 16;
        		int px_y = tile_y * 16;

        		int i;
			uint32_t y_increase = 0;
			uint8_t x_increase=0;

			//index of first pixel in our current tile
        		uint32_t offset = (px_y * width * 4) + (px_x * 4);
		
			uint32_t run_count =1;
			uint32_t total_run_count =0;

			//number of pixels in past tiles (index into num_runs)
			uint32_t index = tile_x*256 + tile_y*(width/16) *256 ;
			
			//for all the runs in the tile
			for ( i=0; i<sums[tid]; i++)
			{
				//go to index of next run
				//reset run_count
				index = index+1;
				run_count = 1;
				
				
				//while it is the same color increase the count
				while(num_runs[index] ==0)
				{
					run_count++;
					index++;
					total_run_count++;
					
				}
				total_run_count++;
			 
		
				output[(total_runs[tid]-sums[tid]) *4 + i*4] = run_count;
				//minus 4 becuase this actually indexes into the start of the next run.
				output[(total_runs[tid]-sums[tid]) *4 + i*4 + 1] = rgba[offset+ y_increase + x_increase];
				output[(total_runs[tid]-sums[tid]) *4 + i*4 + 2] = rgba[offset +  y_increase + x_increase+1];
				output[(total_runs[tid]-sums[tid]) *4 + i*4 + 3] = rgba[offset +  y_increase + x_increase+2];
				 x_increase = fmodf((float)total_run_count,16.0) *4;
                                y_increase = (total_run_count/16)*4*width;
				
			
                                
			}		
        	}
	}
};



struct decryptTrleFunctor
{
        uint8_t *trle;
        uint8_t *rgba;
	uint32_t size;
	uint8_t *total_runs;
        int width;
	int height;


        decryptTrleFunctor(thrust::device_vector<uint8_t>& rgba_output,thrust::device_vector<uint8_t>& trle_input, uint32_t * size_input ,thrust::device_vector<uint8_t>&total_runs_input, int width_input, int height_input)
        {
           rgba = thrust::raw_pointer_cast(rgba_output.data());
	   trle = thrust::raw_pointer_cast(trle_input.data());
	//   total_runs = thrust::raw_pointer_cast(total_runs_input.data());
	   size = *size_input;
	   width = width_input;
	   height = height_input;

        }
        __device__
        void operator()(int tid)
        {
		//one per tile
    		if (tid < size)
    		{
			
			int tile_x = tid % (width / 16);
        		int tile_y = tid / (width / 16);
        		int px_x = tile_x * 16;
        		int px_y = tile_y * 16;
			
		
			printf("tid %d\n", tid);
		 	int runs = trle[0];
                        int color_index = trle[1];
			if(tid ==0)
			{
				runs = trle[total_runs[tid-1]*4];
                                color_index = trle[total_runs[tid-1]*4 +1];
			}
			
			printf("runs : %d color index: %d, trle index %d\n " ,runs, color_index, total_runs[0]);
			int curRun = runs;
			int runsSoFar = runs;
			int total_runs = runs;
			int j=1;
			int i=0;
			int x_increase =0;
			int y_increase =0;
			//index of first pixel in our current tile
                        uint32_t offset = (px_y * width * 4) + (px_x * 4);
			/*
			//while still runs in tile
			while(j<total_runs[tid])
			{
				//for all in the current run
				while(i<curRun)
				{
					x_increase = (i/16) *width *4;
					y_increase = (i%16)*4;
					rgba[offset +x_increase + y_increase] = color_index;
					rgba[offset +x_increase + y_increase+1] = color_index+1;
					rgba[offset +x_increase + y_increase+2  ] = color_index+2;	
					i++;
				}
				//increment to next run
				runs = trle[total_runs[tid-1]*4 +j*4]; //increment by number of runs weve done so far in this tile
				color_index = trle[total_runs[tid-1]*4 +j*4];
				j++;
			}*/

				/*runs = trle[tid+1];
				if(runsSoFar != 256)
				{
					runsSoFar = runsSoFar + curRun;
				}
				else
				{
					runsSoFar =0;
					if(offset + 16 < width)
					{
						offset = offset +256;
					}
					else
					{
						offset = offset + width*256*4;
					}
				}
				total_runs = total_runs + runs;

				*/
				
		
			                               
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
    img_w = width;
    img_h = height;
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

void RgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint8_t *run_offsets)
{
    
    thrust::device_vector<uint8_t> rgba_input(rgba, rgba+img_w*img_h*4);
    thrust::device_vector<uint8_t> gpu_output(img_w * img_h*4);
    //thrust::device_vector<uint8_t> gpu_sizes(img_w*img_h*4);
    thrust::device_vector<uint8_t> num_runs(img_w*img_h);
    thrust::device_vector<uint8_t> total_runs(img_w*img_h/256);
    thrust::counting_iterator<size_t> it(0);
    thrust::for_each_n(thrust::device, it, rgba_input.size()/4, trleFunctor(rgba_input, img_w, img_h,num_runs));
    
    
    //reduce to sum how many runs there are per tile, segmented reduction (reduece by key)
    //reduce by keys, make a transform iterator, divide by k the number of pixels per tile
    const int N = (img_w*img_h) /256;
    const int K = 256;
    thrust::device_vector<uint16_t> sums(N);
    thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(0), _1/K), thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(N*K), _1/K), num_runs.begin(), thrust::discard_iterator<uint16_t>(), sums.begin());
   
    thrust::copy_n(sums.begin(), (img_w*img_h)/256, std::ostream_iterator<uint16_t>(std::cout, ","));
    std::cout << std::endl;

    // use prefix sum on all those sizes, (add everything before you) which gives us the offset into our final (again doing a rduece to get the total size)
    thrust::inclusive_scan(sums.begin(), sums.end(), total_runs.begin());
     
    thrust::copy_n(total_runs.begin(), img_w*img_h/256, std::ostream_iterator<uint32_t>(std::cout, ","));
    std::cout << std::endl;
     
    


    //precompute the offset and runline per block before doing the rle
    //we only need one for each tile so input/4/256
    thrust::for_each_n(thrust::device, it, rgba_input.size()/1024, finalizeTrleFunctor(rgba_input, gpu_output, num_runs, sums,total_runs,img_w, img_h));
   
    uint8_t *trle_device;
    uint8_t *runs_device;

    trle_device = thrust::raw_pointer_cast(&gpu_output[0]);
    runs_device = thrust::raw_pointer_cast(&total_runs[0]);

    thrust::host_vector<uint32_t> output_size(img_w*img_h/256);
    thrust::copy_n(total_runs.begin(), img_w*img_h/256, output_size.begin());
    uint32_t  *trle_size = thrust::raw_pointer_cast(&output_size[(img_w*img_h/256) -1]);
    printf("trle size 1 : %u\n", *trle_size);
    cudaMemcpy(run_offsets, runs_device, ((img_w*img_h)/256)*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(trle, trle_device, output_size[(img_w*img_h/256) -1] *4* sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_size, trle_size,  sizeof(uint32_t), cudaMemcpyDeviceToHost);


    printf("TRLE Size: %u\n", output_size[(img_w*img_h/256) -1]);
  /*  for(int i=0; i<output_size[(img_w*img_h/256) -1] *4; i=i+4)
    {
	    printf("%d,%d,%d,%d\n", trle[i], trle[i+1], trle[i+2], trle[i+3]);
    }
*/

}


void TrleToRgba(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint8_t *run_offsets)
{

	//size of total runs correct??
    thrust::device_vector<uint8_t> trle_input(trle, trle+ *buffer_size*4);	
    //thrust::device_vector<uint8_t> total_runs_input(run_offsets, run_offsets+img_w*img_h/256);
	//error when trying to acesss run offsets above and below
    printf("run_offests %u\n",run_offsets[0]);

    printf("trle input size %d\n",trle_input.size());
    printf("buffer size %d\n", *buffer_size);
    thrust::device_vector<uint8_t> rgba_output(img_w * img_h*4);
    thrust::counting_iterator<size_t> it(0);
    
    /*thrust::copy_n(total_runs_input.begin(), (img_w*img_h)/256, std::ostream_iterator<uint16_t>(std::cout, ","));
    std::cout << std::endl;
    */
    /*thrust::copy_n(trle_input.begin(),39, std::ostream_iterator<uint16_t>(std::cout, ","));
    std::cout << std::endl;*/
    //thrust::for_each_n(thrust::device, it, (img_w*img_h)/256, decryptTrleFunctor(trle_input, rgba_output,buffer_size,total_runs_input,img_w, img_h));

/*
    for(int i=0; i<img_w; i=i+3)
    {
	    printf("%d,%d,%d\n", rgba_output[i], rgba_output[i+1], rgba_output[i+2]);
    }
*/
}
void FinalizeImageConverter()
{
    //cudaFree(rgba_gpu_input);
   // cudaFree(gpu_output);
    cudaDeviceSynchronize();
}

