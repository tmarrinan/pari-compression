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
//static uint32_t *final_size;
static int img_w;
static int img_h;
static thrust::device_vector<uint8_t> *rgba_input_ptr;
static thrust::device_vector<uint8_t> *image_output_ptr;

using namespace thrust::placeholders;

/*
__global__ void rgbaToGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height);
__global__ void rgbaToTileGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height);
__global__ void rgbaToDxt1Kernel(uint8_t *rgba, uint8_t *dxt1, int width, int height);
__global__ void rgbaToTrleKernel(uint8_t *rgba, uint8_t *trle_tmp, uint8_t *trle_size, int width, int height);
__global__ void finalizeTrleKernel(uint8_t *trle_tmp, uint8_t *trle, uint8_t *trle_size, int width, int height, uint32_t *final_size);
*/
__host__ __device__ void extractTile4x4(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[64]);
__host__ __device__ void extractTile16x16(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[1024]);
__host__ __device__ void getMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ uint16_t colorTo565(uint8_t color[3]);
__host__ __device__ uint32_t colorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
__host__ __device__ uint32_t colorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ uint32_t colorFromRgba(const uint8_t *rgba, uint32_t offset);
__host__ __device__ void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__host__ __device__ void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value);

struct GrayscaleFunctor
{
    const uint8_t *rgba;
    uint8_t *gray;
    size_t size;       
    GrayscaleFunctor(thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t>& gray_output)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        gray = thrust::raw_pointer_cast(gray_output.data());
        size = rgba_input.size() / 4;
    } 
    __host__ __device__	void operator()(int thread_id)
    {
        if(thread_id < size)
        {
            float red = (float)rgba[4 * thread_id + 0];
            float green = (float)rgba[4 * thread_id + 1];
            float blue = (float)rgba[4 * thread_id + 2];
            gray[thread_id] = (uint8_t)(0.299f * red + 0.587f * green + 0.114f * blue);
        }
    }
};

struct Dxt1Functor
{
    const uint8_t *rgba;
    uint8_t *dxt1;
    int width;
    size_t size;
    Dxt1Functor(int width_input, thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t>& dxt1_output)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        dxt1 = thrust::raw_pointer_cast(dxt1_output.data());
        width = width_input;
        size = rgba_input.size() / 16;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < size)
        {
            uint8_t tile[64];
            uint8_t color_min[3];
            uint8_t color_max[3];

      	    int tile_x = thread_id % (width / 4);
            int tile_y = thread_id / (width / 4);
            int px_x = tile_x * 4;
            int px_y = tile_y * 4;

            uint32_t offset = (px_y * width * 4) + (px_x * 4);
            uint32_t write_pos = (tile_y * (width / 4) * 8) + (tile_x * 8);

            extractTile4x4(offset, rgba, width, tile);
            getMinMaxColors(tile, color_min, color_max);
            writeUint16(dxt1, write_pos, colorTo565(color_max));
       	    writeUint16(dxt1, write_pos + 2, colorTo565(color_min));
       	    writeUint32(dxt1, write_pos + 4, colorIndices(tile, color_min, color_max));
        }
    }
};

struct TrleFunctor
{
    const uint8_t *rgba;
    uint8_t *runs; // boolean array, one index per pixel (1 indicates start of a new run)
    int width;
    int height;
    TrleFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t>& runs_output)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        runs = thrust::raw_pointer_cast(runs_output.data());
        width = width_input;
        height = height_input;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        // inner_ (x and y indices within its tile)
        if (thread_id < (width * height))
        {
            int px_x = thread_id  % width;
            int px_y = thread_id / width;

            int tile_x = px_x / 16;
            int tile_y = px_y / 16;
            int tile_idx = tile_y * (width / 16) + tile_x;

            int inner_x = px_x % 16;
            int inner_y = px_y % 16;

            uint32_t color;
            uint32_t color_prev;
            uint32_t prev;
			
            // first pixel in tile always starts new run
            if(inner_x == 0 && inner_y == 0) 
            {
                runs[(tile_idx * 256)] = 1;
            }
            else
            {
                prev = thread_id - 1;
                if (inner_x == 0) // on new row; go to last pixel in tile on previous row
                {
                    prev = (thread_id - width) + 15;
                }	

                color = colorFromRgba(rgba, thread_id);
                color_prev = colorFromRgba(rgba, prev);

                // index so a block is consecutive
                runs[(tile_idx * 256) + (inner_y * 16) + inner_x] = (uint8_t)(color_prev != color);
            }
        }
    }
};

struct FinalizeTrleFunctor
{
    const uint8_t *rgba;
    const uint8_t *runs; // boolean array, one index per pixel (1 indicates start of a new run)
    const uint16_t *num_runs; // total number of runs per tile
    const uint16_t *sums; // prefix sum array of num_runs
    uint8_t *trle;
    int width;
    int height;
    FinalizeTrleFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t> const& runs_input, thrust::device_vector<uint16_t>& num_runs_input, thrust::device_vector<uint16_t> const& sums_input, thrust::device_vector<uint8_t>& trle_output)  
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        runs = thrust::raw_pointer_cast(runs_input.data());
        num_runs = thrust::raw_pointer_cast(num_runs_input.data());
        sums = thrust::raw_pointer_cast(sums_input.data());
        trle = thrust::raw_pointer_cast(trle_output.data());
        width = width_input;
        height = height_input;
    }
    __host__  __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < (width * height / 256))
        {
            int i;
            int tile_x = thread_id % (width / 16);
            int tile_y = thread_id / (width / 16);
            int px_x = tile_x * 16;
            int px_y = tile_y * 16;

			uint8_t x_increase = 0;
            uint32_t y_increase = 0;

            // rgba index of first pixel in our current tile
            uint32_t offset = (px_y * width * 4) + (px_x * 4);

            uint32_t run_count = 1;
            uint32_t total_run_count = 0;

            // number of pixels in past tiles (index into runs)
            uint32_t index = (tile_x * 256) + (tile_y * (width / 16) * 256) ;

            // for all the runs in the tile
            for (i = 0; i < num_runs[thread_id]; i++)
            {
                // go to index of next run and reset run_count
                index++;
                run_count = 1;

                // while pixel is the same color increase the count
                while (runs[index] == 0)
                {
                    run_count++;
                    index++;
                    total_run_count++;
                }
                total_run_count++;

                // trle indexed by block
				trle[((sums[thread_id] - num_runs[thread_id]) * 4) + (i * 4)] = run_count;
				trle[((sums[thread_id] - num_runs[thread_id]) * 4) + ((i * 4) + 1)] = rgba[offset + y_increase + x_increase];
				trle[((sums[thread_id] - num_runs[thread_id]) * 4) + ((i * 4) + 2)] = rgba[offset + y_increase + x_increase + 1];
				trle[((sums[thread_id] - num_runs[thread_id]) * 4) + ((i * 4) + 3)] = rgba[offset + y_increase + x_increase + 2];
                x_increase = fmodf((float)total_run_count, 16.0) * 4;
                y_increase = (total_run_count / 16) * 4 * width;
            }
        }
    }
};

struct DecryptTrleFunctor
{
    const uint8_t *trle;
    const uint8_t *sums; // prefix sum array of num_runs
    uint8_t *rgba;
    uint32_t size;
    int width;
    int height;
    DecryptTrleFunctor(int width_input, int height_input, uint32_t size_input, thrust::device_vector<uint8_t> const& trle_input, thrust::device_vector<uint8_t> const& sums_input, thrust::device_vector<uint8_t>& rgba_output)
    {
        trle = thrust::raw_pointer_cast(trle_input.data());
        sums = trle = thrust::raw_pointer_cast(sums_input.data());
        rgba = thrust::raw_pointer_cast(rgba_output.data());
        size = size_input;
        width = width_input;
        height = height_input;
    }
    __host__  __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < size)
        {
            /*int tile_x = thread_id % (width / 16);
            int tile_y = thread_id / (width / 16);
            int px_x = tile_x * 16;
            int px_y = tile_y * 16;

            int index_runs;
            int runs;
            int tile_runs;
            int j = 1;
            int i = 0;
            int x_increase = 0;
            int y_increase = 0;

            // index of first pixel in our current tile
            uint32_t offset = (px_y * width * 4) + (px_x * 4);

            if(tid ==0)
			{
				index_runs =0;
				runs = trle[0];
				tile_runs = runs;
			}
			else
			{
				index_runs = total_runs[tid-1]*4;
				runs = trle[index_runs];
				tile_runs = trle[tid] -trle[index_runs]; 
			}
					

			//while still runs in tile
			while(j<tile_runs)
			{
				//printf("offset: %d, x_increase: %d, y_increase: %d\n",offset,x_increase,y_increase);
				
				//for all in the current run
				while(i<runs)
				{
					x_increase = (i/16) *width *4;
					y_increase = (i%16)*4;
					//if(i==3)
					//{
					//	printf("rgba: %d,%d,%d\n",trle[index_runs+1],trle[index_runs+2],trle[index_runs+3]);
					//	printf("x: %d, y%d\n", y_increase, x_increase);
					//}
					rgba[offset +x_increase + y_increase] = trle[index_runs+1];
					rgba[offset +x_increase + y_increase+1] = trle[index_runs +2];
					rgba[offset +x_increase + y_increase+2] = trle[index_runs +3];
					rgba[offset +x_increase + y_increase+3] = 255; //set alpha to full	
					i++;
				}
				
				//increment to next run
				runs = trle[index_runs +j*4]; //increment by number of runs weve done so far in this tile
				index_runs = index_runs+j*4 +1;

				j++;
			}*/
        }
    }
};



/*
__global__ void rgbaToGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height)
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

__global__ void rgbaToTileGrayscaleKernel(uint8_t *rgba, uint8_t *gray, int width, int height)
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

__global__ void rgbaToDxt1Kernel(uint8_t *rgba, uint8_t *dxt1, int width, int height)
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

__global__ void rgbaToTrleKernel(uint8_t *rgba, uint8_t *trle_tmp, uint8_t *trle_size, int width, int height)
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

__global__ void finalizeTrleKernel(uint8_t *trle_tmp, uint8_t *trle, uint8_t *trle_size, int width, int height, uint32_t *final_size)
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
*/

void extractTile4x4(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[64])
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

void extractTile16x16(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[1024])
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

void getMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3])
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

uint16_t colorTo565(uint8_t color[3])
{
    return ((color[0] >> 3) << 11) | ((color[1] >> 2) << 5) | (color[2] >> 3);
}

uint32_t colorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset)
{
    int dx = tile[t_offset + 0] - colors[c_offset + 0];
    int dy = tile[t_offset + 1] - colors[c_offset + 1];
    int dz = tile[t_offset + 2] - colors[c_offset + 2];
    
    return (dx*dx) + (dy*dy) + (dz*dz);
}

uint32_t colorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3])
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
            dist = colorDistance(tile, i * 4, colors, j * 4);
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

uint32_t colorFromRgba(const uint8_t *rgba, uint32_t offset)
{
    
    uint32_t result = rgba[offset * 4] << 24;
    result |= rgba[offset * 4 + 1] << 16;
    result |= rgba[offset * 4 + 2] << 8;
    result |= rgba[offset * 4 + 3];
    return result;
}

void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value)
{
   buffer[offset + 0] = value & 0xFF;
   buffer[offset + 1] = (value >> 8) & 0xFF;
}

void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value)
{
    buffer[offset + 0] = value & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
}

// ----------------------------------------------------------------------------- //

void initImageConverter(int width, int height)
{
    img_w = width;
    img_h = height;

    rgba_input_ptr = new thrust::device_vector<uint8_t>(img_w * img_h * 4);
    image_output_ptr = new thrust::device_vector<uint8_t>(img_w * img_h * 4);
    
    //cudaMalloc((void**)&rgba_gpu_input, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_temp, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_sizes, img_w * img_h / 256);
    //cudaMalloc((void**)&gpu_output, img_w * img_h * 4);
    //cudaMalloc((void**)&final_size, sizeof(uint32_t));
}

void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray)
{
    thrust::copy(rgba, rgba + (img_w * img_h * 4), rgba_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per pixel
    thrust::for_each_n(thrust::device, it, img_w * img_h, GrayscaleFunctor(*rgba_input_ptr, *image_output_ptr));

    // copy back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (img_w * img_h), gray);
}

void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1)
{
    thrust::copy(rgba, rgba + (img_w * img_h * 4), rgba_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per 4x4 tile
    thrust::for_each_n(thrust::device, it, img_w * img_h / 16, Dxt1Functor(img_w, *rgba_input_ptr, *image_output_ptr));

    // copy back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (img_w * img_h / 2), dxt1);
}

void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint8_t *run_offsets)
{
    thrust::copy(rgba, rgba + (img_w * img_h * 4), rgba_input_ptr->begin());
    thrust::device_vector<uint8_t> runs(img_w * img_h); // move to static global declaration
    thrust::device_vector<uint16_t> num_runs(img_w * img_h / 256); // move to static global declaration
    thrust::device_vector<uint16_t> sums(img_w * img_h / 256); // move to static global declaration
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per pixel
    thrust::for_each_n(thrust::device, it, img_w * img_h, TrleFunctor(img_w, img_h, *rgba_input_ptr, runs));

    // thrust reduce_by_key - sum number of new runs per tile
    const int n = (img_w * img_h) / 256; // number of tiles
    const int k = 256; // pixels per tile
    thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(0), _1 / k), thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(n * k), _1 / k), runs.begin(), thrust::discard_iterator<uint16_t>(), num_runs.begin());

    // thrust inclusive_scan (prefix sum) - create array where each index is sum of all numbers in num_runs before its index
    // results in the offset into our final trle array
    thrust::inclusive_scan(num_runs.begin(), num_runs.end(), sums.begin());

    // thrust for_each_n - one thread per 16x16 tile
    thrust::for_each_n(thrust::device, it, img_w * img_h / 256, FinalizeTrleFunctor(img_w, img_h, *rgba_input_ptr, runs, num_runs, sums, *image_output_ptr));

    // copy total size to host
    uint32_t total_runs;
    thrust::copy(sums.end() - 1, sums.end(), &total_runs);
    *buffer_size = total_runs * 4;
    printf("TRLE Size: %u\n", *buffer_size);

    // copy data back to host
    //thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (img_w * img_h / 2), dxt1);
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
}

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
    uint32_t  *trle_size;

    trle_device = thrust::raw_pointer_cast(&gpu_output[0]);
    runs_device = thrust::raw_pointer_cast(&total_runs[0]);
    
   
    thrust::host_vector<uint32_t> output_size(img_w*img_h/256);
    thrust::copy_n(total_runs.begin(), img_w*img_h/256, output_size.begin());
    *buffer_size = output_size[(img_w*img_h/256) -1];//thrust::raw_pointer_cast(&output_size[(img_w*img_h/256) -1]);
   
    cudaMemcpy(run_offsets, runs_device, ((img_w*img_h)/256)*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(trle, trle_device, output_size[(img_w*img_h/256) -1] *4* sizeof(uint8_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(buffer_size, trle_size,  sizeof(uint32_t), cudaMemcpyDeviceToHost);


    printf("TRLE Size: %u\n", output_size[(img_w*img_h/256) -1]);
    //for(int i=0; i<output_size[(img_w*img_h/256) -1] *4; i=i+4)
    //{
	//    printf("%d,%d,%d,%d\n", trle[i], trle[i+1], trle[i+2], trle[i+3]);
    //}
}

void TrleToRgba(uint8_t *rgba_trle, uint8_t *trle, uint32_t *buffer_size, uint8_t *run_offsets)
{
    thrust::device_vector<uint8_t> trle_input(trle, trle+( *buffer_size*4));	
    thrust::device_vector<uint8_t> total_runs_input(run_offsets, run_offsets+img_w*img_h/256);
	
    //error when trying to acesss run offsets above and below
    thrust::device_vector<uint8_t> rgba_output(img_w * img_h*4);
    thrust::counting_iterator<size_t> it(0);
    
    thrust::for_each_n(thrust::device, it, (img_w*img_h)/256, decryptTrleFunctor(trle_input, rgba_output,buffer_size,total_runs_input,img_w, img_h));
    uint8_t *rgba_device;

    rgba_device = thrust::raw_pointer_cast(&rgba_output[0]);
    cudaMemcpy(rgba_trle, rgba_device, (img_w*img_h*4)* sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //for(int i=0; i<img_w*3; i=i+3)
    //{
	//    printf("%d,%d,%d\n", rgba_trle[i], rgba_trle[i+1], rgba_trle[i+2]);
    //}
}
*/

void finalizeImageConverter()
{
    //cudaFree(rgba_gpu_input);
    // cudaFree(gpu_output);
    cudaDeviceSynchronize();
}

