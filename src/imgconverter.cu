#include "imgconverter.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <cstdio>

static uint8_t *rgba_gpu_input;
static uint8_t *gpu_temp;
static uint8_t *gpu_sizes;
static uint8_t *gpu_output;
static uint32_t *final_size;
static int img_w;
static int img_h;

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
	uint8_t *trle_tmp;
       	uint8_t *trle_size;
        size_t size_img;
        trleFunctor(thrust::device_vector<uint8_t>& rgba_input,uint8_t *tmp_input, uint8_t *size_input, int width_input, int height_input)
        {
          rgba = thrust::raw_pointer_cast(rgba_input.data());
          width = width_input;
	  height = height_input;
	  trle_tmp = tmp_input;
	  trle_size = size_input;
          //size_img = trle_output.size();
        }
        __device__
        void operator()(int x)
        {
		 if (x < width*height / 256)
   		 {
        		uint8_t tile[1024];

        		int tile_x = x % (width / 16);
	        	int tile_y = x / (width / 16);
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
        		trle_size[x] = size;
        		printf("bye trle\n");
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
    		if (x < width*height / 256)
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
    img_w = width;
    img_h = height;
    //cudaMalloc((void**)&rgba_gpu_input, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_temp, img_w * img_h * 4);
    //cudaMalloc((void**)&gpu_sizes, img_w * img_h / 256);
    //cudaMalloc((void**)&gpu_output, img_w * img_h * 4);
    //cudaMalloc((void**)&final_size, sizeof(uint32_t));
}

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
    printf("before thrust n \n");
    
    thrust::for_each_n(thrust::device, it, rgba_input.size()/16, dxt1Functor(img_w, rgba_input, dxt1_output));
    dxt1_device = thrust::raw_pointer_cast(&dxt1_output[0]);
    cudaMemcpy(dxt1, dxt1_device, img_w * img_h / 2, cudaMemcpyDeviceToHost);
    
    printf("after\n");
    /*cudaMemcpy(rgba_gpu_input, rgba, img_w * img_h * 4, cudaMemcpyHostToDevice);
    int block_size = 256;
    int num_blocks = ((img_w * img_h / 16) + block_size - 1) / block_size;
    RgbaToDxt1Kernel<<<num_blocks, block_size>>>(rgba_gpu_input, gpu_output, img_w, img_h);
    cudaMemcpy(dxt1, gpu_output, img_w * img_h / 2, cudaMemcpyDeviceToHost);*/
}

void RgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size)
{
    /*uint32_t *trle_device;
    thrust::device_vector<uint8_t> rgba_input(rgba, rgba+img_w*img_h*4);
    //in worst case no compression? all pixels are different
    thrust::device_vector<uint32_t> trle_output(img_w*img_h*4);
    thrust::counting_iterator<size_t> it(0);
//use thrust reduce to sum how many runs there are per tile
// use prefix sum on all those sizes, (add everything before you) which gives us the offset into our final (again doing a rduece to get the total size)
//prcompute the offset and runline per block before doing the rle

    //what to put for size of gpu_temp here?
    thrust::for_each_n(thrust::device, it, trle_output.size(), trleFunctor(rgba_input, gpu_temp, gpu_sizes, img_w, img_h));
    //thrust::for_each_n(thrust::device, it, trle_output.size(), finalizeTrleFunctor(gpu_temp, gpu_output, gpu_sizes, img_w, img_h, final_size));
    //trle_device = thrust::raw_pointer_cast(&trle_output[0]);
    //cudaMemcpy(buffer_size, trle_device, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    */

    cudaMemcpy(rgba_gpu_input, rgba, img_w * img_h * 4, cudaMemcpyHostToDevice);
    int block_size = 256;
    int num_blocks = ((img_w * img_h / 256) + block_size - 1) / block_size;
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
    cudaFree(rgba_gpu_input);
    cudaFree(gpu_output);
    cudaDeviceSynchronize();
}

