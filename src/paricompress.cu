#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include "paricompress.h"

static double _compute_time;
static double _mem_transfer_time;
static double _total_time;

__host__ __device__ static void extractTile4x4(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[64]);
__host__ __device__ static void getMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ static uint16_t colorTo565(uint8_t color[3]);
__host__ __device__ static uint32_t colorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
__host__ __device__ static uint32_t colorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ static void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__host__ __device__ static void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value);

__device__ static void extractCGTile4x4(uint32_t offset_x, uint32_t offset_y, const cudaSurfaceObject_t pixels, uint8_t out_tile[64]);

static uint64_t currentTime();


// CUDA Thrust transformer to change data type
template<typename T1, typename T2>
struct typecast
{
    __host__ __device__ T2 operator()(const T1 &x) const
    {
        return static_cast<T2>(x);
    }
};

// CUDA Thrust functors
struct PariGrayscaleFunctor
{
    const uint8_t *rgba;
    uint8_t *gray;
    size_t size;       
    PariGrayscaleFunctor(thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t>& gray_output)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        gray = thrust::raw_pointer_cast(gray_output.data());
        size = rgba_input.size() / 4;
    } 
    __host__ __device__	void operator()(int thread_id)
    {
        if (thread_id < size)
        {
            float red = (float)rgba[4 * thread_id + 0];
            float green = (float)rgba[4 * thread_id + 1];
            float blue = (float)rgba[4 * thread_id + 2];
            gray[thread_id] = (uint8_t)(0.299f * red + 0.587f * green + 0.114f * blue);
        }
    }
};

struct PariDxt1Functor
{
    const uint8_t *rgba;
    uint8_t *dxt1;
    uint32_t width;
    size_t size;
    PariDxt1Functor(thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t>& dxt1_output, uint32_t width_input)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        dxt1 = thrust::raw_pointer_cast(dxt1_output.data());
        width = width_input;
        size = rgba_input.size() / 64;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        if (thread_id < size)
        {
            uint8_t tile[64];
            uint8_t color_min[3];
            uint8_t color_max[3];

            // px_ (x and y pixel indices)
            // tile_ (x and y tile indices)
      	    uint32_t tile_x = thread_id % (width / 4);
            uint32_t tile_y = thread_id / (width / 4);
            uint32_t px_x = tile_x * 4;
            uint32_t px_y = tile_y * 4;

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

struct PariActivePixelFunctor
{
    const uint8_t *rgba;
    const float *depth;
    uint8_t *new_run;
    uint8_t *is_active;
    int width;
    int height;
    float max_depth;
    PariActivePixelFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& rgba_input,
                           thrust::device_vector<float> const& depth_input, thrust::device_vector<uint8_t>& new_run_output,
                           thrust::device_vector<uint8_t>& is_active_output)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        depth = thrust::raw_pointer_cast(depth_input.data());
        new_run = thrust::raw_pointer_cast(new_run_output.data());
        is_active = thrust::raw_pointer_cast(is_active_output.data());
        width = width_input;
        height = height_input;
        max_depth = 1.0f;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        if (thread_id < width * height)
        {
            // whether or not pixel is active
            is_active[thread_id] = (uint8_t)(depth[thread_id] != max_depth);
            
            // whether or not pixel starts a new run
            if (thread_id == 0)
            {
                new_run[thread_id] = 1;
            }
            else
            {
                uint8_t prev_active = (uint8_t)(depth[thread_id - 1] != max_depth);
                new_run[thread_id] = (uint8_t)(is_active[thread_id] != prev_active);
            }
        }
    }
};

struct PariActivePixelFinalizeFunctor
{
    const uint8_t *rgba;
    const float *depth;
    uint8_t *is_active;
    uint8_t *new_run;
    uint32_t *run_index;
    uint32_t *run_length;
    uint32_t *active_index;
    int width;
    int height;
    uint8_t *compressed;
    uint32_t *compressed_size;
    float max_depth;
    PariActivePixelFinalizeFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& rgba_input,
                                   thrust::device_vector<float> const& depth_input, thrust::device_vector<uint8_t>& is_active_input,
                                   thrust::device_vector<uint8_t>& new_run_input, thrust::device_vector<uint32_t>& run_idx_input,
                                   thrust::device_vector<uint32_t>& run_length_input, thrust::device_vector<uint32_t>& active_idx_input,
                                   thrust::device_vector<uint8_t>& output, thrust::device_vector<uint32_t>& output_size)
    {
        rgba = thrust::raw_pointer_cast(rgba_input.data());
        depth = thrust::raw_pointer_cast(depth_input.data());
        is_active = thrust::raw_pointer_cast(is_active_input.data());
        new_run = thrust::raw_pointer_cast(new_run_input.data());
        run_index = thrust::raw_pointer_cast(run_idx_input.data());
        run_length = thrust::raw_pointer_cast(run_length_input.data());
        active_index = thrust::raw_pointer_cast(active_idx_input.data());
        compressed = thrust::raw_pointer_cast(output.data());
        compressed_size = thrust::raw_pointer_cast(output_size.data());
        width = width_input;
        height = height_input;
        max_depth = 1.0f;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        if(thread_id < width * height && is_active[thread_id]) // active pixels only
        {
            uint32_t write_pos = 8 * (active_index[thread_id] + ((run_index[thread_id] - 1) / 2) + 1);
            
            memcpy(compressed + write_pos, rgba + (4 * thread_id), 4);
            memcpy(compressed + write_pos + 4, depth + thread_id, 4);
            if (new_run[thread_id] == 1)
            {
                uint32_t num_inactive = (run_index[thread_id] > 1) ? run_length[run_index[thread_id] - 2] : 0;
                uint32_t num_active = run_length[run_index[thread_id] - 1];
                memcpy(compressed + write_pos - 8, &num_inactive, 4);
                memcpy(compressed + write_pos - 4, &num_active, 4);
            }
        }
        if (thread_id == (width * height) - 1) // final pixel - write compressed size
        {
            uint32_t active_run = run_index[thread_id] + is_active[thread_id] - 2;
            uint32_t write_pos = 8 * (active_index[thread_id] + (active_run / 2) + 1);
        
            compressed_size[0] = write_pos + 8;
            if (is_active[thread_id] == 0)
            {
                uint32_t num_inactive = run_length[run_index[thread_id] - 1];
                uint32_t num_active = 0;
                memcpy(compressed + write_pos, &num_inactive, 4);
                memcpy(compressed + write_pos + 4, &num_active, 4);
            }
        }
    }
};

struct PariCGGrayscaleFunctor
{
    cudaSurfaceObject_t rgba;
    uint8_t *gray;
    uint32_t width;
    uint32_t height;
    PariCGGrayscaleFunctor(cudaSurfaceObject_t const& rgba_input, thrust::device_vector<uint8_t>& gray_output,
                           uint32_t width_input, uint32_t height_input)
    {
        rgba = rgba_input;
        gray = thrust::raw_pointer_cast(gray_output.data());
        width = width_input;
        height = height_input;
    } 
    __device__	void operator()(int thread_id)
    {
        if (thread_id < (width * height))
        {  
            uchar4 color;
            surf2Dread(&color, rgba, 4 * (thread_id % width), thread_id / width);
            gray[thread_id] = (uint8_t)(0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
        }
    }
};

struct PariCGDxt1Functor
{
    cudaSurfaceObject_t rgba;
    uint8_t *dxt1;
    uint32_t width;
    uint32_t height;
    PariCGDxt1Functor(cudaSurfaceObject_t const& rgba_input, thrust::device_vector<uint8_t>& dxt1_output,
                      uint32_t width_input, uint32_t height_input)
    {
        rgba = rgba_input;
        dxt1 = thrust::raw_pointer_cast(dxt1_output.data());
        width = width_input;
        height = height_input;
    }
    __device__ void operator()(int thread_id)
    {
        if (thread_id < (width * height / 16))
        {
            uint8_t tile[64];
            uint8_t color_min[3];
            uint8_t color_max[3];

            // px_ (x and y pixel indices)
            // tile_ (x and y tile indices)
      	    uint32_t tile_x = thread_id % (width / 4);
            uint32_t tile_y = thread_id / (width / 4);
            uint32_t px_x = tile_x * 4;
            uint32_t px_y = tile_y * 4;

            uint32_t write_pos = (tile_y * (width / 4) * 8) + (tile_x * 8);

            extractCGTile4x4(px_x, px_y, rgba, tile);
            getMinMaxColors(tile, color_min, color_max);
            writeUint16(dxt1, write_pos, colorTo565(color_max));
       	    writeUint16(dxt1, write_pos + 2, colorTo565(color_min));
       	    writeUint32(dxt1, write_pos + 4, colorIndices(tile, color_min, color_max));
        }
    }
};

struct PariCGActivePixelFunctor
{
    cudaSurfaceObject_t depth;
    uint8_t *new_run;
    uint8_t *is_active;
    int width;
    int height;
    float max_depth;
    PariCGActivePixelFunctor(cudaSurfaceObject_t const& depth_input, thrust::device_vector<uint8_t>& new_run_output,
                             thrust::device_vector<uint8_t>& is_active_output, int width_input, int height_input)
    {
        depth = depth_input;
        new_run = thrust::raw_pointer_cast(new_run_output.data());
        is_active = thrust::raw_pointer_cast(is_active_output.data());
        width = width_input;
        height = height_input;
        max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        if (thread_id < width * height)
        {
            float px_depth;
            surf2Dread(&px_depth, depth, 4 * (thread_id % width), thread_id / width);
            
            // whether or not pixel is active
            is_active[thread_id] = (uint8_t)(px_depth != max_depth);
            
            // whether or not pixel starts a new run
            if (thread_id == 0)
            {
                new_run[thread_id] = 1;
            }
            else
            {
                float prev_depth;
                surf2Dread(&prev_depth, depth, 4 * ((thread_id - 1) % width), (thread_id - 1) / width);
                
                uint8_t prev_active = (uint8_t)(prev_depth != max_depth);
                new_run[thread_id] = (uint8_t)(is_active[thread_id] != prev_active);
            }
        }
    }
};

struct PariCGActivePixelFinalizeFunctor
{
    cudaSurfaceObject_t rgba;
    cudaSurfaceObject_t depth;
    uint8_t *is_active;
    uint8_t *new_run;
    uint32_t *run_index;
    uint32_t *run_length;
    uint32_t *active_index;
    int width;
    int height;
    uint8_t *compressed;
    uint32_t *compressed_size;
    float max_depth;
    PariCGActivePixelFinalizeFunctor(cudaSurfaceObject_t const& rgba_input, cudaSurfaceObject_t const& depth_input,
                                     thrust::device_vector<uint8_t>& is_active_input, thrust::device_vector<uint8_t>& new_run_input,
                                     thrust::device_vector<uint32_t>& run_idx_input, thrust::device_vector<uint32_t>& run_length_input,
                                     thrust::device_vector<uint32_t>& active_idx_input, thrust::device_vector<uint8_t>& output,
                                     thrust::device_vector<uint32_t>& output_size, int width_input, int height_input)
    {
        rgba = rgba_input;
        depth = depth_input;
        is_active = thrust::raw_pointer_cast(is_active_input.data());
        new_run = thrust::raw_pointer_cast(new_run_input.data());
        run_index = thrust::raw_pointer_cast(run_idx_input.data());
        run_length = thrust::raw_pointer_cast(run_length_input.data());
        active_index = thrust::raw_pointer_cast(active_idx_input.data());
        compressed = thrust::raw_pointer_cast(output.data());
        compressed_size = thrust::raw_pointer_cast(output_size.data());
        width = width_input;
        height = height_input;
        max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        if(thread_id < width * height && is_active[thread_id]) // active pixels only
        {
            uint32_t write_pos = 8 * (active_index[thread_id] + ((run_index[thread_id] - 1) / 2) + 1);
            
            uchar4 px_color;
            float px_depth;
            surf2Dread(&px_color, rgba, 4 * (thread_id % width), thread_id / width);
            surf2Dread(&px_depth, depth, 4 * (thread_id % width), thread_id / width);
            memcpy(compressed + write_pos, &px_color, 4);
            memcpy(compressed + write_pos + 4, &px_depth, 4);
            if (new_run[thread_id] == 1)
            {
                uint32_t num_inactive = (run_index[thread_id] > 1) ? run_length[run_index[thread_id] - 2] : 0;
                uint32_t num_active = run_length[run_index[thread_id] - 1];
                memcpy(compressed + write_pos - 8, &num_inactive, 4);
                memcpy(compressed + write_pos - 4, &num_active, 4);
            }
        }
        if (thread_id == (width * height) - 1) // final pixel - write compressed size
        {
            uint32_t active_run = run_index[thread_id] + is_active[thread_id] - 2;
            uint32_t write_pos = 8 * (active_index[thread_id] + (active_run / 2) + 1);
        
            compressed_size[0] = write_pos + 8;
            if (is_active[thread_id] == 0)
            {
                uint32_t num_inactive = run_length[run_index[thread_id] - 1];
                uint32_t num_active = 0;
                memcpy(compressed + write_pos, &num_inactive, 4);
                memcpy(compressed + write_pos + 4, &num_active, 4);
            }
        }
    }
};

struct PariCGSubActivePixelFunctor
{
    cudaSurfaceObject_t depth;
    uint8_t *new_run;
    uint8_t *is_active;
    int texture_width;
    int texture_height;
    int texture_viewport_x;
    int texture_viewport_y;
    int texture_viewport_w;
    int texture_viewport_h;
    int ap_width;
    int ap_height;
    int ap_viewport_x;
    int ap_viewport_y;
    int ap_viewport_w;
    int ap_viewport_h;
    float max_depth;
    PariCGSubActivePixelFunctor(cudaSurfaceObject_t const& depth_input, thrust::device_vector<uint8_t>& new_run_output,
                             thrust::device_vector<uint8_t>& is_active_output, int texture_width_input, int texture_height_input,
                             int *texture_viewport_input, int ap_width_input, int ap_height_input, int *ap_viewport_input)
    {
        depth = depth_input;
        new_run = thrust::raw_pointer_cast(new_run_output.data());
        is_active = thrust::raw_pointer_cast(is_active_output.data());
        texture_width = texture_width_input;
        texture_height = texture_height_input;
        texture_viewport_x = texture_viewport_input[0];
        texture_viewport_y = texture_viewport_input[1];
        texture_viewport_w = texture_viewport_input[2];
        texture_viewport_h = texture_viewport_input[3];
        ap_width = ap_width_input;
        ap_height = ap_height_input;
        ap_viewport_x = ap_viewport_input[0];
        ap_viewport_y = ap_viewport_input[1];
        ap_viewport_w = ap_viewport_input[2];
        ap_viewport_h = ap_viewport_input[3];
        max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        if (thread_id < ap_width * ap_height)
        {
            // whether or not pixel is active
            is_active[thread_id] = isActive(thread_id);
        
            // whether or not pixel starts a new run
            if (thread_id == 0)
            {
                new_run[thread_id] = 1;
            }
            else
            {
                uint8_t prev_active = isActive(thread_id - 1);
                new_run[thread_id] = (uint8_t)(is_active[thread_id] != prev_active);
            }
        }
    }
    __device__ uint8_t isActive(int thread_id)
    {
        uint8_t active = 0;
        int px_x = thread_id % ap_width;
        int px_y = thread_id / ap_width;
        
        // pixel inside viewport
        if (px_x >= ap_viewport_x && px_x < (ap_viewport_x + ap_viewport_w) &&
            px_y >= ap_viewport_y && px_y < (ap_viewport_y + ap_viewport_h))
        {
            int px_texture_x = px_x - ap_viewport_x + texture_viewport_x;
            int px_texture_y = px_y - ap_viewport_y + texture_viewport_y;
            
            float px_depth;
            surf2Dread(&px_depth, depth, 4 * px_texture_x, px_texture_y);
            
            active = (uint8_t)(px_depth != max_depth);
        }
        return active;
    }
};

struct PariCGSubActivePixelFinalizeFunctor
{
    cudaSurfaceObject_t rgba;
    cudaSurfaceObject_t depth;
    uint8_t *is_active;
    uint8_t *new_run;
    uint32_t *run_index;
    uint32_t *run_length;
    uint32_t *active_index;
    int texture_width;
    int texture_height;
    int texture_viewport_x;
    int texture_viewport_y;
    int texture_viewport_w;
    int texture_viewport_h;
    int ap_width;
    int ap_height;
    int ap_viewport_x;
    int ap_viewport_y;
    int ap_viewport_w;
    int ap_viewport_h;
    uint8_t *compressed;
    uint32_t *compressed_size;
    float max_depth;
    PariCGSubActivePixelFinalizeFunctor(cudaSurfaceObject_t const& rgba_input, cudaSurfaceObject_t const& depth_input,
                                        thrust::device_vector<uint8_t>& is_active_input, thrust::device_vector<uint8_t>& new_run_input,
                                        thrust::device_vector<uint32_t>& run_idx_input, thrust::device_vector<uint32_t>& run_length_input,
                                        thrust::device_vector<uint32_t>& active_idx_input, thrust::device_vector<uint8_t>& output,
                                        thrust::device_vector<uint32_t>& output_size, int texture_width_input, int texture_height_input,
                                        int *texture_viewport_input, int ap_width_input, int ap_height_input, int *ap_viewport_input)
    {
        rgba = rgba_input;
        depth = depth_input;
        is_active = thrust::raw_pointer_cast(is_active_input.data());
        new_run = thrust::raw_pointer_cast(new_run_input.data());
        run_index = thrust::raw_pointer_cast(run_idx_input.data());
        run_length = thrust::raw_pointer_cast(run_length_input.data());
        active_index = thrust::raw_pointer_cast(active_idx_input.data());
        compressed = thrust::raw_pointer_cast(output.data());
        compressed_size = thrust::raw_pointer_cast(output_size.data());
        texture_width = texture_width_input;
        texture_height = texture_height_input;
        texture_viewport_x = texture_viewport_input[0];
        texture_viewport_y = texture_viewport_input[1];
        texture_viewport_w = texture_viewport_input[2];
        texture_viewport_h = texture_viewport_input[3];
        ap_width = ap_width_input;
        ap_height = ap_height_input;
        ap_viewport_x = ap_viewport_input[0];
        ap_viewport_y = ap_viewport_input[1];
        ap_viewport_w = ap_viewport_input[2];
        ap_viewport_h = ap_viewport_input[3];
        max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        if(thread_id < ap_width * ap_height && is_active[thread_id]) // active pixels only
        {
            uint32_t write_pos = 8 * (active_index[thread_id] + ((run_index[thread_id] - 1) / 2) + 1);
            
            int px_texture_x = (thread_id % ap_width) - ap_viewport_x + texture_viewport_x;
            int px_texture_y = (thread_id / ap_width) - ap_viewport_y + texture_viewport_y;
            
            uchar4 px_color;
            float px_depth;
            surf2Dread(&px_color, rgba, 4 * px_texture_x, px_texture_y);
            surf2Dread(&px_depth, depth, 4 * px_texture_x, px_texture_y);
            memcpy(compressed + write_pos, &px_color, 4);
            memcpy(compressed + write_pos + 4, &px_depth, 4);
            if (new_run[thread_id] == 1)
            {
                uint32_t num_inactive = (run_index[thread_id] > 1) ? run_length[run_index[thread_id] - 2] : 0;
                uint32_t num_active = run_length[run_index[thread_id] - 1];
                memcpy(compressed + write_pos - 8, &num_inactive, 4);
                memcpy(compressed + write_pos - 4, &num_active, 4);
            }
        }
        if (thread_id == (ap_width * ap_height) - 1) // final pixel - write compressed size
        {
            uint32_t active_run = run_index[thread_id] + is_active[thread_id] - 2;
            uint32_t write_pos = 8 * (active_index[thread_id] + (active_run / 2) + 1);
        
            compressed_size[0] = write_pos + 8;
            if (is_active[thread_id] == 0)
            {
                uint32_t num_inactive = run_length[run_index[thread_id] - 1];
                uint32_t num_active = 0;
                memcpy(compressed + write_pos, &num_inactive, 4);
                memcpy(compressed + write_pos + 4, &num_active, 4);
            }
        }
    }
};


// Standard PARI functions
PARI_DLLEXPORT void pariSetGpuDevice(int device)
{
    if (device == PARI_DEVICE_OPENGL)
    {
        unsigned int device_count;
        int devices[8];
        cudaGLGetDevices(&device_count, devices, 8, cudaGLDeviceListAll);
        device = devices[0];
    }
    cudaSetDevice(device);
}

PARI_DLLEXPORT PariGpuBuffer pariAllocateGpuBuffer(uint32_t width, uint32_t height, PariEnum type)
{
    PariGpuBuffer buffers;
    switch (type)
    {
        case PARI_IMAGE_RGBA:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint8_t>(width * height * 4));
            break;
        case PARI_IMAGE_DEPTH32F:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<float>(width * height));
            break;
        case PARI_IMAGE_GRAYSCALE:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint8_t>(width * height));
            break;
        case PARI_IMAGE_RGB:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint8_t>(width * height * 3));
            break;
        case PARI_IMAGE_DXT1:
            if (width % 4 != 0 || height % 4 != 0)
            {
                buffers = NULL;
            }
            else
            {
                buffers = (PariGpuBuffer)malloc(sizeof(void*));
                buffers[0] = (void*)(new thrust::device_vector<uint8_t>(width * height / 2));
            }
            break;
        case PARI_IMAGE_ACTIVE_PIXEL:
            buffers = (PariGpuBuffer)malloc(7 * sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint8_t>(width * height));         // whether or not each pixel starts a new run (0 or 1)
            buffers[1] = (void*)(new thrust::device_vector<uint8_t>(width * height));         // whether or not each pixel is active (0 or 1)
            buffers[2] = (void*)(new thrust::device_vector<uint32_t>(width * height));        // id for each run (inclusive scan of buffers[0])
            buffers[3] = (void*)(new thrust::device_vector<uint32_t>(width * height));        // number of pixels in each run (reduce_by_key of buffers[2])
            buffers[4] = (void*)(new thrust::device_vector<uint32_t>(width * height));        // number of active pixels prior to each pixel (exclusive scan of buffers[1])
            buffers[5] = (void*)(new thrust::device_vector<uint8_t>(width * height * 8 + 8)); // final compressed image
            buffers[6] = (void*)(new thrust::device_vector<uint32_t>(1));                     // size in bytes of final compressed image
            break;
        default:
            buffers = NULL;
            break;
    }
    return buffers;
}

PARI_DLLEXPORT void pariFreeGpuBuffer(PariGpuBuffer buffer, PariEnum type)
{
    switch (type)
    {
        case PARI_IMAGE_RGBA:
            {
                thrust::device_vector<uint8_t> *rgba = (thrust::device_vector<uint8_t>*)buffer[0];
                rgba->clear();
                delete rgba;
            }
            break;
        case PARI_IMAGE_DEPTH32F:
            {
                thrust::device_vector<float> *depth = (thrust::device_vector<float>*)buffer[0];
                depth->clear();
                delete depth;
            }
            break;
        case PARI_IMAGE_GRAYSCALE:
            {
                thrust::device_vector<uint8_t> *gray = (thrust::device_vector<uint8_t>*)buffer[0];
                gray->clear();
                delete gray;
            }
            break;
        case PARI_IMAGE_RGB:
            {
                thrust::device_vector<uint8_t> *rgb = (thrust::device_vector<uint8_t>*)buffer[0];
                rgb->clear();
                delete rgb;
            }
            break;
        case PARI_IMAGE_DXT1:
            {
                thrust::device_vector<uint8_t> *dxt1 = (thrust::device_vector<uint8_t>*)buffer[0];
                dxt1->clear();
                delete dxt1;
            }
            break;
        case PARI_IMAGE_ACTIVE_PIXEL:
            {
                thrust::device_vector<uint8_t> *new_run = (thrust::device_vector<uint8_t>*)buffer[0];
                thrust::device_vector<uint8_t> *is_active = (thrust::device_vector<uint8_t>*)buffer[1];
                thrust::device_vector<uint32_t> *run_id = (thrust::device_vector<uint32_t>*)buffer[2];
                thrust::device_vector<uint32_t> *run_counts = (thrust::device_vector<uint32_t>*)buffer[3];
                thrust::device_vector<uint32_t> *active_idx = (thrust::device_vector<uint32_t>*)buffer[4];
                thrust::device_vector<uint8_t> *ap_image = (thrust::device_vector<uint8_t>*)buffer[5];
                thrust::device_vector<uint32_t> *ap_size = (thrust::device_vector<uint32_t>*)buffer[6];
                new_run->clear();
                is_active->clear();
                run_id->clear();
                run_counts->clear();
                active_idx->clear();
                ap_image->clear();
                ap_size->clear();
                delete new_run;
                delete is_active;
                delete run_id;
                delete run_counts;
                delete active_idx;
                delete ap_image;
                delete ap_size;
            }
            break;
        default:
            break;
    }
}

PARI_DLLEXPORT void pariRgbaBufferToGrayscale(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                              PariGpuBuffer gpu_out_buf, uint8_t *gray)
{
    uint64_t start = currentTime();

    // Get handles to input and output image pointers
    thrust::device_vector<uint8_t> *input_ptr = (thrust::device_vector<uint8_t>*)(gpu_in_buf[0]);
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);

    // Upload RGBA buffer to GPU
    thrust::copy(rgba, rgba + (width * height * 4), input_ptr->begin());

    // Convert RGBA buffer to Grayscale buffer (one thread per pixel)
    thrust::counting_iterator<size_t> it(0);
    thrust::for_each_n(thrust::device, it, width * height, PariGrayscaleFunctor(*input_ptr, *output_ptr));

    // Copy image data back to host
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (width * height), gray);

    uint64_t end = currentTime();
    printf("PARI> pariRgbaBufferToGrayscale (%dx%d): %.6lf\n", width, height, (double)(end - start) / 1000000.0);
}

PARI_DLLEXPORT void pariRgbaBufferToDxt1(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                         PariGpuBuffer gpu_out_buf,uint8_t *dxt1)
{
    uint64_t start = currentTime();

    // Get handles to input and output image pointers
    thrust::device_vector<uint8_t> *input_ptr = (thrust::device_vector<uint8_t>*)(gpu_in_buf[0]);
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);

    // Upload RGBA buffer to GPU
    thrust::copy(rgba, rgba + (width * height * 4), input_ptr->begin());

    // Convert RGBA buffer to DXT1 buffer (one thread per 4x4 tile)
    const int k = 16;                        // pixels per tile
    const int n = (width * height) / k;      // number of tiles
    thrust::counting_iterator<size_t> it(0);
    thrust::for_each_n(thrust::device, it, n, PariDxt1Functor(*input_ptr, *output_ptr, width));

    // Copy image data back to host
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (width * height / 2), dxt1);

    uint64_t end = currentTime();
    printf("PARI> pariRgbaBufferToDxt1 (%dx%d): %.6lf\n", width, height, (double)(end - start) / 1000000.0);
}

PARI_DLLEXPORT void pariRgbaDepthBufferToActivePixel(uint8_t *rgba, float *depth, uint32_t width, uint32_t height,
                                                     PariGpuBuffer gpu_rgba_in_buf, PariGpuBuffer gpu_depth_in_buf,
                                                     PariGpuBuffer gpu_out_buf, uint8_t *active_pixel, uint32_t *active_pixel_size)
{
    uint64_t start = currentTime();

    // Get handles to input and output image pointers
    thrust::device_vector<uint8_t> *input_rgba_ptr = (thrust::device_vector<uint8_t>*)(gpu_rgba_in_buf[0]);
    thrust::device_vector<float> *input_depth_ptr = (thrust::device_vector<float>*)(gpu_depth_in_buf[0]);
    thrust::device_vector<uint8_t> *new_run_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    thrust::device_vector<uint8_t> *is_active_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[1]);
    thrust::device_vector<uint32_t> *run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    thrust::device_vector<uint32_t> *run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[3]);
    thrust::device_vector<uint32_t> *active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[5]);
    thrust::device_vector<uint32_t> *output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[6]);

    // Upload RGBA and Depth buffers to GPU
    thrust::copy(rgba, rgba + (width * height * 4), input_rgba_ptr->begin());
    thrust::copy(depth, depth + (width * height), input_depth_ptr->begin());

    uint64_t start_compute = currentTime();

    // Convert RGBA and Depth buffers to Active Pixel buffer
    thrust::counting_iterator<size_t> it(0);
    typecast<uint8_t, uint32_t> ubyteToUint;
    thrust::plus<uint32_t> uintSum;
    //   - whether or not each pixel starts a new run (0 or 1) and whether or not each pixel is active (0 or 1)
    thrust::for_each_n(thrust::device, it, width * height, PariActivePixelFunctor(width, height, *input_rgba_ptr,
                       *input_depth_ptr, *new_run_ptr, *is_active_ptr));
    
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, new_run_ptr->begin(), new_run_ptr->end(), run_id_ptr->begin(),
                                     ubyteToUint, uintSum);
    
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->end(), thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, is_active_ptr->begin(), is_active_ptr->end(), active_idx_ptr->begin(),
                                     ubyteToUint, 0, uintSum);
    
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, width * height, PariActivePixelFinalizeFunctor(width, height, *input_rgba_ptr,
                       *input_depth_ptr, *is_active_ptr, *new_run_ptr, *run_id_ptr, *run_counts_ptr, *active_idx_ptr,
                       *output_ptr, *output_size_ptr));

    uint64_t end_compute = currentTime();

    // Copy image data back to host
    thrust::copy(output_size_ptr->begin(), output_size_ptr->end(), active_pixel_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*active_pixel_size), active_pixel);

    uint64_t end = currentTime();
    printf("PARI> pariRgbaDepthBufferToActivePixel (%dx%d): %.6lf (%.6lf compute)\n", width, height, (double)(end - start) / 1000000.0, (double)(end_compute - start_compute) / 1000000.0);
}

PARI_DLLEXPORT double pariGetTime(PariEnum time)
{
    double elapsed = 0.0;
    switch (time)
    {
        case PARI_TIME_COMPUTE:
            elapsed = _compute_time;
            break;
        case PARI_TIME_MEMORY_TRANSFER:
            elapsed = _mem_transfer_time;
            break;
        case PARI_TIME_TOTAL:
            elapsed = _total_time;
            break;
    }
    return elapsed;
}

// OpenGL - PARI functions
PARI_DLLEXPORT PariCGResource pariRegisterImage(uint32_t texture, PariCGResourceDescription *resrc_description_ptr)
{
    struct cudaGraphicsResource *cuda_resource;
    struct cudaResourceDesc **description_ptr = (struct cudaResourceDesc **)resrc_description_ptr;
    
    // NOTE: GL_DEPTH_COMPONENT not supported - only the following:
    //  - GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
    //  - {GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
    //  - {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "PARI> PariCGResource: cudaGraphicsGLRegisterImage - %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }

    *description_ptr = new struct cudaResourceDesc();
    memset(*description_ptr, 0, sizeof(struct cudaResourceDesc));
    (*description_ptr)->resType = cudaResourceTypeArray;
    
    return (PariCGResource)cuda_resource;
}

PARI_DLLEXPORT void pariUnregisterImage(PariCGResource resrc, PariCGResourceDescription resrc_description)
{
    struct cudaGraphicsResource *cuda_resource = (struct cudaGraphicsResource *)resrc;
    struct cudaResourceDesc *description = (struct cudaResourceDesc *)resrc_description;
    
    delete description;
    cudaGraphicsUnregisterResource(cuda_resource);
}

PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *gray)
{
    glFinish(); // wait for OpenGL commands to finish and GPU to become available

    uint64_t start = currentTime();
    
    cudaArray *array;
    cudaSurfaceObject_t target;

    // Get handles to output image pointer as well as cuda resource and its description
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    struct cudaGraphicsResource *cuda_resource = (struct cudaGraphicsResource *)cg_resource;
    struct cudaResourceDesc description = *(struct cudaResourceDesc *)resrc_description;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
    description.res.array.array = array;
    cudaCreateSurfaceObject(&target, &description);
    
    // Convert RGBA texture to Grayscale buffer
    uint64_t start_compute = currentTime();
    thrust::counting_iterator<size_t> it(0);
    thrust::for_each_n(thrust::device, it, width * height, PariCGGrayscaleFunctor(target, *output_ptr, width, height));
    cudaDeviceSynchronize();
    uint64_t end_compute = currentTime();

    // Copy image data back to host
    uint64_t start_mem_transfer = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (width * height), gray);
    uint64_t end_mem_transfer = currentTime();

    // Release texture for use by OpenGL again
    cudaDestroySurfaceObject(target);
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);

    uint64_t end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetRgbaTextureAsDxt1(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                             PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *dxt1)
{
    glFinish(); // wait for OpenGL commands to finish and GPU to become available

    uint64_t start = currentTime();
    
    cudaArray *array;
    cudaSurfaceObject_t target;

    // Get handles to output image pointer as well as cuda resource and its description
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    struct cudaGraphicsResource *cuda_resource = (struct cudaGraphicsResource *)cg_resource;
    struct cudaResourceDesc description = *(struct cudaResourceDesc *)resrc_description;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
    description.res.array.array = array;
    cudaCreateSurfaceObject(&target, &description);

    // Convert RGBA texture to DXT1 buffer
    uint64_t start_compute = currentTime();
    const int k = 16;                        // pixels per tile
    const int n = (width * height) / k;      // number of tiles
    thrust::counting_iterator<size_t> it(0);
    thrust::for_each_n(thrust::device, it, n, PariCGDxt1Functor(target, *output_ptr, width, height));
    cudaDeviceSynchronize();
    uint64_t end_compute = currentTime();

    // Copy image data back to host
    uint64_t start_mem_transfer = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (width * height / 2), dxt1);
    uint64_t end_mem_transfer = currentTime();

    // Release texture for use by OpenGL again
    cudaDestroySurfaceObject(target);
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);

    uint64_t end = currentTime();
    
    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                         PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                         PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *active_pixel,
                                                         uint32_t *active_pixel_size)
{
    glFinish(); // wait for OpenGL commands to finish and GPU to become available
    //cudaDeviceSynchronize();

    uint64_t start = currentTime();

    cudaArray *array_color;
    cudaArray *array_depth;
    cudaSurfaceObject_t target_color;
    cudaSurfaceObject_t target_depth;

    // Get handles to output image pointers as well as cuda resources and their descriptions
    thrust::device_vector<uint8_t> *new_run_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    thrust::device_vector<uint8_t> *is_active_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[1]);
    thrust::device_vector<uint32_t> *run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    thrust::device_vector<uint32_t> *run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[3]);
    thrust::device_vector<uint32_t> *active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[5]);
    thrust::device_vector<uint32_t> *output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[6]);
    struct cudaGraphicsResource *cuda_resource_color = (struct cudaGraphicsResource *)cg_resource_color;
    struct cudaGraphicsResource *cuda_resource_depth = (struct cudaGraphicsResource *)cg_resource_depth;
    struct cudaResourceDesc description_color = *(struct cudaResourceDesc *)resrc_description_color;
    struct cudaResourceDesc description_depth = *(struct cudaResourceDesc *)resrc_description_depth;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource_color, 0);
    cudaGraphicsMapResources(1, &cuda_resource_depth, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_color, cuda_resource_color, 0, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_depth, cuda_resource_depth, 0, 0);
    description_color.res.array.array = array_color;
    description_depth.res.array.array = array_depth;
    cudaCreateSurfaceObject(&target_color, &description_color);
    cudaCreateSurfaceObject(&target_depth, &description_depth);

    // Convert RGBA and Depth buffers to Active Pixel buffer
    uint64_t start_compute = currentTime();
    thrust::counting_iterator<size_t> it(0);
    typecast<uint8_t, uint32_t> ubyteToUint;
    thrust::plus<uint32_t> uintSum;
    //   - whether or not each pixel starts a new run (0 or 1) and whether or not each pixel is active (0 or 1)
    thrust::for_each_n(thrust::device, it, width * height, PariCGActivePixelFunctor(target_depth, *new_run_ptr,
                       *is_active_ptr, width, height));
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, new_run_ptr->begin(), new_run_ptr->end(), run_id_ptr->begin(),
                                     ubyteToUint, uintSum);
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->end(), thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, is_active_ptr->begin(), is_active_ptr->end(), active_idx_ptr->begin(),
                                     ubyteToUint, 0, uintSum);
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, width * height, PariCGActivePixelFinalizeFunctor(target_color, target_depth,
                       *is_active_ptr, *new_run_ptr, *run_id_ptr, *run_counts_ptr, *active_idx_ptr, *output_ptr,
                       *output_size_ptr, width, height));
    cudaDeviceSynchronize();
    uint64_t end_compute = currentTime();

    // Copy image data back to host
    uint64_t start_mem_transfer = currentTime();
    thrust::copy(output_size_ptr->begin(), output_size_ptr->end(), active_pixel_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*active_pixel_size), active_pixel);
    uint64_t end_mem_transfer = currentTime();

    uint64_t end = currentTime();
    
    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetSubRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                            PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                            PariGpuBuffer gpu_out_buf, uint32_t texture_width, uint32_t texture_height,
                                                            int32_t *texture_viewport, uint32_t ap_width, uint32_t ap_height,
                                                            int32_t *ap_viewport, uint8_t *active_pixel, uint32_t *active_pixel_size)
{
    glFinish(); // wait for OpenGL commands to finish and GPU to become available
    //cudaDeviceSynchronize();

    uint64_t start = currentTime();

    cudaArray *array_color;
    cudaArray *array_depth;
    cudaSurfaceObject_t target_color;
    cudaSurfaceObject_t target_depth;

    // Get handles to output image pointers as well as cuda resources and their descriptions
    thrust::device_vector<uint8_t> *new_run_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    thrust::device_vector<uint8_t> *is_active_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[1]);
    thrust::device_vector<uint32_t> *run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    thrust::device_vector<uint32_t> *run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[3]);
    thrust::device_vector<uint32_t> *active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);
    thrust::device_vector<uint8_t> *output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[5]);
    thrust::device_vector<uint32_t> *output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[6]);
    struct cudaGraphicsResource *cuda_resource_color = (struct cudaGraphicsResource *)cg_resource_color;
    struct cudaGraphicsResource *cuda_resource_depth = (struct cudaGraphicsResource *)cg_resource_depth;
    struct cudaResourceDesc description_color = *(struct cudaResourceDesc *)resrc_description_color;
    struct cudaResourceDesc description_depth = *(struct cudaResourceDesc *)resrc_description_depth;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource_color, 0);
    cudaGraphicsMapResources(1, &cuda_resource_depth, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_color, cuda_resource_color, 0, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_depth, cuda_resource_depth, 0, 0);
    description_color.res.array.array = array_color;
    description_depth.res.array.array = array_depth;
    cudaCreateSurfaceObject(&target_color, &description_color);
    cudaCreateSurfaceObject(&target_depth, &description_depth);

    // Convert RGBA and Depth buffers to Active Pixel buffer
    uint64_t start_compute = currentTime();
    thrust::counting_iterator<size_t> it(0);
    typecast<uint8_t, uint32_t> ubyteToUint;
    thrust::plus<uint32_t> uintSum;
    uint32_t size = ap_width * ap_height;
    //   - whether or not each pixel starts a new run (0 or 1) and whether or not each pixel is active (0 or 1)
    thrust::for_each_n(thrust::device, it, size, PariCGSubActivePixelFunctor(target_depth, *new_run_ptr,
                       *is_active_ptr, texture_width, texture_height, texture_viewport, ap_width, ap_height, ap_viewport));
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, new_run_ptr->begin(), new_run_ptr->begin() + size, run_id_ptr->begin(),
                                     ubyteToUint, uintSum);
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->begin() + size, thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, is_active_ptr->begin(), is_active_ptr->begin() + size, active_idx_ptr->begin(),
                                     ubyteToUint, 0, uintSum);
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, size, PariCGSubActivePixelFinalizeFunctor(target_color, target_depth,
                       *is_active_ptr, *new_run_ptr, *run_id_ptr, *run_counts_ptr, *active_idx_ptr, *output_ptr,
                       *output_size_ptr, texture_width, texture_height, texture_viewport, ap_width, ap_height, ap_viewport));
    cudaDeviceSynchronize();
    uint64_t end_compute = currentTime();

    // Copy image data back to host
    uint64_t start_mem_transfer = currentTime();
    thrust::copy(output_size_ptr->begin(), output_size_ptr->end(), active_pixel_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*active_pixel_size), active_pixel);
    uint64_t end_mem_transfer = currentTime();

    uint64_t end = currentTime();
    
    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}


// Internal functions
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

__device__ void extractCGTile4x4(uint32_t offset_x, uint32_t offset_y, const cudaSurfaceObject_t pixels, uint8_t out_tile[64])
{
    int i, j;
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 4; i++)
        {
            uchar4 color;
            surf2Dread(&color, pixels, 4 * (offset_x + i), offset_y + j);
            out_tile[j * 16 + 4 * i + 0] = color.x;
            out_tile[j * 16 + 4 * i + 1] = color.y;
            out_tile[j * 16 + 4 * i + 2] = color.z;
            out_tile[j * 16 + 4 * i + 3] = color.w;
        }
    }
}

// --------------------------------------------------------------- //

static uint64_t currentTime()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (ts.tv_sec * 1000000ull) + (ts.tv_nsec / 1000ull);
}
