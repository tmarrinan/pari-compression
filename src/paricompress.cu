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

#if __CUDACC_VER_MAJOR__ < 11
static struct cudaTextureDesc _texture_description = {
    {cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeWrap}, // addressMode
    cudaFilterModePoint,                                               // filterMode
    cudaReadModeElementType,                                           // readMode
    0,                                                                 // sRGB
    {0.0f, 0.0f, 0.0f, 0.0f},                                          // borderColor
    0,                                                                 // normalizedCoords
    0,                                                                 // maxAnisotropy
    cudaFilterModePoint,                                               // mipmapFilterMode
    0.0f,                                                              // mipmapLevelBias
    0.0f,                                                              // minMipmapLevelClamp
    0.0f                                                               // maxMipmapLevelClamp
};
#else
static struct cudaTextureDesc _texture_description = {
    {cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeWrap}, // addressMode
    cudaFilterModePoint,                                               // filterMode
    cudaReadModeElementType,                                           // readMode
    0,                                                                 // sRGB
    {0.0f, 0.0f, 0.0f, 0.0f},                                          // borderColor
    0,                                                                 // normalizedCoords
    0,                                                                 // maxAnisotropy
    cudaFilterModePoint,                                               // mipmapFilterMode
    0.0f,                                                              // mipmapLevelBias
    0.0f,                                                              // minMipmapLevelClamp
    0.0f,                                                              // maxMipmapLevelClamp
    0                                                                  // disableTrilinearOptimization
};
#endif    


__host__ __device__ static void extractTile4x4(uint32_t offset, const uchar4 *pixels, int width, uchar4 out_tile[16]);
__host__ __device__ static void getMinMaxColors(uchar4 tile[16], uchar3 *color_min, uchar3 *color_max);
__host__ __device__ static uint16_t colorTo565(uchar3 &color);
__host__ __device__ static uint32_t colorDistance(uchar4 tile[16], int t_offset, uchar3 colors[4], int c_offset);
__host__ __device__ static uint32_t colorIndices(uchar4 tile[16], uchar3 &color_min, uchar3 &color_max);
__host__ __device__ static void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__host__ __device__ static void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value);
__device__ static void extractCGTile4x4(uint32_t offset_x, uint32_t offset_y, const cudaTextureObject_t pixels, uchar4 out_tile[16]);

static uint64_t currentTime();


// CUDA Custom Functors
struct PariTransformGrayscale : public thrust::unary_function<uchar4, uint8_t>
{
    PariTransformGrayscale()
    {
    }
    __host__ __device__ uint8_t operator()(uchar4 n) const
    {
        return (uint8_t)(0.299f * n.x + 0.587f * n.y + 0.114f * n.z);
    }
};

struct PariForEachNDxt1
{
    const uchar4 *_rgba;
    uint8_t *_dxt1;
    uint32_t _width;
    size_t _size;
    PariForEachNDxt1(thrust::device_vector<uchar4> const& rgba, thrust::device_vector<uint8_t>& dxt1, uint32_t width, uint32_t height)
    {
        _rgba = thrust::raw_pointer_cast(rgba.data());
        _dxt1 = thrust::raw_pointer_cast(dxt1.data());
        _width = width;
        _size = (width * height) / 16;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        if (thread_id < _size)
        {
            uchar4 tile[16];
            uchar3 color_min;
            uchar3 color_max;

            // px_ (x and y pixel indices)
            // tile_ (x and y tile indices)
            uint32_t tile_width = _width / 4;
      	    uint32_t tile_x = thread_id % tile_width;
            uint32_t tile_y = thread_id / tile_width;
            uint32_t px_x = tile_x * 4;
            uint32_t px_y = tile_y * 4;

            uint32_t offset = (px_y * _width) + px_x;
            uint32_t write_pos = (tile_y * tile_width * 8) + (tile_x * 8);

            extractTile4x4(offset, _rgba, _width, tile);
            getMinMaxColors(tile, &color_min, &color_max);
            writeUint16(_dxt1, write_pos, colorTo565(color_max));
       	    writeUint16(_dxt1, write_pos + 2, colorTo565(color_min));
       	    writeUint32(_dxt1, write_pos + 4, colorIndices(tile, color_min, color_max));
        }
    }
};

struct PariTransformActivePixelNewRun : public thrust::unary_function<uint32_t, uint32_t>
{
    const float *_depth;
    float _max_depth;
    PariTransformActivePixelNewRun(thrust::device_vector<float> const& depth)
    {
        _depth = thrust::raw_pointer_cast(depth.data());
        _max_depth = 1.0f;
    }
    __host__ __device__ uint32_t operator()(uint32_t n) const
    {
        uint32_t new_run = 1;
        if (n > 0)
        {
            float px_depth = _depth[n];
            float prev_depth = _depth[n - 1];
            
            uint32_t px_active = (uint32_t)(px_depth != _max_depth);
            uint32_t prev_active = (uint32_t)(prev_depth != _max_depth);
            
            new_run = px_active ^ prev_active; // XOR
        }
        return new_run;
    }
};

struct PariTransformActivePixelIsActive : public thrust::unary_function<float, uint32_t>
{
    float _max_depth;
    PariTransformActivePixelIsActive()
    {
        _max_depth = 1.0f;
    }
    __host__ __device__ uint32_t operator()(float n) const
    {
        return (uint32_t)(n != _max_depth);
    }
};

struct PariForEachNActivePixelWrite
{
    const uchar4 *_rgba;
    const float *_depth;
    const uint32_t *_run_index;
    const uint32_t *_run_counts;
    const uint32_t *_active_index;
    uint8_t *_active_pixel;
    uint32_t *_active_pixel_size;
    size_t _size;
    float _max_depth;
    PariForEachNActivePixelWrite(thrust::device_vector<uchar4> const& rgba, thrust::device_vector<float> const& depth,
                                 thrust::device_vector<uint32_t> const& run_index, thrust::device_vector<uint32_t> const& run_counts,
                                 thrust::device_vector<uint32_t> const& active_index, thrust::device_vector<uint8_t>& active_pixel,
                                 thrust::device_vector<uint32_t>& active_pixel_size, uint32_t width, uint32_t height)
    {
        _rgba = thrust::raw_pointer_cast(rgba.data());
        _depth = thrust::raw_pointer_cast(depth.data());
        _run_index = thrust::raw_pointer_cast(run_index.data());
        _run_counts = thrust::raw_pointer_cast(run_counts.data());
        _active_index = thrust::raw_pointer_cast(active_index.data());
        _active_pixel = thrust::raw_pointer_cast(active_pixel.data());
        _active_pixel_size = thrust::raw_pointer_cast(active_pixel_size.data());
        _size = width * height;
        _max_depth = 1.0f;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        // active pixels only
        if (thread_id < _size - 1 && _active_index[thread_id] < _active_index[thread_id + 1])
        {
            uint32_t run_id = _run_index[thread_id] - 1;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (run_id / 2) + 1);
            
            uchar4 px_color = _rgba[thread_id];
            float px_depth = _depth[thread_id];
            memcpy(_active_pixel + write_pos, &px_color, 4);
            memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            
            // pixel starts a new run
            if (thread_id == 0 || run_id > (_run_index[thread_id - 1] - 1))
            {
                uint32_t num_inactive = (run_id > 0) ? _run_counts[run_id - 1] : 0;
                uint32_t num_active = _run_counts[run_id];
                memcpy(_active_pixel + write_pos - 8, &num_inactive, 4);
                memcpy(_active_pixel + write_pos - 4, &num_active, 4);
            }
        }
        // final pixel - write compressed size
        if (thread_id == _size - 1)
        {
            float px_depth = _depth[thread_id];
            uint32_t px_active = (uint32_t)(px_depth != _max_depth);
            
            uint32_t active_run = _run_index[thread_id] + px_active - 2;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (active_run / 2) + 1);
            
            _active_pixel_size[0] = write_pos + 8;
            
            // inactive - copy final inactive / active counts
            if (px_active == 0)
            {
                uint32_t num_inactive = _run_counts[_run_index[thread_id] - 1];
                uint32_t num_active = 0;
                memcpy(_active_pixel + write_pos, &num_inactive, 4);
                memcpy(_active_pixel + write_pos + 4, &num_active, 4);
            }
            // active - copy final pixel color / depth values
            else
            {
                uchar4 px_color = _rgba[thread_id];
                memcpy(_active_pixel + write_pos, &px_color, 4);
                memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            }
        }
    }
};

// OpenGL / CUDA Interop Custom Functors
struct PariCGTransformGrayscale : public thrust::unary_function<size_t, uint8_t>
{
    cudaTextureObject_t _rgba;
    uint32_t _width;
    PariCGTransformGrayscale(cudaTextureObject_t const& rgba, uint32_t width)
    {
        _rgba = rgba;
        _width = width;
    }
    __device__ uint8_t operator()(size_t n) const
    {
        uchar4 color = tex2D<uchar4>(_rgba, n % _width, n / _width);
        return (uint8_t)(0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
    }
};

struct PariCGForEachNDxt1
{
    cudaTextureObject_t _rgba;
    uint8_t *_dxt1;
    uint32_t _width;
    size_t _size;
    PariCGForEachNDxt1(cudaTextureObject_t const& rgba, thrust::device_vector<uint8_t>& dxt1, uint32_t width, uint32_t height)
    {
        _rgba = rgba;
        _dxt1 = thrust::raw_pointer_cast(dxt1.data());
        _width = width;
        _size = (width * height) / 16;
    }
    __device__ void operator()(int thread_id)
    {
        if (thread_id < _size)
        {
            uchar4 tile[16];
            uchar3 color_min;
            uchar3 color_max;

            // px_ (x and y pixel indices)
            // tile_ (x and y tile indices)
            uint32_t tile_width = _width / 4;
      	    uint32_t tile_x = thread_id % tile_width;
            uint32_t tile_y = thread_id / tile_width;
            uint32_t px_x = tile_x * 4;
            uint32_t px_y = tile_y * 4;

            uint32_t write_pos = (tile_y * tile_width * 8) + (tile_x * 8);

            extractCGTile4x4(px_x, px_y, _rgba, tile);
            getMinMaxColors(tile, &color_min, &color_max);
            writeUint16(_dxt1, write_pos, colorTo565(color_max));
       	    writeUint16(_dxt1, write_pos + 2, colorTo565(color_min));
       	    writeUint32(_dxt1, write_pos + 4, colorIndices(tile, color_min, color_max));
        }
    }
};

struct PariCGTransformActivePixelNewRun : public thrust::unary_function<uint32_t, uint32_t>
{
    cudaTextureObject_t _depth;
    uint32_t _width;
    float _max_depth;
    PariCGTransformActivePixelNewRun(cudaTextureObject_t const& depth, uint32_t width)
    {
        _depth = depth;
        _width = width;
        _max_depth = 1.0f;
    }
    __device__ uint32_t operator()(uint32_t n) const
    {
        uint32_t new_run = 1;
        if (n > 0)
        {
            float px_depth = tex2D<float>(_depth, n % _width, n / _width);
            float prev_depth = tex2D<float>(_depth, (n - 1) % _width, (n - 1) / _width);
            
            uint32_t px_active = (uint32_t)(px_depth != _max_depth);
            uint32_t prev_active = (uint32_t)(prev_depth != _max_depth);
            
            new_run = px_active ^ prev_active; // XOR
        }
        return new_run;
    }
};

struct PariCGTransformActivePixelIsActive : public thrust::unary_function<uint32_t, uint32_t>
{
    cudaTextureObject_t _depth;
    uint32_t _width;
    float _max_depth;
    PariCGTransformActivePixelIsActive(cudaTextureObject_t const& depth, uint32_t width)
    {
        _depth = depth;
        _width = width;
        _max_depth = 1.0f;
    }
    __device__ uint32_t operator()(uint32_t n) const
    {
        float px_depth = tex2D<float>(_depth, n % _width, n / _width);
        return (uint32_t)(px_depth != _max_depth);
    }
};

struct PariCGForEachNActivePixelWrite
{
    cudaTextureObject_t _rgba;
    cudaTextureObject_t _depth;
    const uint32_t *_run_index;
    const uint32_t *_run_counts;
    const uint32_t *_active_index;
    uint8_t *_active_pixel;
    uint32_t *_active_pixel_size;
    uint32_t _width;
    size_t _size;
    float _max_depth;
    PariCGForEachNActivePixelWrite(cudaTextureObject_t const& rgba, cudaTextureObject_t const& depth,
                                 thrust::device_vector<uint32_t> const& run_index, thrust::device_vector<uint32_t> const& run_counts,
                                 thrust::device_vector<uint32_t> const& active_index, thrust::device_vector<uint8_t>& active_pixel,
                                 thrust::device_vector<uint32_t>& active_pixel_size, uint32_t width, uint32_t height)
    {
        _rgba = rgba;
        _depth = depth;
        _run_index = thrust::raw_pointer_cast(run_index.data());
        _run_counts = thrust::raw_pointer_cast(run_counts.data());
        _active_index = thrust::raw_pointer_cast(active_index.data());
        _active_pixel = thrust::raw_pointer_cast(active_pixel.data());
        _active_pixel_size = thrust::raw_pointer_cast(active_pixel_size.data());
        _width = width;
        _size = width * height;
        _max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        // active pixels only
        if (thread_id < _size - 1 && _active_index[thread_id] < _active_index[thread_id + 1])
        {
            uint32_t run_id = _run_index[thread_id] - 1;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (run_id / 2) + 1);
            
            int px_x = thread_id % _width;
            int px_y = thread_id / _width;
            uchar4 px_color = tex2D<uchar4>(_rgba, px_x, px_y);
            float px_depth = tex2D<float>(_depth, px_x, px_y);
            memcpy(_active_pixel + write_pos, &px_color, 4);
            memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            
            // pixel starts a new run
            if (thread_id == 0 || run_id > (_run_index[thread_id - 1] - 1))
            {
                uint32_t num_inactive = (run_id > 0) ? _run_counts[run_id - 1] : 0;
                uint32_t num_active = _run_counts[run_id];
                memcpy(_active_pixel + write_pos - 8, &num_inactive, 4);
                memcpy(_active_pixel + write_pos - 4, &num_active, 4);
            }
        }
        // final pixel - write compressed size
        if (thread_id == _size - 1)
        {
            int px_x = thread_id % _width;
            int px_y = thread_id / _width;
            float px_depth = tex2D<float>(_depth, px_x, px_y);
            uint32_t px_active = (uint32_t)(px_depth != _max_depth);
            
            uint32_t active_run = _run_index[thread_id] + px_active - 2;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (active_run / 2) + 1);
            
            _active_pixel_size[0] = write_pos + 8;
            
            // inactive - copy final inactive / active counts
            if (px_active == 0)
            {
                uint32_t num_inactive = _run_counts[_run_index[thread_id] - 1];
                uint32_t num_active = 0;
                memcpy(_active_pixel + write_pos, &num_inactive, 4);
                memcpy(_active_pixel + write_pos + 4, &num_active, 4);
            }
            // active - copy final pixel color / depth values
            else
            {
                uchar4 px_color = tex2D<uchar4>(_rgba, px_x, px_y);
                memcpy(_active_pixel + write_pos, &px_color, 4);
                memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            }
        }
    }
};

struct PariCGTransformSubActivePixelNewRun : public thrust::unary_function<uint32_t, uint32_t>
{
    cudaTextureObject_t _depth;
    uint32_t _ap_width;
    uint32_t _ap_viewport_x;
    uint32_t _ap_viewport_y;
    uint32_t _ap_viewport_w;
    uint32_t _ap_viewport_h;
    uint32_t _texture_viewport_x;
    uint32_t _texture_viewport_y;
    float _max_depth;
    PariCGTransformSubActivePixelNewRun(cudaTextureObject_t const& depth, uint32_t ap_width, int *ap_viewport,
                                        int *texture_viewport)
    {
        _depth = depth;
        _ap_width = ap_width;
        _ap_viewport_x = ap_viewport[0];
        _ap_viewport_y = ap_viewport[1];
        _ap_viewport_w = ap_viewport[2];
        _ap_viewport_h = ap_viewport[3];
        _texture_viewport_x = texture_viewport[0];
        _texture_viewport_y = texture_viewport[1];
        _max_depth = 1.0f;
    }
    __device__ uint32_t operator()(uint32_t n) const
    {
        uint32_t new_run = 1;
        if (n > 0)
        {
            uint32_t px_active = isActive(n);
            uint32_t prev_active = isActive(n - 1);
            
            new_run = px_active ^ prev_active; // XOR
        }
        return new_run;
    }
    __device__ uint32_t isActive(uint32_t n) const
    {
        uint32_t active = 0;
        int px_x = n % _ap_width;
        int px_y = n / _ap_width + _ap_viewport_y;
        // pixel inside viewport
        if (px_x >= _ap_viewport_x && px_x < (_ap_viewport_x + _ap_viewport_w) &&
            px_y >= _ap_viewport_y && px_y < (_ap_viewport_y + _ap_viewport_h))
        {
            int px_texture_x = px_x - _ap_viewport_x + _texture_viewport_x;
            int px_texture_y = px_y - _ap_viewport_y + _texture_viewport_y;
            
            float px_depth = tex2D<float>(_depth, px_texture_x, px_texture_y);
            active = (uint8_t)(px_depth != _max_depth);
        }
        return active;
    }
};

struct PariCGTransformSubActivePixelIsActive : public thrust::unary_function<uint32_t, uint32_t>
{
    cudaTextureObject_t _depth;
    uint32_t _ap_width;
    uint32_t _ap_viewport_x;
    uint32_t _ap_viewport_y;
    uint32_t _ap_viewport_w;
    uint32_t _ap_viewport_h;
    uint32_t _texture_viewport_x;
    uint32_t _texture_viewport_y;
    float _max_depth;
    PariCGTransformSubActivePixelIsActive(cudaTextureObject_t const& depth, uint32_t ap_width, int *ap_viewport,
                                          int *texture_viewport)
    {
        _depth = depth;
        _ap_width = ap_width;
        _ap_viewport_x = ap_viewport[0];
        _ap_viewport_y = ap_viewport[1];
        _ap_viewport_w = ap_viewport[2];
        _ap_viewport_h = ap_viewport[3];
        _texture_viewport_x = texture_viewport[0];
        _texture_viewport_y = texture_viewport[1];
        _max_depth = 1.0f;
    }
    __device__ uint32_t operator()(uint32_t n) const
    {
        uint32_t active = 0;
        int px_x = n % _ap_width;
        int px_y = n / _ap_width + _ap_viewport_y;
        // pixel inside viewport
        if (px_x >= _ap_viewport_x && px_x < (_ap_viewport_x + _ap_viewport_w) &&
            px_y >= _ap_viewport_y && px_y < (_ap_viewport_y + _ap_viewport_h))
        {
            int px_texture_x = px_x - _ap_viewport_x + _texture_viewport_x;
            int px_texture_y = px_y - _ap_viewport_y + _texture_viewport_y;
            
            float px_depth = tex2D<float>(_depth, px_texture_x, px_texture_y);
            active = (uint32_t)(px_depth != _max_depth);
        }
        return active;
    }
};

struct PariCGForEachNSubActivePixelWrite
{
    cudaTextureObject_t _rgba;
    cudaTextureObject_t _depth;
    const uint32_t *_run_index;
    const uint32_t *_run_counts;
    const uint32_t *_active_index;
    uint8_t *_active_pixel;
    uint32_t *_active_pixel_size;
    uint32_t _ap_width;
    uint32_t _ap_viewport_x;
    uint32_t _ap_viewport_y;
    uint32_t _texture_viewport_x;
    uint32_t _texture_viewport_y;
    size_t _size;
    uint32_t _padding_end;
    float _max_depth;
    PariCGForEachNSubActivePixelWrite(cudaTextureObject_t const& rgba, cudaTextureObject_t const& depth,
                                 thrust::device_vector<uint32_t> const& run_index, thrust::device_vector<uint32_t> const& run_counts,
                                 thrust::device_vector<uint32_t> const& active_index, thrust::device_vector<uint8_t>& active_pixel,
                                 thrust::device_vector<uint32_t>& active_pixel_size, uint32_t ap_width, uint32_t ap_height,
                                 int *ap_viewport, int *texture_viewport)
    {
        _rgba = rgba;
        _depth = depth;
        _run_index = thrust::raw_pointer_cast(run_index.data());
        _run_counts = thrust::raw_pointer_cast(run_counts.data());
        _active_index = thrust::raw_pointer_cast(active_index.data());
        _active_pixel = thrust::raw_pointer_cast(active_pixel.data());
        _active_pixel_size = thrust::raw_pointer_cast(active_pixel_size.data());
        _ap_width = ap_width;
        _ap_viewport_x = ap_viewport[0];
        _ap_viewport_y = ap_viewport[1];
        _texture_viewport_x = texture_viewport[0];
        _texture_viewport_y = texture_viewport[1];
        _size = ap_width * ap_viewport[3];
        _padding_end = (ap_height - (ap_viewport[1] + ap_viewport[3])) * ap_width;
        _max_depth = 1.0f;
    }
    __device__ void operator()(int thread_id)
    {
        // active pixels only
        if (thread_id < _size - 1 && _active_index[thread_id] < _active_index[thread_id + 1])
        {
            uint32_t run_id = _run_index[thread_id] - 1;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (run_id / 2) + 1);
            
            int px_x = thread_id % _ap_width;
            int px_y = thread_id / _ap_width + _ap_viewport_y;
            int px_texture_x = px_x - _ap_viewport_x + _texture_viewport_x;
            int px_texture_y = px_y - _ap_viewport_y + _texture_viewport_y;
            uchar4 px_color = tex2D<uchar4>(_rgba, px_texture_x, px_texture_y);
            float px_depth = tex2D<float>(_depth, px_texture_x, px_texture_y);
            memcpy(_active_pixel + write_pos, &px_color, 4);
            memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            
            // pixel starts a new run
            if (thread_id == 0 || run_id > (_run_index[thread_id - 1] - 1))
            {
                uint32_t num_inactive = (run_id > 0) ? _run_counts[run_id - 1] : 0;
                uint32_t num_active = _run_counts[run_id];
                memcpy(_active_pixel + write_pos - 8, &num_inactive, 4);
                memcpy(_active_pixel + write_pos - 4, &num_active, 4);
            }
        }
        // final pixel - write compressed size
        if (thread_id == _size - 1)
        {
            int px_x = thread_id % _ap_width;
            int px_y = thread_id / _ap_width + _ap_viewport_y;
            int px_texture_x = px_x - _ap_viewport_x + _texture_viewport_x;
            int px_texture_y = px_y - _ap_viewport_y + _texture_viewport_y;
            float px_depth = tex2D<float>(_depth, px_texture_x, px_texture_y);
            uint32_t px_active = (uint32_t)(px_depth != _max_depth);
            
            uint32_t active_run = _run_index[thread_id] + px_active - 2;
            uint32_t write_pos = 8 * (_active_index[thread_id] + (active_run / 2) + 1);
            
            // inactive - copy final inactive / active counts
            if (px_active == 0)
            {
                _active_pixel_size[0] = write_pos + 8;
                
                uint32_t num_inactive = _run_counts[_run_index[thread_id] - 1] + _padding_end;
                uint32_t num_active = 0;
                memcpy(_active_pixel + write_pos, &num_inactive, 4);
                memcpy(_active_pixel + write_pos + 4, &num_active, 4);
            }
            // active - copy final pixel color / depth values
            else if (_padding_end == 0)
            {
                _active_pixel_size[0] = write_pos + 8;
                
                uchar4 px_color = tex2D<uchar4>(_rgba, px_texture_x, px_texture_y);
                memcpy(_active_pixel + write_pos, &px_color, 4);
                memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
            }
            // active w/ inactive padding at end - copy final pixel color / depth values and padding
            else
            {
                _active_pixel_size[0] = write_pos + 16;
                
                uchar4 px_color = tex2D<uchar4>(_rgba, px_texture_x, px_texture_y);
                memcpy(_active_pixel + write_pos, &px_color, 4);
                memcpy(_active_pixel + write_pos + 4, &px_depth, 4);
                
                uint32_t num_inactive = _padding_end;
                uint32_t num_active = 0;
                memcpy(_active_pixel + write_pos + 8, &num_inactive, 4);
                memcpy(_active_pixel + write_pos + 12, &num_active, 4);
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

PARI_DLLEXPORT void pariFreeCpuBuffer(void *buffer)
{
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, buffer) == cudaErrorInvalidValue)
    {
        cudaGetLastError(); // clear error - handling here
        free(buffer);
    }
    else
    {
        cudaFreeHost(buffer);
    }
}

PARI_DLLEXPORT void pariAllocateCpuBuffer(void **buffer, uint32_t size)
{
    // Attempt to allocate pinned memory, but fall back to regular if it fails
    cudaError_t err = cudaMallocHost(buffer, size);
    if (err != cudaSuccess)
    {
        *buffer = malloc(size);
    }
}

PARI_DLLEXPORT PariGpuBuffer pariAllocateGpuBuffer(uint32_t width, uint32_t height, PariEnum type)
{
    PariGpuBuffer buffers;
    uint32_t size = width * height;
    switch (type)
    {
        case PARI_IMAGE_RGBA:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uchar4>(size));
            break;
        case PARI_IMAGE_DEPTH32F:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<float>(size));
            break;
        case PARI_IMAGE_GRAYSCALE:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint8_t>(size));
            break;
        case PARI_IMAGE_RGB:
            buffers = (PariGpuBuffer)malloc(sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uchar3>(size));
            break;
        case PARI_IMAGE_DXT1:
            if (width % 4 != 0 || height % 4 != 0)
            {
                buffers = NULL;
            }
            else
            {
                buffers = (PariGpuBuffer)malloc(sizeof(void*));
                buffers[0] = (void*)(new thrust::device_vector<uint8_t>(size / 2));
            }
            break;
        case PARI_IMAGE_ACTIVE_PIXEL:
            buffers = (PariGpuBuffer)malloc(8 * sizeof(void*));
            buffers[0] = (void*)(new thrust::device_vector<uint32_t>(size));        // id for each run
            buffers[1] = (void*)(new thrust::device_vector<uint32_t>(size));        // number of pixels in each run
            buffers[2] = (void*)(new thrust::device_vector<uint32_t>(size));        // number of active pixels prior to each pixel
            buffers[3] = (void*)(new thrust::device_vector<uint8_t>(8 * size + 8)); // final compressed image
            buffers[4] = (void*)(new thrust::device_vector<uint32_t>(1));           // size in bytes of final compressed image
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
                thrust::device_vector<uchar4> *rgba = (thrust::device_vector<uchar4>*)buffer[0];
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
                thrust::device_vector<uchar3> *rgb = (thrust::device_vector<uchar3>*)buffer[0];
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
                thrust::device_vector<uint32_t> *run_id = (thrust::device_vector<uint32_t>*)buffer[0];
                thrust::device_vector<uint32_t> *run_counts = (thrust::device_vector<uint32_t>*)buffer[1];
                thrust::device_vector<uint32_t> *active_idx = (thrust::device_vector<uint32_t>*)buffer[2];
                thrust::device_vector<uint8_t> *ap_image = (thrust::device_vector<uint8_t>*)buffer[3];
                thrust::device_vector<uint32_t> *ap_size = (thrust::device_vector<uint32_t>*)buffer[4];
                run_id->clear();
                run_counts->clear();
                active_idx->clear();
                ap_image->clear();
                ap_size->clear();
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
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
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


// Compress CPU buffers
PARI_DLLEXPORT void pariRgbaBufferToGrayscale(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                              PariGpuBuffer gpu_out_buf, uint8_t *gray)
{
    uint64_t start, start_compute, start_mem_transfer1, start_mem_transfer2, end, end_compute, end_mem_transfer1, end_mem_transfer2;
    uint32_t size;
    thrust::device_vector<uchar4> *input_ptr;
    thrust::device_vector<uint8_t> *output_ptr;

    start = currentTime();
    size = width * height;

    // Get handles to input and output image pointers
    input_ptr = (thrust::device_vector<uchar4>*)(gpu_in_buf[0]);
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);

    // Upload RGBA buffer to GPU
    start_mem_transfer1 = currentTime();
    thrust::copy((uchar4*)rgba, (uchar4*)rgba + size, input_ptr->begin());
    end_mem_transfer1 = currentTime();

    // Convert RGBA buffer to Grayscale buffer (one thread per pixel)
    start_compute = currentTime();
    thrust::transform(thrust::device, input_ptr->begin(), input_ptr->begin() + size, output_ptr->begin(), PariTransformGrayscale());
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer2 = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + size, gray);
    end_mem_transfer2 = currentTime();

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer2 - start_mem_transfer2 + end_mem_transfer1 - start_mem_transfer1) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariRgbaBufferToDxt1(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                         PariGpuBuffer gpu_out_buf,uint8_t *dxt1)
{
    uint64_t start, start_compute, start_mem_transfer1, start_mem_transfer2, end, end_compute, end_mem_transfer1, end_mem_transfer2;
    uint32_t size, k, n;
    thrust::device_vector<uchar4> *input_ptr;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::counting_iterator<size_t> it(0);

    start = currentTime();
    size = width * height;

    // Get handles to input and output image pointers
    input_ptr = (thrust::device_vector<uchar4>*)(gpu_in_buf[0]);
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);

    // Upload RGBA buffer to GPU
    start_mem_transfer1 = currentTime();
    thrust::copy((uchar4*)rgba, (uchar4*)rgba + size, input_ptr->begin());
    end_mem_transfer1 = currentTime();

    // Convert RGBA buffer to DXT1 buffer (one thread per 4x4 tile)
    start_compute = currentTime();
    k = 16;          // pixels per tile
    n = size / k;    // number of tiles
    thrust::for_each_n(thrust::device, it, n, PariForEachNDxt1(*input_ptr, *output_ptr, width, height));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer2 = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (size / 2), dxt1);
    end_mem_transfer2 = currentTime();

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer2 - start_mem_transfer2 + end_mem_transfer1 - start_mem_transfer1) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariRgbaDepthBufferToActivePixel(uint8_t *rgba, float *depth, uint32_t width, uint32_t height,
                                                     PariGpuBuffer gpu_rgba_in_buf, PariGpuBuffer gpu_depth_in_buf,
                                                     PariGpuBuffer gpu_out_buf, uint8_t *active_pixel, uint32_t *out_size)
{
    uint64_t start, start_compute, start_mem_transfer1, start_mem_transfer2, end, end_compute, end_mem_transfer1, end_mem_transfer2;
    uint32_t size;
    thrust::device_vector<uchar4> *input_rgba_ptr;
    thrust::device_vector<float> *input_depth_ptr;
    thrust::device_vector<uint32_t> *run_id_ptr;
    thrust::device_vector<uint32_t> *run_counts_ptr;
    thrust::device_vector<uint32_t> *active_idx_ptr;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::device_vector<uint32_t> *output_size_ptr;
    thrust::counting_iterator<size_t> it(0);
    thrust::plus<uint32_t> uint_sum;

    start = currentTime();
    size = width * height;

    // Get handles to input and output image pointers
    input_rgba_ptr = (thrust::device_vector<uchar4>*)(gpu_rgba_in_buf[0]);
    input_depth_ptr = (thrust::device_vector<float>*)(gpu_depth_in_buf[0]);
    run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[0]);
    run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[1]);
    active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[3]);
    output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);

    // Upload RGBA and Depth buffers to GPU
    start_mem_transfer1 = currentTime();
    thrust::copy((uchar4*)rgba, (uchar4*)rgba + size, input_rgba_ptr->begin());
    thrust::copy(depth, depth + size, input_depth_ptr->begin());
    end_mem_transfer1 = currentTime();

    // Convert RGBA and Depth buffers to Active Pixel buffer
    start_compute = currentTime();
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, it, it + size, run_id_ptr->begin(),
                                     PariTransformActivePixelNewRun(*input_depth_ptr), uint_sum);
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->begin() + size, thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, input_depth_ptr->begin(), input_depth_ptr->begin() + size, active_idx_ptr->begin(),
                                     PariTransformActivePixelIsActive(), 0, uint_sum);
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, size, PariForEachNActivePixelWrite(*input_rgba_ptr, *input_depth_ptr, *run_id_ptr, *run_counts_ptr,
                       *active_idx_ptr, *output_ptr, *output_size_ptr, width, height));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer2 = currentTime();
    thrust::copy(output_size_ptr->begin(), output_size_ptr->begin() + 1, out_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*out_size), active_pixel);
    end_mem_transfer2 = currentTime();

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer2 - start_mem_transfer2 + end_mem_transfer1 - start_mem_transfer1) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}



// Compress GPU buffers
PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *gray)
{
    uint64_t start, start_compute, start_mem_transfer, end, end_compute, end_mem_transfer;
    uint32_t size;
    cudaArray *array;
    cudaTextureObject_t target;
    struct cudaGraphicsResource *cuda_resource;
    struct cudaResourceDesc description;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::counting_iterator<size_t> it(0);

    // Wait for OpenGL commands to finish and GPU to become available
    glFinish();

    start = currentTime();
    size = width * height;

    // Get handles to output image pointer as well as cuda resource and its description
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    cuda_resource = (struct cudaGraphicsResource *)cg_resource;
    description = *(struct cudaResourceDesc *)resrc_description;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
    description.res.array.array = array;
    cudaCreateTextureObject(&target, &description, &_texture_description, NULL);

    // Convert RGBA texture to Grayscale buffer
    start_compute = currentTime();
    thrust::transform(thrust::device, it, it + size, output_ptr->begin(), PariCGTransformGrayscale(target, width));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + size, gray);
    end_mem_transfer = currentTime();

    // Release texture for use by OpenGL again
    cudaDestroyTextureObject(target);
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetRgbaTextureAsDxt1(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                             PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *dxt1)
{
    uint64_t start, start_compute, start_mem_transfer, end, end_compute, end_mem_transfer;
    uint32_t size, k, n;
    cudaArray *array;
    cudaTextureObject_t target;
    struct cudaGraphicsResource *cuda_resource;
    struct cudaResourceDesc description;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::counting_iterator<size_t> it(0);

    // Wait for OpenGL commands to finish and GPU to become available
    glFinish();

    start = currentTime();
    size = width * height;

    // Get handles to output image pointer as well as cuda resource and its description
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[0]);
    cuda_resource = (struct cudaGraphicsResource *)cg_resource;
    description = *(struct cudaResourceDesc *)resrc_description;

    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
    description.res.array.array = array;
    cudaCreateTextureObject(&target, &description, &_texture_description, NULL);

    // Convert RGBA texture to DXT1 buffer
    start_compute = currentTime();
    k = 16;          // pixels per tile
    n = size / k;    // number of tiles
    thrust::for_each_n(thrust::device, it, n, PariCGForEachNDxt1(target, *output_ptr, width, height));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer = currentTime();
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (size / 2), dxt1);
    end_mem_transfer = currentTime();

    // Release texture for use by OpenGL again
    cudaDestroyTextureObject(target);
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                         PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                         PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *active_pixel,
                                                         uint32_t *out_size)
{
    uint64_t start, start_compute, start_mem_transfer, end, end_compute, end_mem_transfer;
    uint32_t size;
    cudaArray *array_color, *array_depth;
    cudaTextureObject_t target_color, target_depth;
    struct cudaGraphicsResource *cuda_resource_color, *cuda_resource_depth;
    struct cudaResourceDesc description_color, description_depth;
    thrust::device_vector<uint32_t> *run_id_ptr;
    thrust::device_vector<uint32_t> *run_counts_ptr;
    thrust::device_vector<uint32_t> *active_idx_ptr;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::device_vector<uint32_t> *output_size_ptr;
    thrust::counting_iterator<size_t> it(0);
    thrust::plus<uint32_t> uint_sum;
    
    // Wait for OpenGL commands to finish and GPU to become available
    glFinish();

    start = currentTime();
    size = width * height;

    // Get handles to input and output image pointers
    run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[0]);
    run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[1]);
    active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[3]);
    output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);
    cuda_resource_color = (struct cudaGraphicsResource *)cg_resource_color;
    cuda_resource_depth = (struct cudaGraphicsResource *)cg_resource_depth;
    description_color = *(struct cudaResourceDesc *)resrc_description_color;
    description_depth = *(struct cudaResourceDesc *)resrc_description_depth;
    
    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource_color, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_color, cuda_resource_color, 0, 0);
    description_color.res.array.array = array_color;
    cudaCreateTextureObject(&target_color, &description_color, &_texture_description, NULL);
    cudaGraphicsMapResources(1, &cuda_resource_depth, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_depth, cuda_resource_depth, 0, 0);
    description_depth.res.array.array = array_depth;
    cudaCreateTextureObject(&target_depth, &description_depth, &_texture_description, NULL);

    // Convert RGBA and Depth buffers to Active Pixel buffer
    start_compute = currentTime();
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, it, it + size, run_id_ptr->begin(),
                                     PariCGTransformActivePixelNewRun(target_depth, width), uint_sum);
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->begin() + size, thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, it, it + size, active_idx_ptr->begin(),
                                     PariCGTransformActivePixelIsActive(target_depth, width), 0, uint_sum);
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, size, PariCGForEachNActivePixelWrite(target_color, target_depth, *run_id_ptr, *run_counts_ptr,
                       *active_idx_ptr, *output_ptr, *output_size_ptr, width, height));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer = currentTime();
    thrust::copy(output_size_ptr->begin(), output_size_ptr->begin() + 1, out_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*out_size), active_pixel);
    end_mem_transfer = currentTime();
    
    // Release textures for use by OpenGL again
    cudaDestroyTextureObject(target_color);
    cudaDestroyTextureObject(target_depth);
    cudaGraphicsUnmapResources(1, &cuda_resource_color, 0);
    cudaGraphicsUnmapResources(1, &cuda_resource_depth, 0);

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

PARI_DLLEXPORT void pariGetSubRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                            PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                            PariGpuBuffer gpu_out_buf, uint32_t ap_width, uint32_t ap_height, int32_t *ap_viewport,
                                                            int32_t *texture_viewport, uint8_t *active_pixel, uint32_t *out_size)
{
    uint64_t start, start_compute, start_mem_transfer, end, end_compute, end_mem_transfer;
    uint32_t size, padding_start;
    uint32_t *first_run;
    cudaArray *array_color, *array_depth;
    cudaTextureObject_t target_color, target_depth;
    struct cudaGraphicsResource *cuda_resource_color, *cuda_resource_depth;
    struct cudaResourceDesc description_color, description_depth;
    thrust::device_vector<uint32_t> *run_id_ptr;
    thrust::device_vector<uint32_t> *run_counts_ptr;
    thrust::device_vector<uint32_t> *active_idx_ptr;
    thrust::device_vector<uint8_t> *output_ptr;
    thrust::device_vector<uint32_t> *output_size_ptr;
    thrust::counting_iterator<size_t> it(0);
    thrust::plus<uint32_t> uint_sum;
    
    // Wait for OpenGL commands to finish and GPU to become available
    glFinish();
    
    start = currentTime();
    padding_start = ap_viewport[1] * ap_width;
    size = ap_width * ap_viewport[3];
    
    // Get handles to input and output image pointers
    run_id_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[0]);
    run_counts_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[1]);
    active_idx_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[2]);
    output_ptr = (thrust::device_vector<uint8_t>*)(gpu_out_buf[3]);
    output_size_ptr = (thrust::device_vector<uint32_t>*)(gpu_out_buf[4]);
    cuda_resource_color = (struct cudaGraphicsResource *)cg_resource_color;
    cuda_resource_depth = (struct cudaGraphicsResource *)cg_resource_depth;
    description_color = *(struct cudaResourceDesc *)resrc_description_color;
    description_depth = *(struct cudaResourceDesc *)resrc_description_depth;
    
    // Enable CUDA to access OpenGL texture
    cudaGraphicsMapResources(1, &cuda_resource_color, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_color, cuda_resource_color, 0, 0);
    description_color.res.array.array = array_color;
    cudaCreateTextureObject(&target_color, &description_color, &_texture_description, NULL);
    cudaGraphicsMapResources(1, &cuda_resource_depth, 0);
    cudaGraphicsSubResourceGetMappedArray(&array_depth, cuda_resource_depth, 0, 0);
    description_depth.res.array.array = array_depth;
    cudaCreateTextureObject(&target_depth, &description_depth, &_texture_description, NULL);
    
    // Convert RGBA and Depth buffers to Active Pixel buffer
    start_compute = currentTime();
    //   - id for each run
    thrust::transform_inclusive_scan(thrust::device, it, it + size, run_id_ptr->begin(),
                                     PariCGTransformSubActivePixelNewRun(target_depth, ap_width, ap_viewport, texture_viewport), uint_sum);
    //   - number of pixels in each run
    thrust::reduce_by_key(thrust::device, run_id_ptr->begin(), run_id_ptr->begin() + size, thrust::make_constant_iterator(1),
                          thrust::discard_iterator<uint32_t>(), run_counts_ptr->begin());
    //   - number of active pixels prior to each pixel
    thrust::transform_exclusive_scan(thrust::device, it, it + size, active_idx_ptr->begin(),
                                     PariCGTransformSubActivePixelIsActive(target_depth, ap_width, ap_viewport, texture_viewport), 0, uint_sum);
    //   -  finalize compressed active pixel image
    thrust::for_each_n(thrust::device, it, size, PariCGForEachNSubActivePixelWrite(target_color, target_depth, *run_id_ptr, *run_counts_ptr,
                       *active_idx_ptr, *output_ptr, *output_size_ptr, ap_width, ap_height, ap_viewport, texture_viewport));
    cudaDeviceSynchronize();
    end_compute = currentTime();

    // Copy image data back to host
    start_mem_transfer = currentTime();
    thrust::copy(output_size_ptr->begin(), output_size_ptr->begin() + 1, out_size);
    thrust::copy(output_ptr->begin(), output_ptr->begin() + (*out_size), active_pixel);
    end_mem_transfer = currentTime();
    
    // Add padding to first inactive block
    first_run = (uint32_t*)active_pixel;
    *first_run = *first_run + padding_start;
    
    // Release textures for use by OpenGL again
    cudaDestroyTextureObject(target_color);
    cudaDestroyTextureObject(target_depth);
    cudaGraphicsUnmapResources(1, &cuda_resource_color, 0);
    cudaGraphicsUnmapResources(1, &cuda_resource_depth, 0);

    end = currentTime();

    _compute_time = (double)(end_compute - start_compute) / 1000000.0;
    _mem_transfer_time = (double)(end_mem_transfer - start_mem_transfer) / 1000000.0;
    _total_time = (double)(end - start) / 1000000.0;
}

// --------------------------------------------------------------- //

void extractTile4x4(uint32_t offset, const uchar4 *pixels, int width, uchar4 out_tile[16])
{
    int i, j;
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 4; i++)
        {
            memcpy(out_tile + (j * 4 + i), pixels + (offset + i), sizeof(uchar4));
        }
        offset += width;
    }
}

void getMinMaxColors(uchar4 tile[16], uchar3 *color_min, uchar3 *color_max)
{
    uchar3 inset;
    int i;
    memset(color_min, 255, sizeof(uchar3));
    memset(color_max, 0, sizeof(uchar3));

    for (i = 0; i < 16; i++)
    {
        color_min->x = min(color_min->x, tile[i].x);
        color_min->y = min(color_min->y, tile[i].y);
        color_min->z = min(color_min->z, tile[i].z);
        color_max->x = max(color_max->x, tile[i].x);
        color_max->y = max(color_max->y, tile[i].y);
        color_max->z = max(color_max->z, tile[i].z);
    }

    inset.x = (color_max->x - color_min->x) >> 4;
    inset.y = (color_max->y - color_min->y) >> 4;
    inset.z = (color_max->z - color_min->z) >> 4;

    color_min->x = min(color_min->x + inset.x, 255);
    color_min->y = min(color_min->y + inset.y, 255);
    color_min->z = min(color_min->z + inset.z, 255);
    color_max->x = max(color_max->x - inset.x, 0);
    color_max->y = max(color_max->y - inset.y, 0);
    color_max->z = max(color_max->z - inset.z, 0);
}

uint16_t colorTo565(uchar3 &color)
{
    uint16_t red = color.x;
    uint16_t green = color.y;
    uint16_t blue = color.z;
    return ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3);
}

uint32_t colorDistance(uchar4 tile[16], int t_offset, uchar3 colors[4], int c_offset)
{
    int dx = tile[t_offset].x - colors[c_offset].x;
    int dy = tile[t_offset].y - colors[c_offset].y;
    int dz = tile[t_offset].z - colors[c_offset].z;

    return (dx*dx) + (dy*dy) + (dz*dz);
}

uint32_t colorIndices(uchar4 tile[16], uchar3 &color_min, uchar3 &color_max)
{
    uchar3 colors[4];
    uint8_t indices[16];
    uint32_t dist, min_dist, result;
    int i, j;
    uint8_t C565_5_MASK = 0xF8;   // 0xFF minus last three bits
    uint8_t C565_6_MASK = 0xFC;   // 0xFF minus last two bits

    colors[0].x = (color_max.x & C565_5_MASK) | (color_max.x >> 5);
    colors[0].y = (color_max.y & C565_6_MASK) | (color_max.y >> 6);
    colors[0].z = (color_max.z & C565_5_MASK) | (color_max.z >> 5);
    colors[1].x = (color_min.x & C565_5_MASK) | (color_min.x >> 5);
    colors[1].y = (color_min.y & C565_6_MASK) | (color_min.y >> 6);
    colors[1].z = (color_min.z & C565_5_MASK) | (color_min.z >> 5);
    colors[2].x = (2 * colors[0].x + colors[1].x) / 3;
    colors[2].y = (2 * colors[0].y + colors[1].y) / 3;
    colors[2].z = (2 * colors[0].z + colors[1].z) / 3;
    colors[3].x = (colors[0].x + 2 * colors[1].x) / 3;
    colors[3].y = (colors[0].y + 2 * colors[1].y) / 3;
    colors[3].z = (colors[0].z + 2 * colors[1].z) / 3;

    for (i = 0; i < 16; i++)
    {
        min_dist = 195076;  // 255 * 255 * 3 + 1
        for (j = 0; j < 4; j++)
        {
            dist = colorDistance(tile, i, colors, j);
            if (dist < min_dist)
            {
                min_dist = dist;
                indices[i] = j;
            }
        }
    }
    
    result = 0;
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

__device__ void extractCGTile4x4(uint32_t offset_x, uint32_t offset_y, const cudaTextureObject_t pixels, uchar4 out_tile[16])
{
    int i, j;
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 4; i++)
        {
            uchar4 color = tex2D<uchar4>(pixels, offset_x + i, offset_y + j);
            memcpy(out_tile + (j * 4 + i), &color, sizeof(uchar4));
        }
    }
}


static uint64_t currentTime()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (ts.tv_sec * 1000000ull) + (ts.tv_nsec / 1000ull);
}
