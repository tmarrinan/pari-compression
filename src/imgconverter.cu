#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "imgconverter.h"

static int img_w;
static int img_h;
static thrust::device_vector<uint8_t> *image_input_ptr;
static thrust::device_vector<uint8_t> *image_output_ptr;
static thrust::device_vector<uint16_t> *runs;
static thrust::device_vector<uint16_t> *num_runs;
static thrust::device_vector<uint32_t> *sums;


__host__ __device__ void extractTile4x4(const uint8_t *pixels, int x, int y, int img_width, int img_height, uint8_t out_tile[64]);
__host__ __device__ void extractTile16x16(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[1024]);
__host__ __device__ void getMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ uint16_t colorTo565(uint8_t color[3]);
__host__ __device__ uint32_t colorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
__host__ __device__ uint32_t colorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__host__ __device__ uint32_t colorFromRgba(const uint8_t *rgba, int width, int height, int x, int y);
__host__ __device__ void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__host__ __device__ void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value);

uint64_t currentTime();


struct GrayscaleFunctor
{
    const uint8_t *rgba;
    uint8_t *gray;
    size_t size;       
    GrayscaleFunctor(thrust::device_vector<uint8_t> const& rgba_in, thrust::device_vector<uint8_t>& gray_out)
    {
        rgba = thrust::raw_pointer_cast(rgba_in.data());
        gray = thrust::raw_pointer_cast(gray_out.data());
        size = rgba_in.size() / 4;
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
    int img_width;
    int img_height;
    int dxt1_width;
    int dxt1_height;
    size_t size;
    Dxt1Functor(int width_in, int height_in, thrust::device_vector<uint8_t> const& rgba_in, thrust::device_vector<uint8_t>& dxt1_out)
    {
        rgba = thrust::raw_pointer_cast(rgba_in.data());
        dxt1 = thrust::raw_pointer_cast(dxt1_out.data());
        img_width = width_in;
        img_height = height_in;
        dxt1_width = (img_width + 3) & ~0x03; // round up to nearest multiple of 4
        dxt1_height = (img_height + 3) & ~0x03; // round up to nearest multiple of 4
        size = dxt1_width * dxt1_height / 16;
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

      	    int tile_x = thread_id % (dxt1_width / 4);
            int tile_y = thread_id / (dxt1_width / 4);
            int px_x = tile_x * 4;
            int px_y = tile_y * 4;

            uint32_t write_pos = (tile_y * (dxt1_width / 4) * 8) + (tile_x * 8);

            extractTile4x4(rgba, px_x, px_y, img_width, img_height, tile);
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
    uint16_t *runs; // boolean array, one index per pixel (1 indicates start of a new run)
    int img_width;
    int img_height;
    int trle_width;
    int trle_height;
    size_t size;
    TrleFunctor(int width_in, int height_in, thrust::device_vector<uint8_t> const& rgba_in, thrust::device_vector<uint16_t>& runs_out)
    {
        rgba = thrust::raw_pointer_cast(rgba_in.data());
        runs = thrust::raw_pointer_cast(runs_out.data());
        img_width = width_in;
        img_height = height_in;
        trle_width = (img_width + 15) & ~0x0f; // round up to nearest multiple of 16
        trle_height = (img_height + 15) & ~0x0f; // round up to nearest multiple of 16
        size = trle_width * trle_height;
    }
    __host__ __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        // inner_ (x and y indices within its tile)
        if (thread_id < size)
        {
            int px_x = thread_id % trle_width;
            int px_y = thread_id / trle_width;

            int tile_x = px_x / 16;
            int tile_y = px_y / 16;
            int tile_idx = tile_y * (trle_width / 16) + tile_x;

            int inner_x = px_x % 16;
            int inner_y = px_y % 16;

            uint32_t color;
            uint32_t color_prev;
            uint32_t prev_x;
            uint32_t prev_y;
			
            // first pixel in tile always starts new run
            if(inner_x == 0 && inner_y == 0) 
            {
                runs[(tile_idx * 256)] = 1;
            }
            else
            {
                if (inner_x == 0) // on new row; go to last pixel in tile on previous row
                {
                    prev_x = px_x + 15;
                    prev_y = px_y - 1;
                }
                else // go left one pixel
                {
                    prev_x = px_x - 1;
                    prev_y = px_y;
                }    

                color = colorFromRgba(rgba, img_width, img_height, px_x, px_y);
                color_prev = colorFromRgba(rgba, img_width, img_height, prev_x, prev_y);

                // index so a block is consecutive
                runs[(tile_idx * 256) + (inner_y * 16) + inner_x] = (uint16_t)(color_prev != color);
            }
        }
    }
};

struct FinalizeTrleFunctor
{
    const uint8_t *rgba;
    const uint16_t *runs; // boolean array, one index per pixel (1 indicates start of a new run)
    const uint16_t *num_runs; // total number of runs per tile
    const uint32_t *sums; // prefix sum array of num_runs
    uint8_t *trle;
    int img_width;
    int img_height;
    int trle_width;
    int trle_height;
    size_t size;
    FinalizeTrleFunctor(int width_in, int height_in, thrust::device_vector<uint8_t> const& rgba_in, thrust::device_vector<uint16_t> const& runs_in, thrust::device_vector<uint16_t>& num_runs_in, thrust::device_vector<uint32_t> const& sums_in, thrust::device_vector<uint8_t>& trle_out)  
    {
        rgba = thrust::raw_pointer_cast(rgba_in.data());
        runs = thrust::raw_pointer_cast(runs_in.data());
        num_runs = thrust::raw_pointer_cast(num_runs_in.data());
        sums = thrust::raw_pointer_cast(sums_in.data());
        trle = thrust::raw_pointer_cast(trle_out.data());
        img_width = width_in;
        img_height = height_in;
        trle_width = (img_width + 15) & ~0x0f; // round up to nearest multiple of 16
        trle_height = (img_height + 15) & ~0x0f; // round up to nearest multiple of 16
        size = trle_width * trle_height / 256;
    }
    __host__  __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < size)
        {
            int i;
            int tile_x = thread_id % (trle_width / 16);
            int tile_y = thread_id / (trle_width / 16);
            int px_x = tile_x * 16;
            int px_y = tile_y * 16;

			uint8_t x_increase = 0;
            uint32_t y_increase = 0;

            // rgba index of first pixel in our current tile
            uint32_t offset = (px_y * img_width * 4) + (px_x * 4);

            uint32_t run_count;
            uint32_t total_run_count = 0;

            // number of pixels in past tiles (index into runs)
            uint32_t index = (tile_y * (trle_width / 16) * 256) + (tile_x * 256) ;

            // for all the runs in the tile
            for (i = 0; i < num_runs[thread_id]; i++)
            {
                // go to index of next run and reset run_count
                index++;
                run_count = 0;

                // while pixel is the same color increase the count
                while (runs[index] == 0)
                {
                    run_count++;
                    index++;
                    total_run_count++;
                }
                total_run_count++;

                // trle indexed by block
				trle[(sums[thread_id] * 4) + (i * 4)] = run_count;
				trle[(sums[thread_id] * 4) + ((i * 4) + 1)] = rgba[offset + y_increase + x_increase];
				trle[(sums[thread_id] * 4) + ((i * 4) + 2)] = rgba[offset + y_increase + x_increase + 1];
				trle[(sums[thread_id] * 4) + ((i * 4) + 3)] = rgba[offset + y_increase + x_increase + 2];
                x_increase = (total_run_count % 16) * 4;
                y_increase = (total_run_count / 16) * 4 * img_width;
            }
        }
    }
};

struct DecodeTrleFunctor
{
    const uint8_t *trle;
    const uint32_t *sums; // prefix sum array of num_runs
    uint8_t *rgb;
    int width;
    int height;
    size_t size;
    DecodeTrleFunctor(int width_in, int height_in, thrust::device_vector<uint8_t> const& trle_in, thrust::device_vector<uint32_t> const& sums_in, thrust::device_vector<uint8_t>& rgb_out)
    {
        trle = thrust::raw_pointer_cast(trle_in.data());
        sums = thrust::raw_pointer_cast(sums_in.data());
        rgb = thrust::raw_pointer_cast(rgb_out.data());
        width = width_in;
        height = height_in;
        size = width * height / 256;
    }
    __host__  __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < size)
        {
            int i;
            int tile_x = thread_id % (width / 16); 
            int tile_y = thread_id / (width / 16);
            int px_x = tile_x * 16;
            int px_y = tile_y * 16;

            // index of first pixel in our current tile
            uint32_t rgb_offset = (px_y * width * 3) + (px_x * 3);
            int inner_count = 0;

            uint32_t trle_offset = sums[thread_id] * 4;
            uint16_t run_length;

            // until all 256 pixels are decoded
            while (inner_count < 256)
            {
                run_length = (uint16_t)trle[trle_offset] + 1;
                for (i = 0; i < run_length; i++)
                {
                    memcpy(rgb + rgb_offset, trle + trle_offset + 1, 3);
                    //rgb[rgb_offset] = trle[trle_offset + 1];
                    //rgb[rgb_offset + 1] = trle[trle_offset + 2];
                    //rgb[rgb_offset + 2] = trle[trle_offset + 3];
                    inner_count++;

                    if (inner_count % 16 == 0)
                    {
                        rgb_offset += (width - 15) * 3;
                    }
                    else
                    {
                        rgb_offset += 3;
                    }
                }

                trle_offset += 4;
            }
        }
    }
};


void extractTile4x4(const uint8_t *pixels, int x, int y, int img_width, int img_height, uint8_t out_tile[64])
{
    int i, j;
    int rows = min(img_height - y, 4);
    int cols = min(img_width - x, 4);
    uint8_t first[4];
    memcpy(first, pixels + ((y * img_width * 4) + (x * 4)), 4);
    for (i = 0; i < rows; i++)
    {
        memcpy(out_tile + (i * 16), pixels + (((y + i) * img_width * 4) + (x * 4)), cols * 4);
        for (j = cols; j < 4; j++)
        {
            memcpy(out_tile + ((i * 16) + (j * 4)), first, 4);
        }
    }
    for (i = rows; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            memcpy(out_tile + ((i * 16) + (j * 4)), first, 4);
        }
    }
}

// TODO: change to match Tile4x4
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

uint32_t colorFromRgba(const uint8_t *rgba, int width, int height, int x, int y)
{
    int ix = x;
    int iy = y;
    if (iy >= height)
    {
        iy = height - 1;
        ix = ix + (15 - (ix % 16));
    }
    ix = min(ix, width - 1);

    int offset = (iy * width * 4) + (ix * 4);

    uint32_t result = rgba[offset] << 24;
    result |= rgba[offset + 1] << 16;
    result |= rgba[offset + 2] << 8;
    result |= rgba[offset + 3];
    
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

uint64_t currentTime()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (ts.tv_sec * 1000000ull) + (ts.tv_nsec / 1000ull);
}

// ----------------------------------------------------------------------------- //


void initImageConverter(int width, int height)
{
    img_w = width;
    img_h = height;

    int out_width = (img_w + 15) & ~0x0f; // round up to nearest multiple of 16
    int out_height = (img_h + 15) & ~0x0f; // round up to nearest multiple of 16

    image_input_ptr = new thrust::device_vector<uint8_t>(img_w * img_h * 4);
    image_output_ptr = new thrust::device_vector<uint8_t>(out_width * out_height * 4);
    runs = new thrust::device_vector<uint16_t>(out_width * out_height);
    num_runs = new thrust::device_vector<uint16_t>((out_width * out_height) / 256);
    sums = new thrust::device_vector<uint32_t>((out_width * out_height) / 256);
}

void getDxt1Dimensions(int *dxt1_width, int *dxt1_height, uint32_t *size)
{
    *dxt1_width = (img_w + 3) & ~0x03; // round up to nearest multiple of 4
    *dxt1_height = (img_h + 3) & ~0x03; // round up to nearest multiple of 4
    *size = (*dxt1_width) * (*dxt1_height) / 2;
}

void getTrleDimensions(int *trle_width, int *trle_height, uint32_t *max_size, uint32_t *offset_size)
{
    *trle_width = (img_w + 15) & ~0x0f; // round up to nearest multiple of 16
    *trle_height = (img_h + 15) & ~0x0f; // round up to nearest multiple of 16
    *max_size = (*trle_width) * (*trle_height) * 4;
    *offset_size = (*trle_width) * (*trle_height) / 256;
}

void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray)
{
    uint64_t start = currentTime();

    thrust::copy(rgba, rgba + (img_w * img_h * 4), image_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per pixel
    thrust::for_each_n(thrust::device, it, img_w * img_h, GrayscaleFunctor(*image_input_ptr, *image_output_ptr));

    // copy image data back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (img_w * img_h), gray);

    uint64_t end = currentTime();
    printf("THRUST - Grayscale (%dx%d): %.6lf\n", img_w, img_h, (double)(end - start) / 1000000.0);
}

void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1)
{
    uint64_t start = currentTime();

    const int dxt1_width = (img_w + 3) & ~0x03; // round up to nearest multiple of 4
    const int dxt1_height = (img_h + 3) & ~0x03; // round up to nearest multiple of 4
    const int k = 16; // pixels per tile
    const int n = (dxt1_width * dxt1_height) / k; // number of tiles
    thrust::copy(rgba, rgba + (img_w * img_h * 4), image_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per 4x4 tile
    thrust::for_each_n(thrust::device, it, n, Dxt1Functor(img_w, img_h, *image_input_ptr, *image_output_ptr));

    // copy image data back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (dxt1_width * dxt1_height / 2), dxt1);

    uint64_t end = currentTime();
    printf("THRUST - DXT1 (%dx%d): %.6lf\n", img_w, img_h, (double)(end - start) / 1000000.0);
}

void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint32_t *run_offsets)
{
    uint64_t start = currentTime();

    const int trle_width = (img_w + 15) & ~0x0f; // round up to nearest multiple of 16
    const int trle_height = (img_h + 15) & ~0x0f; // round up to nearest multiple of 16
    const int k = 256; // pixels per tile
    const int n = (trle_width * trle_height) / k; // number of tiles
    thrust::copy(rgba, rgba + (img_w * img_h * 4), image_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per pixel
    thrust::for_each_n(thrust::device, it, trle_width * trle_height, TrleFunctor(img_w, img_h, *image_input_ptr, *runs));

    //std::cout << "RUNS" << std::endl;
    //thrust::copy(runs->begin(), runs->end(), std::ostream_iterator<uint16_t>(std::cout, "\n"));
    //std::cout << std::endl;

    // thrust reduce_by_key - sum number of new runs per tile
    thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0), thrust::placeholders::_1 / k), thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(n * k), thrust::placeholders::_1 / k), runs->begin(), thrust::discard_iterator<uint32_t>(), num_runs->begin());

    //std::cout << "NUM RUNS" << std::endl;
    //thrust::copy(num_runs->begin(), num_runs->end(), std::ostream_iterator<uint16_t>(std::cout, "\n"));
    //std::cout << std::endl;


    // thrust inclusive_scan (prefix sum) - create array where each index is sum of all numbers in num_runs before its index
    // results in the offset into our final trle array
    thrust::exclusive_scan(thrust::device, num_runs->begin(), num_runs->end(), sums->begin());

    // thrust for_each_n - one thread per 16x16 tile
    thrust::for_each_n(thrust::device, it, n, FinalizeTrleFunctor(img_w, img_h, *image_input_ptr, *runs, *num_runs, *sums, *image_output_ptr));

    // copy offset data to host
    uint32_t last_size;
    thrust::copy(sums->begin(), sums->end(), run_offsets);
    thrust::copy(num_runs->end() - 1, num_runs->end(), &last_size);
    *buffer_size = (run_offsets[n - 1] + last_size) * 4;

    // copy image data back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (*buffer_size), trle);

    uint64_t end = currentTime();
    printf("THRUST - TRLE (%dx%d): %.6lf\n", img_w, img_h, (double)(end - start) / 1000000.0);
}

void trleToRgb(uint8_t *trle, uint8_t *rgb, uint32_t buffer_size, uint32_t *run_offsets)
{
    uint64_t start = currentTime(); 

    const int trle_width = (img_w + 15) & ~0x0f; // round up to nearest multiple of 16
    const int trle_height = (img_h + 15) & ~0x0f; // round up to nearest multiple of 16
    const int k = 256; // pixels per tile
    const int n = (trle_width * trle_height) / k; // number of tiles
    thrust::copy(trle, trle + buffer_size, image_input_ptr->begin());
    thrust::copy(run_offsets, run_offsets + n, sums->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per 16x16 tile
    thrust::for_each_n(thrust::device, it, n, DecodeTrleFunctor(trle_width, trle_height, *image_input_ptr, *sums, *image_output_ptr));

    // copy image data back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (trle_width * trle_height * 3), rgb);

    uint64_t end = currentTime();
    printf("THRUST - Decode TRLE (%dx%d): %.6lf\n", trle_width, trle_height, (double)(end - start) / 1000000.0);
}

void finalizeImageConverter()
{
    cudaDeviceSynchronize();
}

