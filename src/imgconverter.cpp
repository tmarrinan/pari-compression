#include "imgconverter.h"

static int img_w;
static int img_h;
static uint8_t *image_input_ptr;
static uint8_t *image_output_ptr;
static uint8_t *runs;
static uint16_t *num_runs;
static uint32_t *sums;


void extractTile4x4(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[64]);
void extractTile16x16(uint32_t offset, const uint8_t *pixels, int width, uint8_t out_tile[1024]);
void getMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
uint16_t colorTo565(uint8_t color[3]);
uint32_t colorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
uint32_t colorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
uint32_t colorFromRgba(const uint8_t *rgba, uint32_t offset);
void writeUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
void writeUint32(uint8_t *buffer, uint32_t offset, uint32_t value);

uint64_t currentTime();

/*
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
    const uint32_t *sums; // prefix sum array of num_runs
    uint8_t *trle;
    int width;
    int height;
    FinalizeTrleFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& rgba_input, thrust::device_vector<uint8_t> const& runs_input, thrust::device_vector<uint16_t>& num_runs_input, thrust::device_vector<uint32_t> const& sums_input, thrust::device_vector<uint8_t>& trle_output)  
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

            uint32_t run_count;
            uint32_t total_run_count = 0;

            // number of pixels in past tiles (index into runs)
            uint32_t index = (tile_x * 256) + (tile_y * (width / 16) * 256) ;

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
                y_increase = (total_run_count / 16) * 4 * width;
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
    DecodeTrleFunctor(int width_input, int height_input, thrust::device_vector<uint8_t> const& trle_input, thrust::device_vector<uint32_t> const& sums_input, thrust::device_vector<uint8_t>& rgb_output)
    {
        trle = thrust::raw_pointer_cast(trle_input.data());
        sums = thrust::raw_pointer_cast(sums_input.data());
        rgb = thrust::raw_pointer_cast(rgb_output.data());
        width = width_input;
        height = height_input;
    }
    __host__  __device__ void operator()(int thread_id)
    {
        // px_ (x and y pixel indices)
        // tile_ (x and y tile indices)
        if (thread_id < width * height / 256)
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
                    rgb[rgb_offset] = trle[trle_offset + 1];
                    rgb[rgb_offset + 1] = trle[trle_offset + 2];
                    rgb[rgb_offset + 2] = trle[trle_offset + 3];
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
        color_min[0] = std::min(color_min[0], tile[i * 4 + 0]);
        color_min[1] = std::min(color_min[1], tile[i * 4 + 1]);
        color_min[2] = std::min(color_min[2], tile[i * 4 + 2]);
        color_max[0] = std::max(color_max[0], tile[i * 4 + 0]);
        color_max[1] = std::max(color_max[1], tile[i * 4 + 1]);
        color_max[2] = std::max(color_max[2], tile[i * 4 + 2]);
    }
    
    inset[0] = (color_max[0] - color_min[0]) >> 4;
    inset[1] = (color_max[1] - color_min[1]) >> 4;
    inset[2] = (color_max[2] - color_min[2]) >> 4;
    
    color_min[0] = std::min(color_min[0] + inset[0], 255);
    color_min[1] = std::min(color_min[1] + inset[1], 255);
    color_min[2] = std::min(color_min[2] + inset[2], 255);
    color_max[0] = std::max(color_max[0] - inset[0], 0);
    color_max[1] = std::max(color_max[1] - inset[1], 0);
    color_max[2] = std::max(color_max[2] - inset[2], 0);
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

uint64_t currentTime()
{
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    return us.count();

    //struct timespec ts;
    //timespec_get(&ts, TIME_UTC);
    //return (ts.tv_sec * 1000000ull) + (ts.tv_nsec / 1000ull);
}

// ----------------------------------------------------------------------------- //


void initImageConverter(int width, int height)
{
    img_w = width;
    img_h = height;

    image_input_ptr = new uint8_t[img_w * img_h * 4];
    image_output_ptr = new uint8_t[img_w * img_h * 4];
    runs = new uint8_t[img_w * img_h];
    num_runs = new uint16_t[(img_w * img_h) / 256];
    sums = new uint32_t[(img_w * img_h) / 256];

    int target[1];
    target[0] = omp_get_initial_device();
    std::cout << "Total Devices: " << omp_get_num_devices() << std::endl;
    std::cout << "Target: " << target[0] << std::endl;
}

void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray)
{
    double start = omp_get_wtime();

    int i;
    const int size = img_w * img_h;
    const int n_rgba = img_w * img_h * 4;
    const int n_gray = img_w * img_h;

    double start_compute, end_compute;
    //#pragma omp target data map(to: rgba[0:n_rgba]) map(from: gray[0:n_gray]) device(0)
    {
        start_compute = omp_get_wtime();

        //#pragma omp target map(to: rgba[0:n_rgba]) map(from: gray[0:n_gray]) device(0)
        //#pragma omp teams distribute parallel for private(i)
        #pragma omp parallel for private(i)
        for (i = 0; i < size; i++)
        {
            float red = (float)rgba[4 * i + 0];
            float green = (float)rgba[4 * i + 1];
            float blue = (float)rgba[4 * i + 2];
            gray[i] = (uint8_t)(0.299f * red + 0.587f * green + 0.114f * blue);
        }

        end_compute = omp_get_wtime();
    }

    double end = omp_get_wtime();
    printf("OpenMP - Grayscale (%dx%d): %.6lf total, %.6lf compute\n", img_w, img_h, end - start, end_compute - start_compute);
}

void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1)
{
    double start = omp_get_wtime();

    int i;
    const int k = 16; // pixels per tile
    const int size = (img_w * img_h) / k; // number of tiles
    const int n_rgba = img_w * img_h * 4;
    const int n_dxt1 = img_w * img_h / 2;

    double start_compute, end_compute;
    #pragma omp target data map(to: rgba[0:n_rgba]) map(from: dxt1[0:n_dxt1]) device(0)
    {
        start_compute = omp_get_wtime();

        #pragma omp target map(to: rgba[0:n_rgba]) map(from: dxt1[0:n_dxt1]) device(0)
        #pragma omp teams distribute parallel for private(i)
        //#pragma omp parallel for private(i)
        for (i = 0; i < size; i++)
        {
            uint8_t tile[64];
            uint8_t color_min[3];
            uint8_t color_max[3];

            int tile_x = i % (img_w / 4);
            int tile_y = i / (img_w / 4);
            int px_x = tile_x * 4;
            int px_y = tile_y * 4;

            uint32_t offset = (px_y * img_w * 4) + (px_x * 4);
            uint32_t write_pos = (tile_y * (img_w / 4) * 8) + (tile_x * 8);

            extractTile4x4(offset, rgba, img_w, tile);
            getMinMaxColors(tile, color_min, color_max);
            writeUint16(dxt1, write_pos, colorTo565(color_max));
            writeUint16(dxt1, write_pos + 2, colorTo565(color_min));
            writeUint32(dxt1, write_pos + 4, colorIndices(tile, color_min, color_max));
        }

        end_compute = omp_get_wtime();
    }

    double end = omp_get_wtime();
    printf("OpenMP - DXT1 (%dx%d): %.6lf total, %.6lf compute\n", img_w, img_h, end - start, end_compute - start_compute);
}

void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint32_t *run_offsets)
{
    /*
    uint64_t start = currentTime();

    const int k = 256; // pixels per tile
    const int n = (img_w * img_h) / k; // number of tiles
    thrust::copy(rgba, rgba + (img_w * img_h * 4), image_input_ptr->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per pixel
    thrust::for_each_n(thrust::device, it, img_w * img_h, TrleFunctor(img_w, img_h, *image_input_ptr, *runs));

    // thrust reduce_by_key - sum number of new runs per tile
    thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(0), thrust::placeholders::_1 / k), thrust::make_transform_iterator(thrust::counting_iterator<uint16_t>(n * k), thrust::placeholders::_1 / k), runs->begin(), thrust::discard_iterator<uint16_t>(), num_runs->begin());

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
    */
}

void trleToRgb(uint8_t *trle, uint8_t *rgb, uint32_t buffer_size, uint32_t *run_offsets)
{
    /*
    uint64_t start = currentTime(); 

    const int k = 256; // pixels per tile
    const int n = (img_w * img_h) / k; // number of tiles
    thrust::copy(trle, trle + buffer_size, image_input_ptr->begin());
    thrust::copy(run_offsets, run_offsets + n, sums->begin());
    thrust::counting_iterator<size_t> it(0);

    // thrust for_each_n - one thread per 16x16 tile
    thrust::for_each_n(thrust::device, it, n, DecodeTrleFunctor(img_w, img_h, *image_input_ptr, *sums, *image_output_ptr));

    // copy image data back to host
    thrust::copy(image_output_ptr->begin(), image_output_ptr->begin() + (img_w * img_h * 3), rgb);

    uint64_t end = currentTime();
    printf("THRUST - Decode TRLE (%dx%d): %.6lf\n", img_w, img_h, (double)(end - start) / 1000000.0);
    */
}

void finalizeImageConverter()
{
    //cudaDeviceSynchronize();
}

