#include "readpx.h"

uint64_t CurrentTime();
__global__ void RgbaToGrayscaleKernel(cudaSurfaceObject_t rgba, uint8_t *gray, int width, int height);
__global__ void RgbaToTileGrayscaleKernel(cudaSurfaceObject_t rgba, uint8_t *gray, int width, int height);
__global__ void RgbaToDxt1Kernel(cudaSurfaceObject_t rgba, uint8_t *dxt1, int width, int height);
__device__ void ExtractTile4x4(uint32_t x, uint32_t y, cudaSurfaceObject_t pixels, int width, uint8_t out_tile[64]);
__device__ void GetMinMaxColors(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__device__ uint16_t ColorTo565(uint8_t color[3]);
__device__ uint32_t ColorDistance(uint8_t tile[64], int t_offset, uint8_t colors[16], int c_offset);
__device__ uint32_t ColorIndices(uint8_t tile[64], uint8_t color_min[3], uint8_t color_max[3]);
__device__ void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
__device__ void WriteUint32(uint8_t *buffer, uint32_t offset, uint32_t value);


// CUDA Kernels
__global__ void RgbaToGrayscaleKernel(cudaSurfaceObject_t rgba, uint8_t *gray, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < width * height)
    {
        uchar4 color;
        surf2Dread(&color, rgba, 4 * (tid % width), tid / width);
        gray[tid] = (uint8_t)(0.299 * color.x + 0.587 * color.y + 0.114 * color.z);
    }
}

__global__ void RgbaToTileGrayscaleKernel(cudaSurfaceObject_t rgba, uint8_t *gray, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < width * height / 16)
    {
        int tile_x = tid % (width / 4);
        int tile_y = tid / (width / 4);
        int px_x = tile_x * 4;
        int px_y = tile_y * 4;

        int i, j;
        uchar4 color;
        for (j = px_y; j < px_y + 4; j++)
        {
            for (i = px_x; i < px_x + 4; i++)
            {
                surf2Dread(&color, rgba, 4 * i, j);
                gray[j * width + i] = (uint8_t)(0.299 * color.x + 0.587 * color.y + 0.114 * color.z);
            }
        }
    }
}

__global__ void RgbaToDxt1Kernel(cudaSurfaceObject_t rgba, uint8_t *dxt1, int width, int height)
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

        uint32_t write_pos = (tile_y * (width / 4) * 8) + (tile_x * 8);

        ExtractTile4x4(px_x, px_y, rgba, width, tile);
        GetMinMaxColors(tile, color_min, color_max);
        WriteUint16(dxt1, write_pos, ColorTo565(color_max));
        WriteUint16(dxt1, write_pos + 2, ColorTo565(color_min));
        WriteUint32(dxt1, write_pos + 4, ColorIndices(tile, color_min, color_max));
    }
}

__device__ void ExtractTile4x4(uint32_t x, uint32_t y, cudaSurfaceObject_t pixels, int width, uint8_t out_tile[64])
{
    int i, j;
    uchar4 color;
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 4; i++)
        {
            surf2Dread(&color, pixels, 4 * (x + i), y);
            out_tile[(16 * j) + (4 * i)] = color.x;
            out_tile[(16 * j) + (4 * i) + 1] = color.y;
            out_tile[(16 * j) + (4 * i) + 2] = color.z;
            out_tile[(16 * j) + (4 * i) + 3] = color.w;
        }
        y++;
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

__device__ void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value)
{
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


// C++ Functions
void BindCudaResourceToTexture(struct cudaGraphicsResource **resource, GLuint texture)
{
    cudaGraphicsGLRegisterImage(resource, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
}

void AllocateGpuOutput(void **dev_ptr, size_t size)
{
    cudaMalloc(dev_ptr, size);
}

void ReadRgbaTextureAsRgb(GLuint texture, uint8_t *rgb)
{
    uint64_t start = CurrentTime();

    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glBindTexture(GL_TEXTURE_2D, 0);

    uint64_t stop = CurrentTime();
    printf("RGBA to RGB\n  compute time: %.3f ms\n  total time: %.3lf ms\n", 0.0, (double)(stop - start) / 1000.0);
}

void ReadRgbaTextureAsRgba(GLuint texture, uint8_t *rgba)
{
    uint64_t start = CurrentTime();

    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glBindTexture(GL_TEXTURE_2D, 0);

    uint64_t stop = CurrentTime();
    printf("RGBA to RGBA\n  compute time: %.3f ms\n  total time: %.3lf ms\n", 0.0, (double)(stop - start) / 1000.0);
}

void ReadRgbaTextureAsGrayscale(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *gpu_gray, uint8_t *gray)
{
    cudaEvent_t start_comp, stop_comp;
    cudaEventCreate(&start_comp);
    cudaEventCreate(&stop_comp);

    uint64_t start = CurrentTime();
    cudaEventRecord(start_comp, 0);

    glBindTexture(GL_TEXTURE_2D, texture);
    cudaGraphicsMapResources(1, resource, 0);
    cudaArray *array;
    cudaGraphicsSubResourceGetMappedArray(&array, *resource, 0, 0);
    struct cudaResourceDesc description;
    memset(&description, 0, sizeof(description));
    description.resType = cudaResourceTypeArray;
    description.res.array.array = array;
    cudaSurfaceObject_t target;
    cudaCreateSurfaceObject(&target, &description);
    glBindTexture(GL_TEXTURE_2D, 0);

    int block_size = 256;
    // normal way (each pixel individually)
    //int num_blocks = (width * height + block_size - 1) / block_size;
    //RgbaToGrayscaleKernel<<<num_blocks, block_size>>>(target, gpu_gray, width, height);
    // tile-based way (4x4 tiles of pixels per thread)
    int num_blocks = ((width * height / 16) + block_size - 1) / block_size;
    RgbaToTileGrayscaleKernel<<<num_blocks, block_size>>>(target, gpu_gray, width, height);

    cudaEventRecord(stop_comp, 0);
    
    cudaMemcpy(gray, gpu_gray, width * height, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_comp);

    uint64_t stop = CurrentTime();
    float compute_ms;
    cudaEventElapsedTime(&compute_ms, start_comp, stop_comp);
    printf("RGBA to Grayscale\n  compute time: %.3f ms\n  total time: %.3lf ms\n", compute_ms, (double)(stop - start) / 1000.0);
}

void ReadRgbaTextureAsDxt1(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *gpu_dxt1, uint8_t *dxt1)
{
    cudaEvent_t start_comp, stop_comp;
    cudaEventCreate(&start_comp);
    cudaEventCreate(&stop_comp);

    uint64_t start = CurrentTime();
    cudaEventRecord(start_comp, 0);

    glBindTexture(GL_TEXTURE_2D, texture);
    cudaGraphicsMapResources(1, resource, 0);
    cudaArray *array;
    cudaGraphicsSubResourceGetMappedArray(&array, *resource, 0, 0);
    struct cudaResourceDesc description;
    memset(&description, 0, sizeof(description));
    description.resType = cudaResourceTypeArray;
    description.res.array.array = array;
    cudaSurfaceObject_t target;
    cudaCreateSurfaceObject(&target, &description);
    glBindTexture(GL_TEXTURE_2D, 0);

    int block_size = 256;
    int num_blocks = ((width * height / 16) + block_size - 1) / block_size;
    RgbaToDxt1Kernel<<<num_blocks, block_size>>>(target, gpu_dxt1, width, height);
    
    cudaEventRecord(stop_comp, 0);

    cudaMemcpy(dxt1, gpu_dxt1, width * height / 2, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_comp);

    uint64_t stop = CurrentTime();
    float compute_ms;
    cudaEventElapsedTime(&compute_ms, start_comp, stop_comp);
    printf("RGBA to DXT1\n  compute time: %.3f ms\n  total time: %.3lf ms\n", compute_ms, (double)(stop - start) / 1000.0);
}

uint64_t CurrentTime()
{
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::chrono::system_clock::duration duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

