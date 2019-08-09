#include "readpx.h"

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
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ReadRgbaTextureAsGrayscale(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *gpu_gray, uint8_t *gray)
{
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
    int num_blocks = (width * height + block_size - 1) / block_size;
    RgbaToGrayscaleKernel<<<num_blocks, block_size>>>(target, gpu_gray, width, height);
    cudaMemcpy(gray, gpu_gray, width * height, cudaMemcpyDeviceToHost);
}

void ReadRgbaTextureAsDxt1(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *dxt1)
{

}

