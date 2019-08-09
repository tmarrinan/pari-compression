#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

void BindCudaResourceToTexture(struct cudaGraphicsResource **resource, GLuint texture);
void AllocateGpuOutput(void **dev_ptr, size_t size);
void ReadRgbaTextureAsRgb(GLuint texture, uint8_t *rgb);
void ReadRgbaTextureAsRgba(GLuint texture, uint8_t *rgba);
void ReadRgbaTextureAsGrayscale(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *gpu_gray, uint8_t *gray);
void ReadRgbaTextureAsDxt1(struct cudaGraphicsResource **resource, GLuint texture, uint32_t width, uint32_t height, uint8_t *gpu_dxt1, uint8_t *dxt1);

