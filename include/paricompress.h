#ifndef PARICOMPRESS_H
#define PARICOMPRESS_H

#ifdef __cplusplus
extern "C" { 
#endif

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

#ifdef _WIN32
#define PARI_DLLEXPORT __declspec(dllexport)
#else
#define PARI_DLLEXPORT
#endif

#define PARI_DEVICE_OPENGL INT_MAX
#define PARI_IMAGE_RGBA            0x01
#define PARI_IMAGE_DEPTH32F        0x02
#define PARI_IMAGE_GRAYSCALE       0x03
#define PARI_IMAGE_RGB             0x04
#define PARI_IMAGE_DXT1            0x05
#define PARI_IMAGE_ACTIVE_PIXEL    0x06
#define PARI_TIME_COMPUTE          0x07
#define PARI_TIME_MEMORY_TRANSFER  0x08
#define PARI_TIME_TOTAL            0x09

typedef void** PariGpuBuffer;
typedef void* PariCGResource;
typedef void* PariCGResourceDescription;
typedef int PariCompressionType;
typedef int PariEnum;


// Standard PARI functions
PARI_DLLEXPORT void pariSetGpuDevice(int device);
PARI_DLLEXPORT void pariAllocateCpuBuffer(void **buffer, uint32_t size);
PARI_DLLEXPORT void pariFreeCpuBuffer(void *buffer);
PARI_DLLEXPORT PariGpuBuffer pariAllocateGpuBuffer(uint32_t width, uint32_t height, PariEnum type);
PARI_DLLEXPORT void pariFreeGpuBuffer(PariGpuBuffer buffer, PariEnum type);
PARI_DLLEXPORT void pariRgbaBufferToGrayscale(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                              PariGpuBuffer gpu_out_buf, uint8_t *gray);
PARI_DLLEXPORT void pariRgbaBufferToDxt1(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                         PariGpuBuffer gpu_out_buf, uint8_t *dxt1);
PARI_DLLEXPORT void pariRgbaDepthBufferToActivePixel(uint8_t *rgba, float *depth, uint32_t width, uint32_t height,
                                                     PariGpuBuffer gpu_rgba_in_buf, PariGpuBuffer gpu_depth_in_buf,
                                                     PariGpuBuffer gpu_out_buf, uint8_t *active_pixel, uint32_t *out_size);
PARI_DLLEXPORT double pariGetTime(PariEnum time);

// OpenGL - PARI functions
PARI_DLLEXPORT PariCGResource pariRegisterImage(uint32_t texture, PariCGResourceDescription *resrc_description_ptr);
PARI_DLLEXPORT void pariUnregisterImage(PariCGResource resrc, PariCGResourceDescription resrc_description);
PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *gray);
PARI_DLLEXPORT void pariGetRgbaTextureAsDxt1(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                             PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *dxt1);
PARI_DLLEXPORT void pariGetRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                         PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                         PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *active_pixel,
                                                         uint32_t *out_size);
PARI_DLLEXPORT void pariGetSubRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                            PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                            PariGpuBuffer gpu_out_buf, uint32_t ap_width, uint32_t ap_height, int32_t *ap_viewport,
                                                            int32_t *texture_viewport, uint8_t *active_pixel, uint32_t *out_size);


#ifdef __cplusplus
}
#endif

#endif // PARICOMPRESS_H
