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

typedef void** PariGpuBuffer;
typedef void* PariCGResource;
typedef void* PariCGResourceDescription;

enum PariCompressionType : uint8_t { Rgba, Depth, Grayscale, Rgb, Dxt1, ActivePixel };

// Standard PARI functions
PARI_DLLEXPORT void pariSetGpuDevice(int device);
PARI_DLLEXPORT PariGpuBuffer pariAllocateGpuBuffer(uint32_t width, uint32_t height, PariCompressionType type);
PARI_DLLEXPORT void pariRgbaBufferToGrayscale(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                              PariGpuBuffer gpu_out_buf, uint8_t *gray);
PARI_DLLEXPORT void pariRgbaBufferToDxt1(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                         PariGpuBuffer gpu_out_buf, uint8_t *dxt1);
PARI_DLLEXPORT void pariRgbaDepthBufferToActivePixel(uint8_t *rgba, float *depth, uint32_t width, uint32_t height,
                                                     PariGpuBuffer gpu_rgba_in_buf, PariGpuBuffer gpu_depth_in_buf,
                                                     PariGpuBuffer gpu_out_buf, uint8_t *active_pixel, uint32_t *active_pixel_size);

// OpenGL - PARI functions
PARI_DLLEXPORT PariCGResource pariRegisterImage(uint32_t texture, PariCGResourceDescription *resrc_description_ptr);
PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  uint32_t texture, PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, 
                                                  uint8_t *gray);
PARI_DLLEXPORT void pariGetRgbaTextureAsDxt1(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                             uint32_t texture, PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, 
                                             uint8_t *dxt1);
PARI_DLLEXPORT void pariGetRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                         uint32_t texture_color, PariCGResource cg_resource_depth,
                                                         PariCGResourceDescription resrc_description_depth, uint32_t texture_depth,
                                                         PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *active_pixel,
                                                         uint32_t *active_pixel_size);


#ifdef __cplusplus
}
#endif

#endif // PARICOMPRESS_H
