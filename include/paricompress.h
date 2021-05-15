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
#define PARI_COMPUTE_TIME 0x01
#define PARI_MEMORY_TRANSFER_TIME 0x02
#define PARI_TOTAL_TIME 0x03

typedef void** PariGpuBuffer;
typedef void* PariCGResource;
typedef void* PariCGResourceDescription;
typedef int PariCompressionType;
typedef int PariEnum;

enum { PariCompressionType_Rgba, PariCompressionType_Depth, PariCompressionType_Grayscale, PariCompressionType_Rgb,
       PariCompressionType_Dxt1, PariCompressionType_ActivePixel };


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
PARI_DLLEXPORT double pariGetTime(PariEnum time);

// OpenGL - PARI functions
PARI_DLLEXPORT PariCGResource pariRegisterImage(uint32_t texture, PariCGResourceDescription *resrc_description_ptr);
PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *gray);
PARI_DLLEXPORT void pariGetRgbaTextureAsDxt1(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                             PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *dxt1);
PARI_DLLEXPORT void pariGetRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                         PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                         PariGpuBuffer gpu_out_buf, uint32_t width, uint32_t height, uint8_t *active_pixel,
                                                         uint32_t *active_pixel_size);
PARI_DLLEXPORT void pariGetSubRgbaDepthTextureAsActivePixel(PariCGResource cg_resource_color, PariCGResourceDescription resrc_description_color,
                                                            PariCGResource cg_resource_depth, PariCGResourceDescription resrc_description_depth,
                                                            PariGpuBuffer gpu_out_buf, uint32_t texture_width, uint32_t texture_height,
                                                            int32_t *texture_viewport, uint32_t ap_width, uint32_t ap_height,
                                                            int32_t *ap_viewport, uint8_t *active_pixel, uint32_t *active_pixel_size);


#ifdef __cplusplus
}
#endif

#endif // PARICOMPRESS_H
