#ifndef PARICOMPRESS_H
#define PARICOMPRESS_H

#ifdef __cplusplus
extern "C" { 
#endif

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#define PARI_DLLEXPORT __declspec(dllexport)
#else
#define PARI_DLLEXPORT
#endif

typedef void** PariGpuBuffer;
typedef void* PariCGResource;
typedef void* PariCGResourceDescription;

enum PariCompressionType : uint8_t { Grayscale, Rgb, Rgba, Dxt1, ActivePixel };

// Standard PARI functions
PARI_DLLEXPORT PariGpuBuffer pariAllocateGpuBuffer(uint32_t width, uint32_t height, PariCompressionType type);
PARI_DLLEXPORT void pariRgbaBufferToGrayscale(uint8_t *rgba, uint32_t width, uint32_t height, PariGpuBuffer gpu_in_buf,
                                              PariGpuBuffer gpu_out_buf, uint8_t *gray);

// OpenGL - PARI functions
PARI_DLLEXPORT PariCGResource pariRegisterImage(uint32_t texture, PariCGResourceDescription *resrc_description_ptr);
PARI_DLLEXPORT void pariGetRgbaTextureAsGrayscale(PariCGResource cg_resource, PariCGResourceDescription resrc_description,
                                                  PariGpuBuffer gpu_out_buf, uint32_t texture, uint32_t width, uint32_t height, 
                                                  uint8_t *gray);


//void initImageConverter(int width, int height);
//void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray);
//void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1);
//void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint32_t *run_offsets);
//void finalizeImageConverter();
//void trleToRgb(uint8_t *trle, uint8_t *rgb, uint32_t buffer_size, uint32_t *run_offsets);

#ifdef __cplusplus
}
#endif

#endif // PARICOMPRESS_H
