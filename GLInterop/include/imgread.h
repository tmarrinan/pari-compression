#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

void initImageConverter(int width, int height, GLuint texture);
void getDxt1Dimensions(int *dxt1_width, int *dxt1_height, uint32_t *size);
void getTrleDimensions(int *trle_width, int *trle_height, uint32_t *max_size, uint32_t *offset_size);
void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray);
void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1);
void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint32_t *run_offsets);
void finalizeImageConverter();
void trleToRgb(uint8_t *trle, uint8_t *rgb, uint32_t buffer_size, uint32_t *run_offsets);

