#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>

void initImageConverter(int width, int height);
void rgbaToGrayscale(uint8_t *rgba, uint8_t *gray);
void rgbaToDxt1(uint8_t *rgba, uint8_t *dxt1);
void rgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size, uint32_t *run_offsets);
void finalizeImageConverter();
void trleToRgb(uint8_t *trle, uint8_t *rgb, uint32_t buffer_size, uint32_t *run_offsets);

