#include <stdio.h>
#include <stdint.h>

void InitImageConverter(int width, int height);
void RgbaToGrayscale(uint8_t *rgba, uint8_t *gray);
void RgbaToDxt1(uint8_t *rgba, uint8_t *dxt1);
void RgbaToTrle(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size,uint8_t *run_offsets );
void FinalizeImageConverter();
void TrleToRgba(uint8_t *rgba, uint8_t *trle, uint32_t *buffer_size,uint8_t *run_offsets);
