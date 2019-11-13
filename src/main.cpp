#include <iostream>
#include "imgconverter.h"

typedef struct DdsPixelFormat {
    uint32_t dw_size;
    uint32_t dw_flags;
    uint32_t dw_four_cc;
    uint32_t dw_rgb_bit_count;
    uint32_t dw_r_bit_mask;
    uint32_t dw_g_bit_mask;
    uint32_t dw_b_bit_mask;
    uint32_t dw_a_bit_mask;
} DdsPixelFormat;

typedef struct DdsHeader {
    uint32_t dw_magic;
    uint32_t dw_size;
    uint32_t dw_flags;
    uint32_t dw_height;
    uint32_t dw_width;
    uint32_t dw_pitch_or_linear_size;
    uint32_t dw_depth;
    uint32_t dw_mip_map_count;
    uint32_t dw_reserved1[11];
    DdsPixelFormat ddspf;
    uint32_t dw_caps;
    uint32_t dw_caps2;
    uint32_t dw_caps3;
    uint32_t dw_caps4;
    uint32_t dw_reserved2;
} DdsHeader;


void CreateDdsHeader(int width, int height, DdsHeader *header);
void ReadPpm(const char *filename, int *width, int *height, uint8_t **pixels);
void SavePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void SaveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);

int main(int argc, char **argv)
{
    // read rgba image (ppm file)
    int img_w;
    int img_h;
    uint8_t *rgba;
    ReadPpm("resrc/rletest_64x32.ppm", &img_w, &img_h, &rgba);
    // allocate output images
  
    uint8_t *gray = new uint8_t[img_w * img_h];
    uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
    uint8_t *trle = new uint8_t[img_w * img_h];
    uint8_t *rgba_trle = new uint8_t[img_w*img_h*4];
    uint8_t *run_offsets = new uint8_t[img_w*img_h/256];
    // buffer size for variable sized outputs
    uint32_t size;
    uint32_t inputSize = 39;

    // initialize image converter
    InitImageConverter(img_w, img_h);
    
   // convert rgba image to grayscale image
   /*RgbaToGrayscale(rgba, gray);
    SavePgm("cuda_result_gray.pgm", img_w, img_h, gray);*/
    
    // convert rgba image to dxt1 image
   // RgbaToDxt1(rgba, dxt1);
   // SaveDds("cuda_result_dxt1.dds", img_w, img_h, dxt1);
    
    //convert rgba image to trle image
    RgbaToTrle(rgba, trle, &size, run_offsets);
     printf("size main: %d\n", size);
    TrleToRgba(rgba_trle,trle,&inputSize,run_offsets);
    for(int i=0; i<img_w*img_h/256; i++)
    {
	    printf("runs main %d , ", run_offsets[i]);
    }
    //SaveDds("cuda_result_trle.dds", img_w,img_h,trle);

    // clean up
    FinalizeImageConverter();
    
    return 0;
}

void CreateDdsHeader(int width, int height, DdsHeader *header)
{
    header->dw_magic = 0x20534444; // 'DSS '
    header->dw_size = 124;
    header->dw_flags = 0x1007;
    header->dw_height = height;
    header->dw_width = width;
    header->dw_pitch_or_linear_size = width * height / 2;
    header->dw_depth = 0;
    header->dw_mip_map_count = 0;
    
    char four_cc[4] = {'D', 'X', 'T', '1'};
    header->ddspf.dw_size = 32;
    header->ddspf.dw_flags = 0x4;
    header->ddspf.dw_four_cc = *(reinterpret_cast<uint32_t*>(four_cc));
    header->ddspf.dw_rgb_bit_count = 16;
    header->ddspf.dw_r_bit_mask = 0xF800;
    header->ddspf.dw_g_bit_mask = 0x07E0;
    header->ddspf.dw_b_bit_mask = 0x001F;
    header->ddspf.dw_a_bit_mask = 0x0000;
    
    header->dw_caps = 0x1000;
    header->dw_caps2 = 0x0;
}

void ReadPpm(const char *filename, int *width, int *height, uint8_t **pixels)
{
    FILE *fp = fopen(filename, "rb");
    int header_count = 0;
    ssize_t read;
    char *line = NULL;
    size_t len;
    while (header_count < 3)
    {
        read = getline(&line, &len, fp);
        if (len > 0 && line[0] != '#')
        {
            if (header_count == 1)
            {
                sscanf(line, "%d %d", width, height);
            }
            header_count++;
        }
    }
    uint8_t *tmp = new uint8_t[(*width) * (*height) * 3];
    *pixels = new uint8_t[(*width) * (*height) * 4];
    fread(tmp, (*width) * (*height) * 3, 1, fp);
    int i;
    for (i = 0; i < (*width) * (*height); i++)
    {
        (*pixels)[4 * i + 0] = tmp[3 * i + 0];
        (*pixels)[4 * i + 1] = tmp[3 * i + 1];
        (*pixels)[4 * i + 2] = tmp[3 * i + 2];
        (*pixels)[4 * i + 3] = 255;
    }
}

void SavePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%u %u\n255\n", width, height);
    fwrite(pixels, width * height, 1, fp);
    fclose(fp);
}

void SaveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    DdsHeader header;
    CreateDdsHeader(width, height, &header);
    
    FILE *fp = fopen(filename, "wb");
    fwrite(&header, sizeof(DdsHeader), 1, fp);
    fwrite(pixels, width * height / 2, 1, fp);
    fclose(fp);
}
