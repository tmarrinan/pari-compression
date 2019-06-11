#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

void ConvertRgbaImageToDxt1(uint8_t *pixels, int width, int height, uint8_t *dxt1);
void CreateDdsHeader(int width, int height, DdsHeader *header);
void ExtractBlock(uint32_t offset, uint8_t *pixels, int width, uint8_t out_block[64]);
void GetMinMaxColors(uint8_t block[64], uint8_t color_min[3], uint8_t color_max[3], uint8_t inset[3]);
uint16_t ColorTo565(uint8_t *rgb);
uint32_t ColorIndices(uint8_t block[64], uint8_t color_min[3], uint8_t color_max[3], uint8_t colors[16], uint8_t indices[16]);
uint32_t ColorDistance(uint8_t block1[64], int c1_offset, uint8_t block2[16], int c2_offset);
void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value);
void WriteUint32(uint8_t *buffer, uint32_t offset, uint32_t value);

int main (int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Error: please specify image to convert\n");
        exit(1);
    }

    // Use first command line parameter as input file (e.g. jpeg, png, ...)
    std::string filename = argv[1];

    // Load image into memory (force image format to RGBA)
    int img_w, img_h, img_c;
    uint8_t *pixels = stbi_load(filename.c_str(), &img_w, &img_h, &img_c, STBI_rgb_alpha);
    if (img_w % 4 != 0 || img_h % 4 != 0)
    {
        fprintf(stderr, "Error: image width and height must be multiples of 4 - please resize image\n");
        exit(1);
    }

    // Convert to DXT1
    uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
    ConvertRgbaImageToDxt1(pixels, img_w, img_h, dxt1);

    // Create header information for DDS file type
    DdsHeader header;
    CreateDdsHeader(img_w, img_h, &header);

    // Write to file
    FILE *fp = fopen("output.dds", "wb");
    fwrite(&header, sizeof(DdsHeader), 1, fp);
    fwrite(dxt1, 1, img_w * img_h / 2, fp);
    fclose(fp);

    delete[] dxt1;

    return 0;
}

// Convert an RGBA pixel buffer to a DXT1 pixel buffer
void ConvertRgbaImageToDxt1(uint8_t *pixels, int width, int height, uint8_t *dxt1)
{
    int i, j;
    uint32_t offset;
    uint32_t current_pos = 0;
    uint8_t block[64];
    uint8_t color_min[3];
    uint8_t color_max[3];
    uint8_t inset[3];
    uint8_t colors[16];
    uint8_t indices[16];

    // Loop over each set of 4 rows (starting at top of image)
    for (j = 0; j < height; j += 4)
    {
        // Loop over each set of 4 columns (starting at left of image)
        for (i = 0; i < width; i+=4)
        {
            offset = (j * width * 4) + (i * 4);
            
            ExtractBlock(offset, pixels, width, block);
            GetMinMaxColors(block, color_min, color_max, inset);
            WriteUint16(dxt1, current_pos, ColorTo565(color_max));
            WriteUint16(dxt1, current_pos + 2, ColorTo565(color_min));
            WriteUint32(dxt1, current_pos + 4, ColorIndices(block, color_min, color_max, colors, indices));

            current_pos += 8;
        }
    }
}

// Create the header info for the DXT1 image
void CreateDdsHeader(int width, int height, DdsHeader *header)
{
    header->dw_magic = 0x20534444; // 'DDS '
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

// Copy 4x4 block of RGBA pixels to `out_block`
void ExtractBlock(uint32_t offset, uint8_t *pixels, int width, uint8_t out_block[64])
{
    int i, j;
    uint32_t start = offset;
    
    for (j = 0; j < 4; j++)
    {
        for (i = 0; i < 16; i++)
        {
            out_block[j * 16 + i] = pixels[start + i];
        }
        start += width * 4;
    }
}

// Calculate minimum and maximum colore values inside a 4x4 block
void GetMinMaxColors(uint8_t block[64], uint8_t color_min[3], uint8_t color_max[3], uint8_t inset[3])
{
    int i;
    
    color_min[0] = 255; //Red
    color_min[1] = 255; //Green
    color_min[2] = 255; //Blue  
    color_max[0] = 0; //Red
    color_max[1] = 0; //Green
    color_max[2] = 0; //Blue
    
    for (i = 0; i < 16; i++) //In 4x4 block find the color min and color max.
    {
        if (block[i * 4]     < color_min[0]) color_min[0] = block[i * 4];
        if (block[i * 4 + 1] < color_min[1]) color_min[1] = block[i * 4 + 1];
        if (block[i * 4 + 2] < color_min[2]) color_min[2] = block[i * 4 + 2];
        if (block[i * 4]     > color_max[0]) color_max[0] = block[i * 4];
        if (block[i * 4 + 1] > color_max[1]) color_max[1] = block[i * 4 + 1];
        if (block[i * 4 + 2] > color_max[2]) color_max[2] = block[i * 4 + 2];
    }
    // why do this??
    inset[0] = (color_max[0] - color_min[0]) >> 4;
    inset[1] = (color_max[1] - color_min[1]) >> 4;
    inset[2] = (color_max[2] - color_min[2]) >> 4;
    
    color_min[0] = std::min(color_min[0] + inset[0], 255);
    color_min[1] = std::min(color_min[1] + inset[1], 255);
    color_min[2] = std::min(color_min[2] + inset[2], 255);
    color_max[0] = std::max(color_max[0] - inset[0], 0);
    color_max[1] = std::max(color_max[1] - inset[1], 0);
    color_max[2] = std::max(color_max[2] - inset[2], 0);
}

// Convert an 8 bits per channel RGB color to a 5,6,5 bits per channel RGB color
uint16_t ColorTo565(uint8_t *rgb)
{
    return ((rgb[0] >> 3) << 11) | ((rgb[1] >> 2) << 5) | (rgb[2] >> 3);
}

// Determine indices of nearest color for each pixel
uint32_t ColorIndices(uint8_t block[64], uint8_t color_min[3], uint8_t color_max[3], uint8_t colors[16], uint8_t indices[16])
{
    int i, j;

    uint8_t C565_5_MASK = 0xF8;   // 0xFF minus last three bits
    uint8_t C565_6_MASK = 0xFC;   // 0xFF minus last two bits

    colors[0] = (color_max[0] & C565_5_MASK) | (color_max[0] >> 5);
    colors[1] = (color_max[1] & C565_6_MASK) | (color_max[1] >> 6);
    colors[2] = (color_max[2] & C565_5_MASK) | (color_max[2] >> 5);

    colors[4] = (color_min[0] & C565_5_MASK) | (color_min[0] >> 5);
    colors[5] = (color_min[1] & C565_6_MASK) | (color_min[1] >> 6);
    colors[6] = (color_min[2] & C565_5_MASK) | (color_min[2] >> 5);
    
    colors[8] = (2 * colors[0] + colors[4]) / 3;
    colors[9] = (2 * colors[1] + colors[5]) / 3;
    colors[10] = (2 * colors[2] + colors[6]) / 3;
    colors[12] = (colors[0] + 2 * colors[4]) / 3;
    colors[13] = (colors[1] + 2 * colors[5]) / 3;
    colors[14] = (colors[2] + 2 * colors[6]) / 3;

    for (i = 0; i < 16; i++)
    {
        uint32_t min_distance = 195076; // (255 * 255 * 255) + 1
        for (j = 0; j < 4; j++)
        {
            uint32_t dist = ColorDistance(block, i * 4, colors, j * 4);
            if (dist < min_distance)
            {
                min_distance = dist;
                indices[i] = j;
            }
        }
    }
    
    uint32_t result = 0;
    for(i = 0; i < 16; i++)
    {
        result |= (indices[i] << (i << 1));
    }
    
    return result;
}

// Determine distance to one of the 4 possible colors for this block
uint32_t ColorDistance(uint8_t block1[64], int c1_offset, uint8_t block2[16], int c2_offset) // c1 offset 0-64, c2 offset 0-16
{

    int dx = block1[c1_offset] - block2[c2_offset]; //red
    int dy = block1[c1_offset + 1] - block2[c2_offset + 1]; //green
    int dz = block1[c1_offset + 2] - block2[c2_offset + 2]; //blue
    
    return (dx*dx) + (dy*dy) + (dz*dz);
}

// Write a 16-bit integer into a buffer
void WriteUint16(uint8_t *buffer, uint32_t offset, uint16_t value) {
    buffer[offset] = value & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
}

// Write a 32-bit integer into a buffer
void WriteUint32(uint8_t *buffer, uint32_t offset, uint32_t value) {
    buffer[offset] = value & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
}
