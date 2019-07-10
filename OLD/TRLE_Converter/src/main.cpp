#include <iostream>
#include <string>
using namespace std;
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// Perform TRLE data compression algorithm



bool equalTo(uint8_t arr1[], uint8_t arr2[]) ;
void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void ConvertRgbaImageToTRLE(uint8_t *pixels, int width, int height, uint8_t *TRLE)


int main (int argc, char **argv)
{
	uint32_t size;
    if (argc < 2)
    {
        fprintf(stderr, "Error: please specify image to convert\n");
        exit(1);
    }

    // Use first command line parameter as input file (e.g. jpeg, png, ...)
    std::string filename = argv[1];

    // Load image into memory (force image format to RGBA)
    int img_w, img_h, img_c;
    uint8_t *pixels = stbi_load(filename.c_str(), &img_w, &img_h, &img_c, STBI_rgb);

	if (img_w % 2 != 0 || img_h % 2 != 0)
    {
        fprintf(stderr, "Error: image width and height must be multiples of 2 - please resize image\n");
        exit(1);
    }

	uint8_t *TRLE= new uint8_t[img_w * img_h / 2];
    ConvertRgbaImageToDxt1(pixels, img_w, img_h, TRLE);



	
    return 0;
}

// Convert an RGBA pixel buffer to a TRLE pixel buffer
void ConvertRgbaImageToTRLE(uint8_t *pixels, int width, int height, uint8_t *TRLE)
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
            printf("%d", block[2]);
            GetMinMaxColors(block, color_min, color_max, inset);
            
            WriteUint16(dxt1, current_pos, ColorTo565(color_max));
            WriteUint16(dxt1, current_pos + 2, ColorTo565(color_min));
            WriteUint32(dxt1, current_pos + 4, ColorIndices(block, color_min, color_max, colors, indices));

            current_pos += 8;
        }
    }
}

void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P6\n%u %u\n255\n", width, height);
	fwrite(pixels, width * height * 3, 1, fp);
	fclose(fp);
}


void Decode(uint8_t *decodedRLE, uint8_t *RLE, uint32_t size)
{
	uint8_t count;
	for(int i=0; i<size; i= i+4)
	{
		//one count for RGB = 3 total
		count =decodedRLE[i] *3;
		for(int j=1; j<count; j++)
		{
			//decoded increments while RLE loops through either 1,2,3 RGB
			decodedRLE[i+j] = RLE[i+(j%4)];
		}

	}

}


