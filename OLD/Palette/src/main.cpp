#include <iostream>
#include <string>
using namespace std;
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>


// Perform Run Length Encoding (RLE) data compression algorithm
//format [NumberPixels, Red, Green, Blue]
typedef struct Palette {
	uint8_t size;
	uint8_t *colors;
    uint8_t *indicies;
} Palette;


void ConvertToPalette(uint8_t *pixels, int width, int height, uint8_t *compressed);

void SavePPM(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void Decode(uint8_t *decodedRLE, uint8_t *RLE, uint32_t size);
bool isInside(uint8_t *colors, uint8_t red, uint8_t green, uint8_t blue, uint8_t num_colors, uint8_t pixel_num, uint8_t *indicies);
void addToPalette(uint16_t num_colors, uint32_t offset, uint8_t *pixels, uint8_t *palette);
void ExtractBlock(uint32_t offset, uint8_t *pixels, int width, uint8_t out_block[768]);
void GetColors(uint8_t block[768], uint8_t *indicies, uint8_t *colors, uint8_t &size);
uint8_t Indicies(uint8_t size, uint8_t *indices, uint8_t *result, uint8_t *colors);

int main ()//int argc, char **argv)
{
	/*uint32_t size;
    if (argc < 2)
    {
        fprintf(stderr, "Error: please specify image to convert\n");
        exit(1);
    }

    // Use first command line parameter as input file (e.g. jpeg, png, ...)
    std::string filename = argv[1];

    // Load image into memory (force image format to RGBA)
    int img_w, img_h, img_c;
    uint8_t *pixels = stbi_load(filename.c_str(), &img_w, &img_h, &img_c, STBI_rgb);*/
    int img_w = 32; 
    int img_h = 32;
    uint8_t *pixels = new uint8_t[img_w * img_h * 3];
    for (int i =0; i<img_w * img_h;i++)
    {
        pixels[3*i] = 255;
        pixels[3*i+1] = 0;
        pixels[3*i+2] = 0;
    } 
    uint8_t *compressed = new uint8_t[((int)ceil((double)img_w*(double)img_h/256.0))*256*253]; //how many palette will be made, *256 (each indicies)*253(max number of colors) 
    ConvertToPalette(pixels, img_w, img_h, compressed);   
    for(int i=0; i<(((int)ceil((double)img_w*(double)img_h/256.0))); i++)
    {
        //printf("compressed [i]%u\n", compressed[i]);
    }

	/*uint8_t *decodedRLE = new uint8_t[3*img_w*img_h]; 
	Decode(decodedRLE, RLE, size);
	SavePPM("decoded.ppm", img_w, img_h, pixels);*/

    return 0;
}

// Convert an RGB pixel buffer to a DXT1 pixel buffer
void ConvertToPalette(uint8_t *pixels, int width, int height, uint8_t *compressed)
{
    uint8_t i;
    uint8_t j;
   	uint8_t block[768];
    uint32_t offset;
    uint8_t resultSize;
    int p_index = 0;
   
    // Loop over each set of 4 rows (starting at top of image)
    for (j = 0; j < height; j += 16)
    {
        
        // Loop over each set of 4 columns (starting at left of image)
        for (i = 0; i < width; i+=16)
        {
            Palette p;
            p.colors = new uint8_t[256*3];
            p.size =0;
            p.indicies = new uint8_t[16*16];
            uint8_t *result = new uint8_t[1+(16*16) + 256];

            offset = (j * width * 3) + (i * 3);
            ExtractBlock(offset, pixels, width, block);
            GetColors(block, p.indicies, p.colors, p.size);
            resultSize = Indicies( p.size, p.indicies, result, p.colors);
            for(int k=0; k< resultSize; k++)
            {
                compressed[p_index +k] = result[k];
            }
           
            p_index = p_index + resultSize;
            //printf("%d %d\n", p_index, p.size);
	     } 
	}
}

// Copy 16x16 block of RGB pixels to `out_block`
void ExtractBlock(uint32_t offset, uint8_t *pixels, int width, uint8_t out_block[768])
{
    int i, j;
    uint32_t start = offset;
    
    for (j = 0; j < 16; j++)
    {
        for (i = 0; i < 48; i++)
        {
            out_block[j * 48 + i] = pixels[start + i];
        }
        start += width * 3;
    }
}

void GetColors(uint8_t block[768], uint8_t *indicies, uint8_t *colors, uint8_t &size)
{

	for (int i = 0; i < 256; i++) //in 16*16 blcok find unique colors
    {
        //printf("here  %u,%u,%u\n", block[i * 3], block[i * 3 + 1], block[i * 3 + 2]);
        if(!isInside(colors, block[i * 3], block[i * 3 + 1], block[i * 3 + 2] , size , i, indicies))
		{
			fflush(stdout);
            //if unique add to unique colors and to pallete
			colors[size ] = block[i * 3];
			colors[size  +1] = block[i * 3 + 1];
			colors[size  +2] = block[i * 3 + 2];
			
			//add that color to pallete index by number
			indicies[i] = size ;
			if(size == 253)//more than 16 colors
			{
				printf("Error cannot be converted to Palette");
				exit(1);
			}
			size  = size  + 3; 
		}
        //printf("old color ");
    } //if unique add to unique colors and to pallete
}

//search through key of colors (palette), and make pixel equal to color it matches.... if it doesnt return false
bool isInside(uint8_t *colors, uint8_t red, uint8_t green, uint8_t blue, uint8_t num_colors, uint8_t pixel_num, uint8_t *indicies)
{
	bool result;
	result = false;

	for(int i =0; i< num_colors; i+=3)
	{
		if( colors[i]== red && colors[i+1] == green && colors[i+2]==blue)
		{
			indicies[pixel_num] = i;
			return true;
		}
	}
	return result;
}

uint8_t Indicies(uint8_t size, uint8_t *indices, uint8_t *result, uint8_t *colors)
{   
    uint8_t bits_percolor;
    uint8_t resultSize;
    if ((size/3) <=2)
    {
        bits_percolor = 1;
    }
    else if((size/3) <=4)
    {
        bits_percolor =2;
    }
    else if((size/3) <=16)
    {
        bits_percolor = 4;
    }
    else
    {
        bits_percolor = 8;
    }
   // printf("size: %d, bpc: %u\n", size, bits_percolor);
    int n_bytes = (int)(16.0*16.0 * ((double)bits_percolor/8.0));
    uint8_t *img = new uint8_t [n_bytes];
	//make array of colors with correct amount of bits??  or packed value
    int i,j;
    for(i = 0; i < n_bytes; i++)
    {
        for(j=0; j< (8/bits_percolor); j++)
       {

            //result |= (indices[i] << (j << bits_percolor));
            img[i] = (indices[i] << (j << bits_percolor));
        }
    }
    //add //full result - bits per color, unique colors (palette) , compressed indicies of each color(img).
    result[0] = bits_percolor;
    int k;
    for (k=1; k< (size+1); k++)
    {

        result[k] = colors[k-1];
        //printf("%d\n", result[k]);
    }
    for (int l=0; l< 256; l++)
    {
        result[k+l] = img[l];
        printf("%d\n", result[k+l]);

        resultSize = k+l;
    }
    //printf("At result");
    return resultSize;

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
	uint16_t count;
	for(int i=0; i<size; i= i+4)
	{
		//one count for RGB = 3 total
		count = (RLE[i] + 1) * 3;
		for(int j=1; j<count; j++)
		{
			//decoded increments while RLE loops through either 1,2,3 RGB
			decodedRLE[i+j] = RLE[i+(j%4)];
		}

	}

}


