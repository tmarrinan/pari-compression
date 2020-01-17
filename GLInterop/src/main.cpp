#include <iostream>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "imgread.h" 

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
void ReadPpm(const char *filename, uint32_t *width, uint32_t *height, uint8_t **pixels);
void SavePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void SavePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void SaveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);


int main(int argc, char **argv)
{
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Error: could not initialize GLFW\n");
        exit(1);
    }

    // Create a window and its OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(128, 64, "CUDA / OpenGL", NULL, NULL);

    // Make window's context current
    glfwMakeContextCurrent(window);
    
    // Disable monitor sync
    glfwSwapInterval(0);
    
    // Initialize Glad OpenGL extension handler
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        fprintf(stderr, "Error: could not initialize Glad\n");
        exit(1);
    }

    // Read image from file
    uint32_t img_w, img_h;
    uint8_t *rgba_in;
    //ReadPpm("resrc/UST_test.ppm", &img_w, &img_h, &rgba_in);
    ReadPpm("resrc/md_section_2048x1152.ppm", &img_w, &img_h, &rgba_in); 
    printf("Image size: %ux%u\n", img_w, img_h);

    // Upload image as OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_in);
    glBindTexture(GL_TEXTURE_2D, 0);

    // initialize image converter
    initImageConverter(img_w, img_h, texture);
    
    // Allocate output image buffers
    int dxt1_w, dxt1_h;
    uint32_t dxt1_size;
    getDxt1Dimensions(&dxt1_w, &dxt1_h, &dxt1_size);
    uint8_t *rgb_out = new uint8_t[img_w * img_h * 3];
    uint8_t *rgba_out = new uint8_t[img_w * img_h * 4];
    uint8_t *gray_out = new uint8_t[img_w * img_h];
    uint8_t *dxt1_out = new uint8_t[dxt1_size];

    // Convert on GPU then copy to CPU
    rgbaToGrayscale(NULL, gray_out);
    rgbaToDxt1(NULL, dxt1_out);

    /*
    // Convert on GPU then copy to CPU
    struct cudaGraphicsResource *resource;
    BindCudaResourceToTexture(&resource, texture);
    ReadRgbaTextureAsRgb(texture, rgb_out);
    ReadRgbaTextureAsRgba(texture, rgba_out);
    ReadRgbaTextureAsGrayscale(&resource, texture, img_w, img_h, gpu_gray, gray_out);
    ReadRgbaTextureAsDxt1(&resource, texture, img_w, img_h, gpu_dxt1, dxt1_out);
    */

    // Write output images to disk
    //SavePpm("out_rgb.ppm", img_w, img_h, rgb_out);
    SavePgm("out_gray.pgm", img_w, img_h, gray_out);
    SaveDds("out_dxt1.dds", img_w, img_h, dxt1_out);

    // Clean up
    glfwPollEvents();
    glfwDestroyWindow(window);
    glfwTerminate();

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

void ReadPpm(const char *filename, uint32_t *width, uint32_t *height, uint8_t **pixels)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: could not read %s\n", filename);
    }
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
                sscanf(line, "%u %u", width, height);
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
    delete[] tmp;
}

void SavePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%u %u\n255\n", width, height);
    fwrite(pixels, width * height * 3, 1, fp);
    fclose(fp);
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

