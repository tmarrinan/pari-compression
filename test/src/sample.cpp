#include <iostream>
#include <string>
#include <fstream>
#define GLFW_INCLUDE_GLEXT
#include <GLFW/glfw3.h>
#include "paricompress.h"

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


void createDdsHeader(int width, int height, DdsHeader *header);
void readPpm(const char *filename, int *width, int *height, uint8_t **pixels);
void savePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void savePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void saveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);

int main(int argc, char **argv)
{
    if (argc > 1 && std::string(argv[1]) == "opengl")
    {
        // Initialize GLFW
        if (!glfwInit())
        {
            fprintf(stderr, "Error: could not initialize GLFW\n");
            exit(EXIT_FAILURE);
        }

        // Create a window and its OpenGL context
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow *window = glfwCreateWindow(320, 180, "PARI-Compress Sample", NULL, NULL);

        // Make window's context current
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // Select GPU to perform compression on (use same as the one used for OpenGL rendering)
        pariSetGpuDevice(PARI_DEVICE_OPENGL);

        // Read input rgba image (ppm file)
        int img_w, img_h;
        uint8_t *rgba;
        readPpm("resrc/airplane_4k.ppm", &img_w, &img_h, &rgba);

        // Create OpenGL texture and upload rgba image
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        glBindTexture(GL_TEXTURE_2D, 0);

        glFinish();

        // Register image and get description
        PariCGResourceDescription description;
        PariCGResource resource = pariRegisterImage(texture, &description);

        // Allocate output images (CPU and GPU)
        uint8_t *gray = new uint8_t[img_w * img_h];
        uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
        PariGpuBuffer gray_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Grayscale);
        PariGpuBuffer dxt1_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Dxt1);

        // Convert rgba to grayscale and dxt1
        pariGetRgbaTextureAsGrayscale(resource, description, texture, gray_gpu_buffer, img_w, img_h, gray);
        pariGetRgbaTextureAsDxt1(resource, description, texture, dxt1_gpu_buffer, img_w, img_h, dxt1);

        // Save results as pgm and dds files
        savePgm("pari_result_cg_gray.pgm", img_w, img_h, gray);
        saveDds("pari_result_cg_dxt1.dds", img_w, img_h, dxt1);
    }
    else
    {
        pariSetGpuDevice(0);

        // Read input rgba image (ppm file) and allocate space on GPU
        int img_w, img_h;
        uint8_t *rgba;
        readPpm("resrc/airplane_4k.ppm", &img_w, &img_h, &rgba);
        PariGpuBuffer rgba_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Rgba);
        
        // Allocate output images (CPU and GPU)
        uint8_t *gray = new uint8_t[img_w * img_h];
        uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
        PariGpuBuffer gray_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Grayscale);
        PariGpuBuffer dxt1_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Dxt1);

        // Convert rgba to grayscale
        pariRgbaBufferToGrayscale(rgba, img_w, img_h, rgba_gpu_buffer, gray_gpu_buffer, gray);
        pariRgbaBufferToDxt1(rgba, img_w, img_h, rgba_gpu_buffer, dxt1_gpu_buffer, dxt1);

        // Save result as pgm file
        savePgm("pari_result_gray.pgm", img_w, img_h, gray);
        saveDds("pari_result_dxt1.dds", img_w, img_h, dxt1);
    }

    return 0;
}

void createDdsHeader(int width, int height, DdsHeader *header)
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

void readPpm(const char *filename, int *width, int *height, uint8_t **pixels)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        int i;
        int header_count = 0;
        std::string line;
        while (header_count < 3)
        {
            std::getline(file, line);
            if (line.length() > 0 && line[0] != '#')
            {
                if (header_count == 1)
                {
                    sscanf(line.c_str(), "%d %d", width, height);
                }
                header_count++;
            }
        }
        char *tmp_char = new char[(*width) * (*height) * 3];
        uint8_t *tmp = reinterpret_cast<uint8_t*>(tmp_char);
        *pixels = new uint8_t[(*width) * (*height) * 4];
        file.read(tmp_char, (*width) * (*height) * 3);
        for (i = 0; i < (*width) * (*height); i++)
        {
            (*pixels)[4 * i + 0] = tmp[3 * i + 0];
            (*pixels)[4 * i + 1] = tmp[3 * i + 1];
            (*pixels)[4 * i + 2] = tmp[3 * i + 2];
            (*pixels)[4 * i + 3] = 255;
        }
    }
    else
    {
        *pixels = NULL;
    }
}

void savePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%u %u\n255\n", width, height);
    fwrite(pixels, width * height * 3, 1, fp);
    fclose(fp);
}

void savePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%u %u\n255\n", width, height);
    fwrite(pixels, width * height, 1, fp);
    fclose(fp);
}

void saveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels)
{
    DdsHeader header;
    createDdsHeader(width, height, &header);
    
    FILE *fp = fopen(filename, "wb");
    fwrite(&header, sizeof(DdsHeader), 1, fp);
    fwrite(pixels, width * height / 2, 1, fp);
    fclose(fp);
}
