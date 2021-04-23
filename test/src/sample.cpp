#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
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
void readDepth(const char *filename, int width, int height, float **depth);
void savePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void savePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void saveDds(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
uint32_t rgbaDepthToActivePixelCPU(int width, int height, uint8_t* rgba, float* depth, uint8_t* result);
uint64_t currentTime();

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
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
        float *depth;
        readPpm("resrc/nuclear_station.ppm", &img_w, &img_h, &rgba);
        readDepth("resrc/nuclear_station.depth", img_w, img_h, &depth);

        // Create OpenGL texture and upload rgba image
        GLuint tex_color;
        glGenTextures(1, &tex_color);
        glBindTexture(GL_TEXTURE_2D, tex_color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Create OpenGL texture and upload depth image
        GLuint tex_depth;
        glGenTextures(1, &tex_depth);
        glBindTexture(GL_TEXTURE_2D, tex_depth);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, img_w, img_h, 0, GL_RED, GL_FLOAT, depth);
        glBindTexture(GL_TEXTURE_2D, 0);

        glFinish();

        // Register images and get descriptions
        PariCGResourceDescription description_color;
        PariCGResourceDescription description_depth;
        PariCGResource resource_color = pariRegisterImage(tex_color, &description_color);
        PariCGResource resource_depth = pariRegisterImage(tex_depth, &description_depth);

        // Allocate output images (CPU and GPU)
        uint8_t *gray = new uint8_t[img_w * img_h];
        uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
        uint8_t* active_pixels = new uint8_t[img_w * img_h * 8 + 8];
        uint8_t* active_pixels_cpu = new uint8_t[img_w * img_h * 8 + 8];
        PariGpuBuffer gray_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Grayscale);
        PariGpuBuffer dxt1_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Dxt1);
        PariGpuBuffer activepixel_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::ActivePixel); 

        // Convert rgba to grayscale, dxt1, and active pixel
        uint32_t ap_size;
        pariGetRgbaTextureAsGrayscale(resource_color, description_color, tex_color, gray_gpu_buffer, img_w, img_h, gray);
        pariGetRgbaTextureAsDxt1(resource_color, description_color, tex_color, dxt1_gpu_buffer, img_w, img_h, dxt1);
        pariGetRgbaDepthTextureAsActivePixel(resource_color, description_color, tex_color, resource_depth, description_depth,
                                             tex_depth, activepixel_gpu_buffer, img_w, img_h, active_pixels, &ap_size);

        // Save grayscale result as pgm file
        savePgm("pari_result_cg_gray.pgm", img_w, img_h, gray);
        // Save dxt1 result as dds file
        saveDds("pari_result_cg_dxt1.dds", img_w, img_h, dxt1);

        // Print stats from active pixel result
        uint32_t num_inactive = 0;
        uint32_t num_active = 0;
        for (int i = 0; i < img_w * img_h; i++)
        {
            if (depth[i] == 1.0f) num_inactive++;
            else num_active++;
        }
        printf("Active Pixel compression:\n");
        printf(" active: %u, inactive: %u\n", num_active, num_inactive);
        printf(" compressed size: %u bytes:\n", ap_size);
        printf(" compression: %.3lf\n", 100.0 * (double)ap_size / (double)(img_w * img_h * 8));

        uint32_t ap_size_cpu = rgbaDepthToActivePixelCPU(img_w, img_h, rgba, depth, active_pixels_cpu);
        if (ap_size == ap_size_cpu)
        {
            int i;
            bool all_same = true;
            for (i = 0; i < ap_size; i++)
            {
                if (active_pixels[i] != active_pixels_cpu[i])
                {
                    printf(" GPU / CPU different data at position %d\n", i);
                    all_same = false;
                }
            }
            if (all_same)
            {
                printf(" all GPU / CPU result in same data!\n");
            }
        }
        else
        {
            printf(" GPU/CPU compressed sizes do not match\n");
        }
    }
    else
    {
        pariSetGpuDevice(0);

        // Read input rgba image (ppm file) and allocate space on GPU
        int img_w, img_h;
        uint8_t *rgba;
        float *depth;
        readPpm("resrc/nuclear_station.ppm", &img_w, &img_h, &rgba);
        readDepth("resrc/nuclear_station.depth", img_w, img_h, &depth);
        PariGpuBuffer rgba_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Rgba);
        PariGpuBuffer depth_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Depth);
        
        // Allocate output images (CPU and GPU)
        uint8_t *gray = new uint8_t[img_w * img_h];
        uint8_t *dxt1 = new uint8_t[img_w * img_h / 2];
        uint8_t* active_pixels = new uint8_t[img_w * img_h * 8 + 8];
        uint8_t* active_pixels_cpu = new uint8_t[img_w * img_h * 8 + 8];
        PariGpuBuffer gray_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Grayscale);
        PariGpuBuffer dxt1_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::Dxt1);
        PariGpuBuffer activepixel_gpu_buffer = pariAllocateGpuBuffer(img_w, img_h, PariCompressionType::ActivePixel); 

        // Convert rgba to grayscale, dxt1, and active pixel
        uint32_t ap_size;
        pariRgbaBufferToGrayscale(rgba, img_w, img_h, rgba_gpu_buffer, gray_gpu_buffer, gray);
        pariRgbaBufferToDxt1(rgba, img_w, img_h, rgba_gpu_buffer, dxt1_gpu_buffer, dxt1);
        pariRgbaDepthBufferToActivePixel(rgba, depth, img_w, img_h, rgba_gpu_buffer, depth_gpu_buffer,
                                         activepixel_gpu_buffer, active_pixels, &ap_size);

        // Save grayscale result as pgm file
        savePgm("pari_result_gray.pgm", img_w, img_h, gray);
        // Save dxt1 result as dds file
        saveDds("pari_result_dxt1.dds", img_w, img_h, dxt1);
        
        // Print stats from active pixel result
        uint32_t num_inactive = 0;
        uint32_t num_active = 0;
        for (int i = 0; i < img_w * img_h; i++)
        {
            if (depth[i] == 1.0f) num_inactive++;
            else num_active++;
        }
        printf("Active Pixel compression:\n");
        printf(" active: %u, inactive: %u\n", num_active, num_inactive);
        printf(" compressed size: %u bytes:\n", ap_size);
        printf(" compression: %.3lf\n", 100.0 * (double)ap_size / (double)(img_w * img_h * 8));

        uint32_t ap_size_cpu = rgbaDepthToActivePixelCPU(img_w, img_h, rgba, depth, active_pixels_cpu);
        if (ap_size == ap_size_cpu)
        {
            int i;
            bool all_same = true;
            for (i = 0; i < ap_size; i++)
            {
                if (active_pixels[i] != active_pixels_cpu[i])
                {
                    printf(" GPU / CPU different data at position %d\n", i);
                    all_same = false;
                }
            }
            if (all_same)
            {
                printf(" all GPU / CPU result in same data!\n");
            }
        }
        else
        {
            printf(" GPU/CPU compressed sizes do not match\n");
        }
        /*
        Active Pixel Header Info:
          - 7 (4-byte) ints
            - ICET_IMAGE_MAGIC_NUM_INDEX              0   ==>  ICET_IMAGE_MAGIC_NUM
            - ICET_IMAGE_COLOR_FORMAT_INDEX           1   ==>  ICET_IMAGE_COLOR_RGBA_UBYTE
            - ICET_IMAGE_DEPTH_FORMAT_INDEX           2   ==>  ICET_IMAGE_DEPTH_FLOAT
            - ICET_IMAGE_WIDTH_INDEX                  3   ==>  image width
            - ICET_IMAGE_HEIGHT_INDEX                 4   ==>  image height
            - ICET_IMAGE_MAX_NUM_PIXELS_INDEX         5   ==>  image width * image height
            - ICET_IMAGE_ACTUAL_BUFFER_SIZE_INDEX     6   ==>  header size + data size (i.e. (ICET_IMAGE_DATA_START_INDEX * sieof(IceTUInt)) +
                                                                                        (width * height * (pixel color size + pixel depth size)))
            - ICET_IMAGE_DATA_START_INDEX             7   ==>  actual compressed image data
        */
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

void readDepth(const char *filename, int width, int height, float **depth)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        char *tmp_char = new char[width * height * sizeof(float)];
        file.read(tmp_char, width * height * sizeof(float));
        *depth = reinterpret_cast<float*>(tmp_char);
    }
    else
    {
        *depth = NULL;
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

/****************************************************************************************/
uint32_t rgbaDepthToActivePixelCPU(int width, int height, uint8_t* rgba, float* depth, uint8_t* result)
{
    uint64_t start = currentTime();

    int i;
    uint32_t inactive = 0;
    uint32_t active = 0;
    uint32_t run_start_index = 0;
    float max_depth = 1.0f;
    bool active_run = false;
    for (i = 0; i < width * height; i++)
    {
        // end of existing active run, start of new inactive run
        if (depth[i] == max_depth && active_run)
        {
            active_run = false;
            memcpy(result + run_start_index, &inactive, 4);
            memcpy(result + run_start_index + 4, &active, 4);
            run_start_index = run_start_index + (8 * active) + 8;
            inactive = 1;
            active = 0;
        }
        // end of existing inactive run, start of new active run
        else if (depth[i] != max_depth && !active_run)
        {
            active_run = true;
            memcpy(result + run_start_index + (8 * active) + 8, rgba + 4 * i, 4);
            memcpy(result + run_start_index + (8 * active) + 12, depth + i, 4);
            active += 1;
        }
        // continuation of existing inactive run
        else if (depth[i] == max_depth && !active_run)
        {
            inactive += 1;
        }
        // continuation of existing active run
        else if (depth[i] != max_depth && active_run)
        {
            memcpy(result + run_start_index + (8 * active) + 8, rgba + 4 * i, 4);
            memcpy(result + run_start_index + (8 * active) + 12, depth + i, 4);
            active = active + 1;
        }
    }
    memcpy(result + run_start_index, &inactive, 4);
    memcpy(result + run_start_index + 4, &active, 4);

    uint64_t end = currentTime();
    printf("rgbaDepthToActivePixelCPU (%dx%d): %.6lf\n", width, height, (double)(end - start) / 1000000.0);

    return run_start_index + 8 * active + 8;
}

uint64_t currentTime()
{
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
    return us;
}