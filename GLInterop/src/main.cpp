#include <iostream>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "readpx.h" 

void ReadPpm(const char *filename, uint32_t *width, uint32_t *height, uint8_t **pixels);
void SavePpm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);
void SavePgm(const char *filename, uint32_t width, uint32_t height, uint8_t *pixels);

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

    // Read image and upload as OpenGL texture
    uint32_t img_w, img_h;
    uint8_t *rgba_in;
    GLuint texture;
    ReadPpm("resrc/UST_test.ppm", &img_w, &img_h, &rgba_in);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_in);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Allocate output image buffers
    uint8_t *rgb_out = new uint8_t[img_w * img_h * 3];
    uint8_t *gray_out = new uint8_t[img_w * img_h];
    uint8_t *gpu_gray;
    AllocateGpuOutput((void**)&gpu_gray, img_w * img_h);

    // Convert on GPU then copy to CPU
    struct cudaGraphicsResource *resource;
    BindCudaResourceToTexture(&resource, texture);
    ReadRgbaTextureAsRgb(texture, rgb_out);
    ReadRgbaTextureAsGrayscale(&resource, texture, img_w, img_h, gpu_gray, gray_out);

    // Write output images to disk
    SavePpm("out_rgb.ppm", img_w, img_h, rgb_out);
    SavePgm("out_gray.pgm", img_w, img_h, gray_out);

    // Clean up
    glfwPollEvents();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void ReadPpm(const char *filename, uint32_t *width, uint32_t *height, uint8_t **pixels)
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

