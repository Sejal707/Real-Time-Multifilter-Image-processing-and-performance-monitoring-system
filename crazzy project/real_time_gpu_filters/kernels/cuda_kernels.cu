#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// Gaussian Blur Kernel
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels, float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_kernel = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                    int pixel_idx = (ny * width + nx) * channels + c;
                    
                    float weight = kernel[kernel_idx];
                    sum += input[pixel_idx] * weight;
                    weight_sum += weight;
                }
            }
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)(sum / weight_sum);
    }
}

// Sobel Edge Detection Kernel
__global__ void sobel_edge_kernel(unsigned char* input, unsigned char* output, 
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;
    
    // Sobel X and Y kernels
    int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int c = 0; c < channels; c++) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                int pixel_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                
                gx += input[pixel_idx] * sobel_x[kernel_idx];
                gy += input[pixel_idx] * sobel_y[kernel_idx];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)fminf(255.0f, magnitude);
    }
}

// Color Temperature Kernel
__global__ void color_temperature_kernel(unsigned char* input, unsigned char* output, 
                                       int width, int height, float temperature) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Temperature adjustment (simplified)
    float r = input[idx + 2];     // BGR format
    float g = input[idx + 1];
    float b = input[idx];
    
    if (temperature > 0) {
        // Warm (increase red, decrease blue)
        r = fminf(255.0f, r * (1.0f + temperature * 0.01f));
        b = fmaxf(0.0f, b * (1.0f - temperature * 0.01f));
    } else {
        // Cool (decrease red, increase blue)
        r = fmaxf(0.0f, r * (1.0f + temperature * 0.01f));
        b = fminf(255.0f, b * (1.0f - temperature * 0.01f));
    }
    
    output[idx] = (unsigned char)b;
    output[idx + 1] = (unsigned char)g;
    output[idx + 2] = (unsigned char)r;
}

// Bilateral Filter Kernel
__global__ void bilateral_filter_kernel(unsigned char* input, unsigned char* output,
                                      int width, int height, int channels,
                                      int radius, float sigma_color, float sigma_space) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        int center_idx = (y * width + x) * channels + c;
        float center_value = input[center_idx];
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int neighbor_idx = (ny * width + nx) * channels + c;
                    float neighbor_value = input[neighbor_idx];
                    
                    // Spatial weight
                    float spatial_dist = sqrtf(dx * dx + dy * dy);
                    float spatial_weight = expf(-(spatial_dist * spatial_dist) / (2.0f * sigma_space * sigma_space));
                    
                    // Color weight
                    float color_dist = fabsf(center_value - neighbor_value);
                    float color_weight = expf(-(color_dist * color_dist) / (2.0f * sigma_color * sigma_color));
                    
                    float weight = spatial_weight * color_weight;
                    sum += neighbor_value * weight;
                    weight_sum += weight;
                }
            }
        }
        
        output[center_idx] = (unsigned char)(sum / weight_sum);
    }
}

// Emboss Filter Kernel
__global__ void emboss_kernel(unsigned char* input, unsigned char* output, 
                            int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width - 1 || y >= height - 1 || x < 1 || y < 1) return;
    
    // Emboss kernel
    int emboss_kernel[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                int pixel_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                
                sum += input[pixel_idx] * emboss_kernel[kernel_idx];
            }
        }
        
        // Add 128 to center around gray and clamp
        int result = (int)(sum + 128.0f);
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, result));
    }
}

}
