#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ float MASK[5120];
// tiled shared memory
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // (TILE_WIDTH + K - 1) = 22
    __shared__ float sm[22][22];
    int sm_size = TILE_WIDTH + K - 1;
    int w_size = ceil(1.0 * Width_out / TILE_WIDTH);

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define Cmask_4d(i3, i2, i1, i0) MASK[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]




    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int h = by / w_size * TILE_WIDTH + threadIdx.x;
    int w = by % w_size * TILE_WIDTH + threadIdx.y;

    float sum = 0.0f;

    for(int c = 0; c < Channel; c++){

        for(int i = h; i < by / w_size * TILE_WIDTH + sm_size; i += TILE_WIDTH){
            for(int j = w; j < by % w_size * TILE_WIDTH + sm_size; j += TILE_WIDTH){
                if(i < Height && j < Width){
                    sm[i - by / w_size * TILE_WIDTH][j - by % w_size * TILE_WIDTH] = in_4d(bz, c, i, j);
                }else{
                  sm[i - by / w_size * TILE_WIDTH][j - by % w_size * TILE_WIDTH] = 0.0f;
                }
            }
        }

        __syncthreads();


        // for(int c = 0; c < Channel; ++c){
        if(h < Height_out && w < Width_out){
            for(int k1 = 0; k1 < K; k1++){
                for(int k2 = 0; k2 < K; k2++){
                    if(threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH){
                      sum += sm[threadIdx.x + k1][threadIdx.y + k2] * Cmask_4d(bx, c, k1, k2);
                    }

                  }

                }
                // __syncthreads();
            }
        // }
      __syncthreads();
    }
    // __syncthreads();
    if(h < Height_out && w < Width_out){
      out_4d(bz, bx, h, w) = sum;
    }



}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{


    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    // cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, host_mask, Map_out * Channel * K * K * sizeof(float));


}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    dim3 dimGrid(Map_out, ceil(1.0 * Height_out / TILE_WIDTH) * ceil(1.0 * Width_out / TILE_WIDTH), Batch);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
