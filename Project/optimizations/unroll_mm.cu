#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
// __constant__ float MASK[5120];


__global__ void conv_forward_kernel(const float *mask, float *input, float *output,
  const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{


    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int tz = threadIdx.z;
    int bz = blockIdx.z;

    int row = by * blockDim.x + ty;
    int col = bx * blockDim.x + tx;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    float Pvalue = 0.0f;

    #define mask_2d(i1, i0) mask[(i1) * (Channel * K * K) + (i0)]
    #define in_3d(i2, i1, i0) input[(i2) * (Channel * K * K * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
    #define out_3d(i2, i1, i0) output[(i2) * (Map_out * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]

    for(int q = 0; q < (ceil((1.0 * Channel * K * K) / TILE_WIDTH)); q++){
      if(row < Map_out && (q * TILE_WIDTH + tx) < Channel * K * K){
        subTileM[ty][tx] = mask_2d(row, q * TILE_WIDTH + tx);
      }else{
        subTileM[ty][tx] = 0.0f;
      }

      if(col < Height_out * Width_out && (q * TILE_WIDTH + ty < Channel * K * K)){
        subTileN[ty][tx] = in_3d(bz, q * TILE_WIDTH + ty, col);
      }else{
        subTileN[ty][tx] = 0.0f;
      }

      __syncthreads();


      for(int i = 0; i < TILE_WIDTH; i++){
        Pvalue += subTileM[ty][i] * subTileN[i][tx];
      }

      __syncthreads();



      }

    if(row < Map_out && col < Height_out * Width_out){
      out_3d(bz, row, col) = Pvalue;
    }



    #undef mask_2d
    #undef in_3d
    #undef out_3d

}

__global__ void input_unroll1( const float *input, float *output, const int Map_out, const int Channel, const int Height, const int Width, const int K){


    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;


    #define out_3d(i2, i1, i0) output[(i2) * (Channel * Height_out * Width_out * K * K) + (i1) * (Height_out * Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bz = blockIdx.z;

    int W_unroll = Height_out * Width_out;
    int H_unroll = Channel * K * K;
    if(idx < Channel * W_unroll){
      int c = idx / W_unroll;
      int s = idx % W_unroll;
      int h_out = s / Width_out;
      int w_out = s % Width_out;
      int h_unroll = h_out * Width_out + w_out;
      int w_base = c * K * K;
      for(int i = 0; i < K; i++){
        for(int j = 0; j < K; j++){
          int w_unroll = w_base + i * K + j;
          out_3d(bz, w_unroll, h_unroll) = in_4d(bz, c, h_out + i, w_out + j);

        }
      }
    }

    #undef out_3d
    #undef in_4d

}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{


    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(MASK, host_mask, Map_out * Channel * K * K * sizeof(float));


}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    float *unrolled;
    int size_per_loop = 100;
    cudaMalloc((void **) &unrolled, Batch * Height_out * Width_out * Channel * K * K * sizeof(float));


    dim3 dimBlockUnroll(TILE_WIDTH, 1, 1);
    dim3 dimGridUnroll(ceil(1.0 * Width_out * Height_out * Channel / TILE_WIDTH), 1, Batch);

    // dim3 dimGrid(ceil(1.0 * Height_out * Width_out / TILE_WIDTH), ceil(1.0 * Map_out / TILE_WIDTH), Batch);
    dim3 dimGrid(ceil(1.0 * Height_out * Width_out / TILE_WIDTH), ceil(1.0 * Map_out / TILE_WIDTH), size_per_loop);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    input_unroll1<<<dimGridUnroll, dimBlockUnroll>>>(device_input, unrolled, Map_out, Channel, Height, Width, K);

    for(int i = 0; i < Batch; i += size_per_loop){
      // offset is i * size of 1 batch
      conv_forward_kernel<<<dimGrid, dimBlock>>>(device_mask, unrolled + i * Height_out * Width_out * Channel * K * K, device_output + i * Map_out * Height_out * Width_out, Batch, Map_out, Channel, Height, Width, K);
    }

    cudaFree(unrolled);


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
