// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//@@ insert code here
__global__ void convert_float_char(float *input, unsigned char *output, int width, int height, int channels, int flag){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int size = width * height * channels;
  if(flag == 0){
    if(i < size){
      output[i] = (unsigned char)(255.0 * input[i]);
    }
  }else{
    if(i < size){
      input[i] = (float) (1.0 * output[i] / 255.0);
    }
  }

}

__global__ void rgb2gray(unsigned char *input, unsigned char *output, int width, int height, int channels){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int size = width * height;
  if(i < size){
    float r = input[3 * i];
    float g = input[3 * i + 1];
    float b = input[3 * i + 2];
    output[i] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void gray2hist(unsigned char *input, unsigned int *output, int width, int height){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ unsigned int hist[HISTOGRAM_LENGTH];
  if(threadIdx.x < HISTOGRAM_LENGTH){
    hist[threadIdx.x] = 0.0f;
  }

  __syncthreads();


  // if(i < width * height){
  //   atomicAdd(&(hist[input[i]]), 1);
  // }

  while(i < width * height){
    atomicAdd(&(hist[(int) (input[i])]), 1);
    i += (blockDim.x * gridDim.x);
  }
  __syncthreads();

  if(threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd(&(output[threadIdx.x]), hist[threadIdx.x]);
  }

}

__global__ void hist2CDF(unsigned int *input, float *output, int width, int height){
  __shared__ float cdf[HISTOGRAM_LENGTH];


  int tx = threadIdx.x;
  // int bx = blockDim.x;
  int i = 1;
  cdf[tx] = 0.0f;
  cdf[128 + tx] = 0.0f;
  if(tx < HISTOGRAM_LENGTH){
    cdf[tx] = (float) (input[tx]);
  }

  if(128 + tx  < HISTOGRAM_LENGTH){
    cdf[128 + tx] = (float) (input[128 + tx]);
  }


  while(i < HISTOGRAM_LENGTH){
    __syncthreads();
    int idx = 2 * i * (tx + 1) - 1;
    if(idx < HISTOGRAM_LENGTH){
      cdf[idx] += cdf[idx - i];
    }
    i *= 2;
  }

  int j = 128;

  while(j > 0){
    __syncthreads();
    int idx = 2 * j * (tx + 1) - 1;
    if(idx + j < HISTOGRAM_LENGTH){
      cdf[idx + j] += cdf[idx];
    }

    j /= 2;
  }

  __syncthreads();

  int s = width * height;

  if(tx < HISTOGRAM_LENGTH){
    output[tx] = 1.0 * cdf[tx] / s;
  }

  if(128 + tx < HISTOGRAM_LENGTH){
    output[128 + tx] = 1.0 * cdf[128 + tx] / s;
  }
}

__global__ void equalization(unsigned char *input, float *cdf, unsigned char *output, int width, int height, int channels){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < width * height * channels){
    int idx = input[i];
    float temp1 = 1.0 * 255 * (cdf[idx] - cdf[0]) / (1.0 - cdf[0]);
    float temp2 = (min(max(temp1, 0.0), 255 * 1.0));
    output[i] = (unsigned char) (temp2);
  }
}




int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputFloat;
  float *deviceOutputFloat;
  unsigned char *deviceInputChar;
  unsigned char *deviceOutputChar;
  unsigned char *deviceInputGrayScale;
  unsigned int *deviceInputHist;
  float *deviceCDF;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **) &deviceInputFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceInputChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceOutputChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceInputGrayScale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &deviceInputHist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));

  int oneChannelSize = imageWidth * imageHeight;
  int threeChannelSize = imageWidth * imageHeight * imageChannels;

  cudaMemcpy(deviceInputFloat, hostInputImageData, threeChannelSize * sizeof(float), cudaMemcpyHostToDevice);


  dim3 dimGridFloat2Char(ceil(1.0 * threeChannelSize / BLOCK_SIZE), 1, 1);
  dim3 dimBlockFloat2Char(BLOCK_SIZE, 1, 1);
  convert_float_char<<<dimGridFloat2Char, dimBlockFloat2Char>>>(deviceInputFloat, deviceInputChar, imageWidth, imageHeight, imageChannels, 0);

  cudaDeviceSynchronize();

  dim3 dimGridRgb2Gray(ceil(1.0 * oneChannelSize / BLOCK_SIZE), 1, 1);
  dim3 dimBlockRgb2Gray(BLOCK_SIZE, 1, 1);
  rgb2gray<<<dimGridRgb2Gray, dimBlockRgb2Gray>>>(deviceInputChar, deviceInputGrayScale, imageWidth, imageHeight, imageChannels);

  cudaDeviceSynchronize();


  dim3 dimGridGray2Hist(ceil(1.0 * oneChannelSize / HISTOGRAM_LENGTH), 1, 1);
  dim3 dimBlockGray2Hist(HISTOGRAM_LENGTH, 1, 1);
  gray2hist<<<dimGridGray2Hist, dimBlockGray2Hist>>>(deviceInputGrayScale, deviceInputHist, imageWidth, imageHeight);

  cudaDeviceSynchronize();

  dim3 dimGridHist2CDF(1, 1, 1);
  dim3 dimBlockHist2CDF(128, 1, 1);
  hist2CDF<<<dimGridHist2CDF, dimBlockHist2CDF>>>(deviceInputHist, deviceCDF, imageWidth, imageHeight);

  cudaDeviceSynchronize();

  dim3 dimGridEqu(ceil(1.0 * threeChannelSize / BLOCK_SIZE), 1, 1);
  dim3 dimBlockEqu(BLOCK_SIZE, 1, 1);
  equalization<<<dimGridEqu, dimBlockEqu>>>(deviceInputChar, deviceCDF, deviceOutputChar, imageWidth, imageHeight, imageChannels);

  cudaDeviceSynchronize();


  convert_float_char<<<dimGridEqu, dimBlockEqu>>>(deviceOutputFloat, deviceOutputChar, imageWidth, imageHeight, imageChannels, 1);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  wbSolution(args, outputImage);

  //@@ insert code here

  cudaFree(deviceInputFloat);
  cudaFree(deviceOutputFloat);
  cudaFree(deviceInputChar);
  cudaFree(deviceOutputChar);
  cudaFree(deviceInputGrayScale);
  cudaFree(deviceInputHist);
  cudaFree(deviceCDF);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
