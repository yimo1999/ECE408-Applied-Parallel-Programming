// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan1(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  int stride = 1;
  int t = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x + t;
  int second = start + blockDim.x;

  T[t] = 0.0f;
  T[t + blockDim.x] = 0.0f;
  if(start < len){
    T[t] = input[start];
  }

  if(second < len){
    T[t + blockDim.x] = input[second];
  }

  while(stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0){
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE){
      T[index + stride] += T[index];
    }

    stride /= 2;
  }

  __syncthreads();
  if(start < len){
    output[start] = T[t];
  }

  if(second < len){
    output[second] = T[t + blockDim.x];
  }

}

__global__ void scan2(float *input, float *output, int len) {
  __shared__ float T[2 * BLOCK_SIZE];
  int stride = 1;
  int t = threadIdx.x;
  int start = (1 + t) * 2 * blockDim.x - 1;
  int second = start + 2 * blockDim.x * BLOCK_SIZE;

  T[t] = 0.0f;
  T[t + blockDim.x] = 0.0f;

  if(start < len){
    T[t] = input[start];
  }

  if(second < len){
    T[t + blockDim.x] = input[second];
  }

  while(stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0){
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE){
      T[index + stride] += T[index];
    }

    stride /= 2;
  }

  __syncthreads();
  if(start < len){
    output[2 * blockIdx.x * blockDim.x + t] = T[t];
  }

  if(second < len){
    output[2 * blockIdx.x * blockDim.x + t + blockDim.x] = T[t + blockDim.x];
  }
}

__global__ void add(float *input1, float *input2, float *output, int len) {
  __shared__ float sum;
  int t = threadIdx.x;
  int b = blockIdx.x;
  int start = 2 * blockIdx.x * blockDim.x + t;
  if(!t){
    if(!b){
      sum = 0;
    }else{
      sum = input2[b - 1];
    }
  }

  __syncthreads();

  if(start < len){
    output[start] = input1[start] + sum;
  }
  if(start + blockDim.x < len){
    output[start + blockDim.x] = input1[start + blockDim.x] + sum;
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGridScan1(ceil((1.0 * numElements) / BLOCK_SIZE * 2), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGridScan2(1, 1, 1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *deviceOutputTemp;
  float *sumOutput;
  cudaMalloc((void **)&deviceOutputTemp, numElements * sizeof(float));
  cudaMalloc((void **)&sumOutput, 2 * BLOCK_SIZE * sizeof(float));
  scan1<<<dimGridScan1, dimBlock>>>(deviceInput, deviceOutputTemp, numElements);
  cudaDeviceSynchronize();
  scan2<<<dimGridScan2, dimBlock>>>(deviceOutputTemp, sumOutput, numElements);
  cudaDeviceSynchronize();
  add<<<dimGridScan1, dimBlock>>>(deviceOutputTemp, sumOutput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  cudaFree(deviceOutputTemp);
  cudaFree(sumOutput);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
