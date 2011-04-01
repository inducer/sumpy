#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE (0x100000)

const int   N       = 32;
const int   THREADS = 8;
const float EPS2    = 1e-4;

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

int main()
{
// Allocate memory on host and device
  float4 *sourceHost = (float4*)malloc( N*sizeof(float4) );
  float  *targetHost = (float *)malloc( N*sizeof(float ) );
// Initialize
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = rand() / (1. + RAND_MAX);
    sourceHost[i].y = rand() / (1. + RAND_MAX);
    sourceHost[i].z = rand() / (1. + RAND_MAX);
    sourceHost[i].w = 1.0 / N;
  }

// Load the source code containing the kernel
  char fileName[] = "./nbody.cl";
  FILE *fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
  size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

// Get platform/device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  clGetPlatformIDs(1, &platform_id, NULL);
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

// Create OpenCL context
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

// Create Command Queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

// Create Memory Buffer
  cl_mem sourceDevc = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float4), NULL, NULL);
  cl_mem targetDevc = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float),  NULL, NULL);

// Copy input data to the memory buffer
  clEnqueueWriteBuffer(command_queue, sourceDevc, CL_TRUE, 0, N*sizeof(float4), sourceHost, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, targetDevc, CL_TRUE, 0, N*sizeof(float),  targetHost, 0, NULL, NULL);

// Create Kernel Program from the source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                 (const size_t *)&source_size, NULL);

// Build Kernel Program
  clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

// Create OpenCL Kernel
  cl_kernel kernel = clCreateKernel(program, "nbody", NULL);

// Set OpenCL Kernel Arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sourceDevc);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&targetDevc);

// Execute OpenCL Kernel
  size_t dimension = N;
  size_t blockSize = THREADS;
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &dimension, &blockSize, 0, NULL, NULL);

// Copy results from the memory buffer
  clEnqueueReadBuffer(command_queue, targetDevc, CL_TRUE, 0, N*sizeof(float), targetHost, 0, NULL, NULL);

// Direct summation on host
  float dx, dy, dz, r;
  for( int i=0; i<N; i++ ) {
    float p = - sourceHost[i].w / sqrtf(EPS2);
    for( int j=0; j<N; j++ ) {
      dx = sourceHost[i].x - sourceHost[j].x;
      dy = sourceHost[i].y - sourceHost[j].y;
      dz = sourceHost[i].z - sourceHost[j].z;
      r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
      p += sourceHost[j].w / r;
    }
    printf("%d %f %f\n",i,p,targetHost[i]);
  }

  /* Finalization */
  clFlush(command_queue);
  clFinish(command_queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(sourceDevc);
  clReleaseMemObject(targetDevc);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  free(source_str);
  free(sourceHost);
  free(targetHost);

  return 0;
}
