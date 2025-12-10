#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>

#define NUM_BINS 256

__global__ void minKernel(const uchar4* grayimage, unsigned int* d_min, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    int stride = blockDim.x * gridDim.x;

    unsigned int localMin = 255;

    for (int i = idx; i < size; i += stride)
    {
        unsigned int pixel = grayimage[i].x;

        if (pixel < localMin)
            localMin = pixel;
    }

    atomicMin(d_min, localMin);
}

__global__ void sumKernel(const uchar4* grayimage, unsigned long long* d_sum, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    int stride = blockDim.x * gridDim.x;

    unsigned long long localSum = 0;

    for (int i = idx; i < size; i += stride)
    {
        unsigned int pixel = grayimage[i].x;
        localSum += pixel;
    }

    // atomic accumulate into global memory
    atomicAdd(d_sum, localSum);
}


__global__ void maxKernel(const uchar4* grayimage, unsigned int* d_max, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    int stride = blockDim.x * gridDim.x;

    unsigned int localMax = 0;

    for (int i = idx; i < size; i += stride)
    {
        unsigned int pixel = grayimage[i].x;

        if (pixel > localMax)
            localMax = pixel;
    }

    // global reduction using atomic
    atomicMax(d_max, localMax);
}


__global__ void GrayscaleKernel(uchar4* image, uchar4* grayimage, int width, int height)
{
    // calculate the rows # of image and grayimage element 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // calculate the columns # og image and grayimage element
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // each thread computes one element of grayimage if in range
    if ( (row < height ) && (col < width) ) 
    {

        int idx = row * width + col;
        uchar4 rgb = image[idx];

        // formula for gray scale 
        // gray = 0.299*R + 0.587*G + 0.114*B
        uint8_t gray = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;


        rgb.x = gray;
        rgb.y = gray;
        rgb.z = gray;

        grayimage[idx] = rgb;
    }
}

__global__ void calcHistogramKernel(uchar4* grayimage,  int *histo, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int size = width * height;

    int stride = blockDim.x * gridDim.x;

    __shared__  int histo_private[NUM_BINS];

    // histogram init
    if( threadIdx.x < NUM_BINS)
    {
        histo_private[threadIdx.x] = 0;
    }
    __syncthreads();

    //build the histogram
    for( int i = idx; i < size; i += stride ) 
    {
        unsigned char pixel = grayimage[i].x;

        atomicAdd(&histo_private[pixel], 1);
    }
    __syncthreads();
    if( threadIdx.x < NUM_BINS)
    {
        atomicAdd(&histo[threadIdx.x], histo_private[threadIdx.x]);
    }
    
}

__global__ void plotHistogramKernel(uchar4* image, int* histogram, int width, int height, int max_freq)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    uchar4 white_pixel = make_uchar4(255, 255, 255, 255);
    uchar4 black_pixel = make_uchar4(0, 0, 0, 255);
    if (index < 256)
    {
        int freq = histogram[index] * height / max_freq;
        for (int i = 0; i < 256; i++)
        {
            int row = height - i - 1;
            if (i <= freq)
            {
                image[row * width + 2*index] = white_pixel;
                image[row * width + 2*index+1] = white_pixel;
            }
            else
            {
                image[row * width + 2*index].x/=2;
                image[row * width + 2*index].y/=2;
                image[row * width + 2*index].z/=2;
                
                image[row * width + 2*index+1].x/=2;
                image[row * width + 2*index+1].y/=2; 
                image[row * width + 2*index+1].z/=2;

            }
        }
    }
}

int main( int argc, char** argv )
{
    // create input/output streams
    videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput* output = videoOutput::Create(argc, argv, ARG_POSITION(1));

    uchar4* image = NULL; // can be uchar3, uchar4, float3, float4
    uchar4* grayimage = NULL;
    int* histogram = NULL;
    int n = 720;  // image height 
    int m = 1280; // image width

    int max_freq = 20000;

    unsigned int h_min, h_max;
    unsigned long long h_sum;
    unsigned int* d_min;
    unsigned int* d_max;
    unsigned long long* d_sum;

    // allocate and copy output image to device 
    cudaMallocManaged(&d_min, sizeof(unsigned int));
    cudaMallocManaged(&d_max, sizeof(unsigned int));
    cudaMallocManaged(&d_sum, sizeof(unsigned long long));

    cudaMallocManaged(&grayimage, (n * m) * sizeof(uchar4));
    cudaMallocManaged(&histogram, (256) * sizeof(int));

    if ( !input )
    return 0;
    // capture/display loop
    while (true)
    {
        int status = 0; // see videoSource::Status

        if ( !input->Capture(&image, 1000, &status) ) // 1000ms timeout (default)
        {
            if (status == videoSource::TIMEOUT)
                continue;
            break; // EOS
        }

        // launch kernel 
        dim3 DimGrid( (m - 1)/16 + 1, (n - 1)/16 + 1, 1 );
        dim3 DimBlock( 16, 16, 1 ); 
        int threads = 256;
        int blocks  = ( m * n + threads - 1 ) / threads;

        
        cudaMemset(histogram, 0, 256 * sizeof( int));

        GrayscaleKernel<<<DimGrid/*blocks*/, DimBlock/*threads*/>>>( image, grayimage, m, n );
        calcHistogramKernel<<<blocks, threads>>>( grayimage, histogram, m, n);

        

        cudaMemset(d_min, 255, sizeof(unsigned int));
        cudaMemset(d_max, 0, sizeof(unsigned int));
        cudaMemset(d_sum, 0, sizeof(unsigned long long));

        minKernel<<<blocks, threads>>>(grayimage, d_min, m, n);
        maxKernel<<<blocks, threads>>>(grayimage, d_max, m, n);
        sumKernel<<<blocks, threads>>>(grayimage, d_sum, m, n);

        plotHistogramKernel<<<1, 256>>>( grayimage, histogram, m, n, max_freq );

        cudaDeviceSynchronize();

    

        cudaMemcpy(&h_min, d_min, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        float mean = float(h_sum) / float(m * n);


        cudaDeviceSynchronize();

        printf("Min: %u, Max: %u, Mean: %.2f\n", h_min, h_max, mean);


        if ( output != NULL )
        {
            output->Render(grayimage, input->GetWidth(), input->GetHeight());
            
            // Update status bar
            char graystr[256];
            
            sprintf(graystr, "Camera Viewer (%ux%u) | %0.1f FPS | %0.1f Mean", input->GetWidth(),
            input->GetHeight(), output->GetFrameRate(), mean);
            output->SetStatus(graystr);

            if (!output->IsStreaming()) // check if the user quit
                break;
        }
    }

    cudaFree(histogram);
    cudaFree(grayimage);
}