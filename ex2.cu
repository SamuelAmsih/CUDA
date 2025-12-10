#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>

#define NUM_BINS 256

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


__global__ void calcHistogramKernel(uchar4* image,  int *histo, int width, int height)
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
    while( idx < size ) 
    {
        uchar4 rgb = image[idx];

        char pixel = (rgb.x + rgb.y + rgb.z) / 3;
       
        atomicAdd(&histo_private[pixel], 1);

        idx += stride;
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
        int freq = histogram[index] * 256 / max_freq;
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
    videoOutput* grayoutput = videoOutput::Create(argc, argv, ARG_POSITION(1));

    uchar4* image = NULL; // can be uchar3, uchar4, float3, float4
    uchar4* grayimage = NULL;
    int* histogram = NULL;
    int n = 720;  // image height 
    int m = 1280; // image width
    
    int max_freq = 20000;


    // allocate and copy output image to device 
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
        
        cudaMemset(histogram, 0, 256 * sizeof( int));

        GrayscaleKernel<<<DimGrid/*blocks*/, DimBlock/*threads*/>>>( image, grayimage, m, n );
        calcHistogramKernel<<<1, 256>>>( grayimage, histogram, m, n);
        plotHistogramKernel<<<1, 256>>>( grayimage, histogram, m, n, max_freq );


        cudaDeviceSynchronize();
        
        if ( output != NULL )
        {
            output->Render(image, input->GetWidth(), input->GetHeight());
        
            // Update status bar
            char str[256];

            sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(),
            input->GetHeight(), output->GetFrameRate());
            output->SetStatus(str);

            if (!output->IsStreaming()) // check if the user quit
                break;
        }


        if ( grayoutput != NULL )
        {
            grayoutput->Render(grayimage, input->GetWidth(), input->GetHeight());
            
            // Update status bar
            char graystr[256];
            
            sprintf(graystr, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(),
            input->GetHeight(), grayoutput->GetFrameRate());
            grayoutput->SetStatus(graystr);

            if (!output->IsStreaming()) // check if the user quit
                break;
        }
    }
    cudaFree(histogram);
    cudaFree(grayimage);
}