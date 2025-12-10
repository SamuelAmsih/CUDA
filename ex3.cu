#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <iostream>
#include <math.h>


__global__ void GrayscaleKernel(uchar4* image, uchar4* grayimage, int width, int height)
{
    // calculate the rows # of image and grayimage element 
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // calculate the columns # og image and grayimage element
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // each thread computes one element of grayimage if in range
    if ( (row < height ) && (col < width) ) 
    {

        uchar4 rgb = image[row * width + col];

        // formula for gray scale 
        // gray = 0.299*R + 0.587*G + 0.114*B
        uchar4 gray = make_uchar4(  0.299f * rgb.x + 0.487f * rgb.y + 0.114f * rgb.z,
                                    0.299f * rgb.x + 0.487f * rgb.y + 0.114f * rgb.z, 
                                    0.299f * rgb.x + 0.487f * rgb.y + 0.114f * rgb.z,
                                    1.0f);        


        grayimage[row * width + col] = gray;
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
    int n = 720;  // height 
    int m = 1280; // width
    
    // allocate and copy output image to device 
    cudaMallocManaged(&grayimage, (n * m) * sizeof(uchar4));

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
        dim3 DimGrid( (n - 1)/16 + 1, (m - 1)/16 + 1, 1 );
        dim3 DimBlock( 16, 16, 1 ); 

        GrayscaleKernel<<<DimGrid, DimBlock>>>( image, grayimage, m, n );

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

    cudaFree(grayimage);
}