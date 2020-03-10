/*
  This is a simple and thoroughly commented app which runs a simple OpenCL kernel in order to compute
  the difference between two images and stores it into an output images.
  This version of the image difference computation app cl image_2D objects as internal cl objects 
  (compare to compute_image_differences_using_buffers.cpp 
              kernels/simple_example_kernels/image_differences_using_image_2d.cl)
*/

#include <iostream>
#include <string>

#include <core/init_opencl.h>
#include <core/utils.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// include plain OpenCL c headers 
#include <CL/cl.h>

#include <vector>

int main(int argc, char** argv) {


	std::string const image_0_path = "./images/Flowerpots/view1.png"; 
	std::string const image_1_path = "./images/Flowerpots/view5.png"; 

    //load image
    int width, height, num_channels;
    
    //force stbi to pad images with an alpha byte for 4 byte alignment. we have to force 4 channels, otherwise stbi
    // messes up the alignment. Note: OpenCV decodes and loads well-known image formats flawlessly and without any problems
    //rgbx rgbx rgbx ...
    unsigned char *rgb_data_image_1 = stbi_load(image_0_path.c_str(), &width, &height, &num_channels, 4);
    unsigned char *rgb_data_image_2 = stbi_load(image_1_path.c_str(), &width, &height, &num_channels, 4);

    //^ cpu buffer for our iamges


    std::string const kernel_filename ="./kernels/simple_example_kernels/image_difference_kernel_using_image_2d.cl";
    cl_context context;
    cl_device_id device_id;
    cl_command_queue command_queue;
    dsm::initialize_cl_environment(context, device_id, command_queue);

    //Step 5: Create program object 
    std::string sourceStr;
    cl_int status = dsm::load_kernel_from_file(kernel_filename.c_str(), sourceStr);
    if(status != CL_SUCCESS){
        std::cout << "Failed to parse kernel source file named: " 
                  << kernel_filename << "\n";
        return 1;
    }
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source) };
    
    if(CL_SUCCESS != status ) {
    	std::cout << "Could not retrieve clContext. Exiting\n";
    	return -1;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);


    if(CL_SUCCESS != status ) {
    	std::cout << "Could not retrieve clDevice. Exiting\n";
    	return -1;
    }

    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // check whether building the program was successfull
    if (CL_SUCCESS != status ) {
        char *log;
        size_t size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        // 1. Laenge des Logbuches?
        log = (char *)malloc(size+1);
        if (log) {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
            size, log, NULL);
            // 2. Hole das Logbuch ab
            log[size] = '\0';
            printf("%s", log);
            free(log);
        }
        return 1;
    }


    size_t read_origin[3] = {0, 0, 0};
    size_t read_dimensions[3] = {size_t(width), size_t(height), 1};

    size_t write_origin[3] = {0, 0, 0};
    size_t write_dimensions[3] = {size_t(width), size_t(height), 1};

    const cl_image_format imageFormat = { CL_RGBA, CL_UNSIGNED_INT8 };
    cl_int error_status = 0;

    // creates GPU resource for containing our second image2D
    cl_mem input_image_buffer_1 = clCreateImage2D(context, CL_MEM_READ_ONLY  , &imageFormat,
                                          width, height,  0, NULL, &error_status);
    
    // separate and explicit call in order to fill our first empty image 2D from our cpu buffer (rgb_data_image_1)
    status = clEnqueueWriteImage (command_queue , 
                                 input_image_buffer_1,
                                 CL_TRUE,
                                 write_origin,
                                 write_dimensions,
                                 0,
                                 0,
                                 rgb_data_image_1,
                                 0,
                                 NULL,
                                 NULL);

    // creates GPU resource for containing our second image2D
    cl_mem input_image_buffer_2 = clCreateImage2D(context, CL_MEM_READ_ONLY, &imageFormat,
                                          width, height,  0, NULL, &error_status);
    
    // separate and explicit call in order to fill our second empty image 2D from our cpu buffer (rgb_data_image_1)
    status = clEnqueueWriteImage (command_queue , 
                                 input_image_buffer_2,
                                 CL_TRUE,
                                 write_origin,
                                 write_dimensions,
                                 0,
                                 0,
                                 rgb_data_image_2,
                                 0,
                                 NULL,
                                 NULL);

    // create a cl_mem object which internally is represemted as a 2D image
    cl_mem outputBuffer = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &imageFormat,
                                          width, height, 0, NULL, &error_status);

    //compile the kernel
    cl_kernel kernel = clCreateKernel(program, "compute_difference_image", NULL);

    // Set Kernel arguments for the subsequent call
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_image_buffer_1);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input_image_buffer_2);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputBuffer);
    status = clSetKernelArg(kernel, 3, sizeof(int), (void *) &width);


    size_t global_work_size[2] = {size_t(width), size_t(height)};

    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                    global_work_size, NULL, 0, NULL, NULL);


    unsigned char rgb_data_result[width*height*4];

    // Enqueue Read Operation GPU Image to CPU Buffer resources.
    status = clEnqueueReadImage (command_queue , 
                                 outputBuffer,
                                 CL_TRUE,
                                 read_origin,
                                 read_dimensions,
                                 0,
                                 0,
                                 rgb_data_result,
                                 0,
                                 NULL,
                                 NULL);

    // Clean the GPU resources.
    status = clReleaseKernel(kernel); //Release kernel.
    status = clReleaseProgram(program); //Release the program object.
    status = clReleaseMemObject(input_image_buffer_1); //Release mem object.
    status = clReleaseMemObject(input_image_buffer_2); //Release mem object.
    status = clReleaseMemObject(outputBuffer);
    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL)
    {
        free(device_id);
        device_id = NULL;
    }


    //convert 4 channel image to 3 channel image for final visualization
    unsigned char rgb_data_result_3_channels[width*height*3];
    for(int pixel_idx = 0; pixel_idx < width*height; ++pixel_idx){
        int pixel_offset_to_r_4_channels = 4 * pixel_idx;
        int pixel_offset_to_r_3_channels = 3 * pixel_idx;

        for(int channel_idx = 0; channel_idx < 3; ++channel_idx) {
            rgb_data_result_3_channels[pixel_offset_to_r_3_channels + channel_idx] 
                = rgb_data_result[pixel_offset_to_r_4_channels + channel_idx];
        }
    }
 
    stbi_write_bmp("./image_difference_flowers.bmp", width, height, num_channels, rgb_data_result_3_channels);
    

    //dealloc dynamic memory
    stbi_image_free(rgb_data_image_1);
    stbi_image_free(rgb_data_image_2);

	return 0;
}