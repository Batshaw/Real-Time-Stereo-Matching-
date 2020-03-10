#include <iostream>
#include <string>

#include <core/init_opencl.h>
#include <core/utils.h>

#include <opencv2/core/core.hpp> //types
#include <opencv2/imgcodecs.hpp> //imread, imwrite, color formats
#include <opencv2/highgui/highgui.hpp> //gui elements, window

//#define STB_IMAGE_IMPLEMENTATION
//#include <stb/stb_image.h>

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include <stb/stb_image_write.h>

#include <CL/cl.h>

//global variables
//****************
//grayscale conversion kernel
cl_kernel grayscale_conversion_kernel_cq0 = NULL;
cl_kernel grayscale_conversion_kernel_cq1 = NULL;
//debug view float to uchar3 conversion
cl_kernel debug_view_float_to_uchar3_conversion_cq0 = NULL;
cl_kernel color_RGB_to_LAB_conversion = NULL;
//cl_kernel random_plane_initialization = NULL;
cl_kernel random_disparity_map_initialization = NULL;
cl_kernel SAD_based_patch_match_propagation = NULL;
cl_kernel patch_match_propagation = NULL;
cl_kernel SAD_based_disparity_map_computation = NULL;
cl_kernel adaptive_support_weights_computation = NULL;


cv::Mat input_image_1;
cv::Mat input_image_2;


void setup_kernels(cl_program const& program) {
 grayscale_conversion_kernel_cq0 = clCreateKernel(program, "convertImageUchar3ToGrayscaleFloatAndNormalize", NULL);
 grayscale_conversion_kernel_cq1 = clCreateKernel(program, "convertImageUchar3ToGrayscaleFloatAndNormalize", NULL);
 //random_plane_initialization = clCreateKernel(program, "planeInitialization", NULL);
 random_disparity_map_initialization = clCreateKernel(program, "randomDisparityMapInitialization", NULL);
 SAD_based_patch_match_propagation = clCreateKernel(program, "simplePatchMatchPropagation", NULL);
 patch_match_propagation = clCreateKernel(program, "patchMatchPropagation", NULL);
 color_RGB_to_LAB_conversion = clCreateKernel(program, "convert_rgb_images_to_lab", NULL);
 SAD_based_disparity_map_computation = clCreateKernel(program, "computeDisparityMap", NULL);
 adaptive_support_weights_computation = clCreateKernel(program, "computeASWbasedDisparityMap", NULL);
 debug_view_float_to_uchar3_conversion_cq0 = clCreateKernel(program, "convertImageFloatToUChar3", NULL);
}

// GPU buffer for RGB uchar3 image 0
cl_mem inputImageBufferA = NULL;
// GPU buffer for RGB uchar3 image 1
cl_mem inputImageBufferB = NULL;
// GPU buffer for GRAYSCALE float image 0
cl_mem grayscaleImageBufferA = NULL;
// GPU buffer for GRAYSCALE float image 1
cl_mem grayscaleImageBufferB = NULL;

// GPU buffer for GRAYSCALE float images
cl_mem intermediateDisparityImageBufferA = NULL;
cl_mem disparityImageBufferA = NULL;

// GPU buffer for GRAYSCALE float image 0
//cl_mem gradientImageBufferA = NULL;
// GPU buffer for GRAYSCALE float image 1
//cl_mem gradientImageBufferB = NULL;

cl_mem colorConvertedLabImageBufferA = NULL;
cl_mem colorConvertedLabImageBufferB = NULL;

// GPU buffer for RANDOM_PLANE_PER_PIXEL float
//cl_mem randomPlaneImageBufferA = NULL;
//cl_mem randomPlaneImageBufferB = NULL;

cl_mem randomDisparityMapBuffer = NULL;

cl_mem resultA = NULL;
cl_mem resultB = NULL;


//writing the final results back to rgb8 values
cl_mem disparityVisBufferA = NULL;
cl_mem disparityVisBufferB = NULL;

int const NUM_INPUT_CHANNELS = 3;
void setup_buffers(cl_context const& context, int image_width, int image_height, void* image_data_A, void* image_data_B) {
    unsigned size_rgb8_image = image_width * image_height * 3 * sizeof(char);
    unsigned size_r32f_image = image_width * image_height * sizeof(float);
    //gradient images: channel for x and y gradient
    //unsigned size_rg32f_image = image_width * image_height * 2 * sizeof(float);

    //CIELab images: L, a, b chanels
    unsigned size_lab32f_image = image_width * image_height * 3 * sizeof(float);

    //4-channel per pixel plane images: depth + normal vector
    unsigned size_rgbd32f = image_width * image_height * 3 * sizeof(float);

    inputImageBufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       size_rgb8_image, image_data_A, NULL);
    inputImageBufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       size_rgb8_image, image_data_B, NULL);

    grayscaleImageBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                           size_r32f_image, NULL, NULL);
    grayscaleImageBufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                           size_r32f_image, NULL, NULL);

    colorConvertedLabImageBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                   size_lab32f_image, NULL, NULL);
    colorConvertedLabImageBufferB = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                   size_lab32f_image, NULL, NULL);

    intermediateDisparityImageBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                           size_r32f_image, NULL, NULL);

    disparityImageBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                           size_r32f_image, NULL, NULL);

    //gradientImageBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, 
    //                                      size_rg32f_image, NULL, NULL);
    //gradientImageBufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, 
    //                                       size_rg32f_image, NULL, NULL);

    /*randomPlaneImageBufferA = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             size_rgbd32f, NULL, NULL);
    randomPlaneImageBufferB = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             size_rgbd32f, NULL, NULL);*/

    randomDisparityMapBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              size_r32f_image, NULL, NULL);

    disparityVisBufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                       size_rgb8_image, NULL, NULL);
    disparityVisBufferB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                         size_rgb8_image, NULL, NULL);
}

//Release mem object.
void release_buffers() {
    cl_int status = 0;
    status = clReleaseMemObject(inputImageBufferA); 
    status = clReleaseMemObject(inputImageBufferB); 

    status = clReleaseMemObject(grayscaleImageBufferA); 
    status = clReleaseMemObject(grayscaleImageBufferB);

    status = clReleaseMemObject(colorConvertedLabImageBufferA);
    status = clReleaseMemObject(colorConvertedLabImageBufferB);  

    status = clReleaseMemObject(disparityImageBufferA); 

   // status = clReleaseMemObject(randomPlaneImageBufferA);
   // status = clReleaseMemObject(randomPlaneImageBufferB);

    status = clReleaseMemObject(disparityVisBufferA);
    status = clReleaseMemObject(disparityVisBufferB);

    if(CL_SUCCESS != status) {
        std::cout << "Error during buffer deallocation\n";
    }
}

const bool SAD_DISPARITY_COMPUTATION = 0;
const bool ASW_DISPARITY_COMPUTATION = 0;
const bool PM_PROPAGATION = 1;

int main(int argc, char** argv) {

    if(argc < 5){
        std::cout << "USAGE: " << argv[0] << " <image_file_one.png> <image_file_two.png> <out_file.bmp> <NUM_PROPAGATION_ITERATIONS> [propDirChange]\n";
        std::cout << "where [propDirChange] is an OPTIONAL sting parameter which will set the propagation direction change to TRUE\n";
        return 1;
    }

	std::string const image_1_path = argv[1]; 
	std::string const image_2_path = argv[2];
    bool switchPropagationDirection = false;
    
    if(PM_PROPAGATION){
        if(argc == 6){
            switchPropagationDirection = true;
            std::cout << "The change of propagation direction was set to TRUE by you entering " << argv[5] <<std::endl;
        }else{
          std::cout << "Only LEFTWARDS and DOWNWARDS propagations are set! To change enter" << argv[0] 
                    << " <image_file_one.png> <image_file_two.png> <out_file.bmp>  <NUM_PROPAGATION_ITERATIONS> [propDirChange]\n";  
        }
    }

 
    //load image
    int width, height, num_channels;
   

    //rgb rgb rgb ...
    input_image_1 = cv::imread(image_1_path.c_str(), cv::IMREAD_COLOR);
    input_image_2 = cv::imread(image_2_path.c_str(), cv::IMREAD_COLOR);

    width = input_image_1.cols;
    height = input_image_1.rows;

    cv::Mat output_image = cv::Mat(height, width, CV_8UC3);

    num_channels = 3;
    //unsigned char *rgb_data_im_1 = stbi_load(image_1_path.c_str(), &width, &height, &num_channels, 0);
    //unsigned char *rgb_data_im_2 = stbi_load(image_2_path.c_str(), &width, &height, &num_channels, 0);

    const std::string kernel_filename = "./kernels/patch_match_stereo.cl";
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

    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    dsm::check_program_build_status(program, device_id);

    //initialize buffers
    setup_buffers(context, width, height , input_image_1.data, input_image_2.data);
    //cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
    //                           width * height * num_channels * sizeof(char), NULL, NULL);

    //create all the kernels from our program
    setup_kernels(program);

    unsigned int window_half_width = 15;
    const int NUM_PROPAGATION_ITERATIONS = std::atoi(argv[4]);

    //actual preparation and launch of  kernels 
    size_t global_work_size[2] = {size_t(width), size_t(height)};

   
    // pass 1: RGB to grayscale conversion of image 1
    // ******
    status = clSetKernelArg(grayscale_conversion_kernel_cq0, 0, sizeof(cl_mem), (void *)&inputImageBufferA);
    status = clSetKernelArg(grayscale_conversion_kernel_cq0, 1, sizeof(cl_mem), (void *)&grayscaleImageBufferA);
    status = clSetKernelArg(grayscale_conversion_kernel_cq0, 2, sizeof(int), (void *) &width);
    status = clSetKernelArg(grayscale_conversion_kernel_cq0, 3, sizeof(int), (void *) &height);

    status = clEnqueueNDRangeKernel(command_queue, grayscale_conversion_kernel_cq0, 2, NULL, 
                                    global_work_size, NULL, 0, NULL, NULL);


    // pass 2: RGB to grayscale conversion of image 2
    // ******
    status = clSetKernelArg(grayscale_conversion_kernel_cq1, 0, sizeof(cl_mem), (void *)&inputImageBufferB);
    status = clSetKernelArg(grayscale_conversion_kernel_cq1, 1, sizeof(cl_mem), (void *)&grayscaleImageBufferB);
    status = clSetKernelArg(grayscale_conversion_kernel_cq1, 2, sizeof(int), (void *) &width);
    status = clSetKernelArg(grayscale_conversion_kernel_cq1, 3, sizeof(int), (void *) &height);

    status = clEnqueueNDRangeKernel(command_queue, grayscale_conversion_kernel_cq1, 2, NULL, 
                                    global_work_size, NULL, 0, NULL, NULL);

        // pass 3: disparity map computation using SAD similarity measure
        // ******
    if(SAD_DISPARITY_COMPUTATION) {
        std::cout << "Compute SAD-based disparity map" << "\n";
        status = clSetKernelArg(SAD_based_disparity_map_computation, 0, sizeof(cl_mem), (void *)&grayscaleImageBufferA);
        status = clSetKernelArg(SAD_based_disparity_map_computation, 1, sizeof(cl_mem), (void *)&grayscaleImageBufferB);
        status = clSetKernelArg(SAD_based_disparity_map_computation, 2, sizeof(cl_mem), (void *)&disparityImageBufferA);
        status = clSetKernelArg(SAD_based_disparity_map_computation, 3, sizeof(int), (void *) &width);
        status = clSetKernelArg(SAD_based_disparity_map_computation, 4, sizeof(int), (void *) &height);
        status = clSetKernelArg(SAD_based_disparity_map_computation, 5, sizeof(int), (void *) &window_half_width);

        status = clEnqueueNDRangeKernel(command_queue, SAD_based_disparity_map_computation, 2, NULL, 
                                        global_work_size, NULL, 0, NULL, NULL); 

    }else {

             std::cout << "Compute AWS-based disparity map" << "\n";
            // pass 1: RGB to LAB conversion of image 1 and image 2
            // ******
            /*status = clSetKernelArg(color_RGB_to_LAB_conversion, 0, sizeof(cl_mem), (void *)&inputImageBufferA);
            status = clSetKernelArg(color_RGB_to_LAB_conversion, 1, sizeof(cl_mem), (void *)&inputImageBufferB);
            status = clSetKernelArg(color_RGB_to_LAB_conversion, 2, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferA);
            status = clSetKernelArg(color_RGB_to_LAB_conversion, 3, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferB);
            status = clSetKernelArg(color_RGB_to_LAB_conversion, 4, sizeof(int), (void *) &width);
            status = clSetKernelArg(color_RGB_to_LAB_conversion, 5, sizeof(int), (void *) &height);

            status = clEnqueueNDRangeKernel(commandQueue, color_RGB_to_LAB_conversion, 2, NULL, 
                                            global_work_size, NULL, 0, NULL, NULL);*/

           // std::cout << "ENQUEUE ERROR: " << status << "\n";


            // pass 2: disparity map computation using ASW similarity measure
            // ******
           // window_half_width = 5;
            if(ASW_DISPARITY_COMPUTATION) {
               /* status = clSetKernelArg(adaptive_support_weights_computation, 0, sizeof(cl_mem), (void *)&inputImageBufferA);
                status = clSetKernelArg(adaptive_support_weights_computation, 1, sizeof(cl_mem), (void *)&inputImageBufferB);
                status = clSetKernelArg(adaptive_support_weights_computation, 2, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferA);
                status = clSetKernelArg(adaptive_support_weights_computation, 3, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferB);
                status = clSetKernelArg(adaptive_support_weights_computation, 4, sizeof(cl_mem), (void *)&disparityImageBufferA);
                status = clSetKernelArg(adaptive_support_weights_computation, 5, sizeof(int), (void *) &width);
                status = clSetKernelArg(adaptive_support_weights_computation, 6, sizeof(int), (void *) &height);
                status = clSetKernelArg(adaptive_support_weights_computation, 7, sizeof(int), (void *) &window_half_width);

                status = clEnqueueNDRangeKernel(commandQueue, adaptive_support_weights_computation, 2, NULL, 
                                                global_work_size, NULL, 0, NULL, NULL);
                
                std::cout << "ENQUEUE ERROR: " << status << "\n";*/
            }else { //test branch for PM development; 
            
                //generate inital random disparity map image
                const unsigned int  max_disparity = 60;
                status = clSetKernelArg(random_disparity_map_initialization, 0, sizeof(int), (void *)&width);
                status = clSetKernelArg(random_disparity_map_initialization, 1, sizeof(int), (void *)&height);
                status = clSetKernelArg(random_disparity_map_initialization, 2, sizeof(int), (void *)&max_disparity);
                status = clSetKernelArg(random_disparity_map_initialization, 3, sizeof(int), (void *)&window_half_width);
                status = clSetKernelArg(random_disparity_map_initialization, 4, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);

                status = clEnqueueNDRangeKernel(command_queue, random_disparity_map_initialization, 2, NULL, 
                                                global_work_size, NULL, 0, NULL, NULL);
               // std::cout << "RAND DM INIT - ENQUEUE ERROR: " << status << "\n";
                const unsigned int LEFTWARDS = 0;
                const unsigned int DOWNWARDS = 1;
                const unsigned int RIGHTWARDS = 2;
                const unsigned int UPWARDS = 3;
                unsigned int propagationDirection = LEFTWARDS;
                for(int propagation_iteration_idx = 1; propagation_iteration_idx <= NUM_PROPAGATION_ITERATIONS; ++propagation_iteration_idx) {
                    if(propagation_iteration_idx % 2 != 0 || !switchPropagationDirection){                     
                        propagationDirection = LEFTWARDS; //left to right value propagation step in odd ittereations
                        std::cout << "Performing LEFTWARDS propagation \n";
                    }else if(switchPropagationDirection){
                        propagationDirection = RIGHTWARDS; //right to left value propagation step in even ittereations
                    }
                    /*status = clSetKernelArg(patch_match_propagation, 0, sizeof(cl_mem), (void *)&inputImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 1, sizeof(cl_mem), (void *)&inputImageBufferB);
                    status = clSetKernelArg(patch_match_propagation, 2, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 3, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferB);
                    status = clSetKernelArg(patch_match_propagation, 4, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);
                    status = clSetKernelArg(patch_match_propagation, 5, sizeof(cl_mem), (void *)&intermediateDisparityImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 6, sizeof(int), (void *) &width);
                    status = clSetKernelArg(patch_match_propagation, 7, sizeof(int), (void *)&height);
                    status = clSetKernelArg(patch_match_propagation, 8, sizeof(int), (void *)&window_half_width);
                    status = clSetKernelArg(patch_match_propagation, 9, sizeof(int), (void *)&propagationDirection);*/

                    status = clSetKernelArg(SAD_based_patch_match_propagation, 0, sizeof(cl_mem), (void *)&grayscaleImageBufferA);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 1, sizeof(cl_mem), (void *)&grayscaleImageBufferB);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 2, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 3, sizeof(cl_mem), (void *)&intermediateDisparityImageBufferA);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 4, sizeof(int), (void *)&width);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 5, sizeof(int), (void *)&height);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 6, sizeof(int), (void *)&window_half_width);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 7, sizeof(int), (void *)&propagationDirection);

                    size_t global_work_size_pm[1] = {size_t(height/* - 2 * window_half_width*/)};
                    status = clEnqueueNDRangeKernel(command_queue, SAD_based_patch_match_propagation, 1, NULL, 
                                                    global_work_size_pm, NULL, 0, NULL, NULL);

                    if(propagation_iteration_idx % 2 != 0 || !switchPropagationDirection){                     
                        propagationDirection = DOWNWARDS; //top to bottom value propagation step in odd ittereations
                        std::cout << "Performing DOWNWARDS propagation \n";
                    }else if(switchPropagationDirection){
                        propagationDirection = UPWARDS;  //bottom to top value propagation step in even ittereations
                    }
                    /*status = clSetKernelArg(patch_match_propagation, 0, sizeof(cl_mem), (void *)&inputImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 1, sizeof(cl_mem), (void *)&inputImageBufferB);
                    status = clSetKernelArg(patch_match_propagation, 2, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 3, sizeof(cl_mem), (void *)&colorConvertedLabImageBufferB);
                    status = clSetKernelArg(patch_match_propagation, 4, sizeof(cl_mem), (void *)&intermediateDisparityImageBufferA);
                    status = clSetKernelArg(patch_match_propagation, 5, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);
                    status = clSetKernelArg(patch_match_propagation, 6, sizeof(int), (void *) &width);
                    status = clSetKernelArg(patch_match_propagation, 7, sizeof(int), (void *)&height);
                    status = clSetKernelArg(patch_match_propagation, 8, sizeof(int), (void *)&window_half_width);
                    status = clSetKernelArg(patch_match_propagation, 9, sizeof(int), (void *)&propagationDirection);*/

                    status = clSetKernelArg(SAD_based_patch_match_propagation, 0, sizeof(cl_mem), (void *)&grayscaleImageBufferA);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 1, sizeof(cl_mem), (void *)&grayscaleImageBufferB);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 2, sizeof(cl_mem), (void *)&intermediateDisparityImageBufferA);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 3, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 4, sizeof(int), (void *)&width);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 5, sizeof(int), (void *)&height);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 6, sizeof(int), (void *)&window_half_width);
                    status = clSetKernelArg(SAD_based_patch_match_propagation, 7, sizeof(int), (void *)&propagationDirection);

                    global_work_size_pm[0] = size_t(width/*  - 2 * window_half_width*/);
                    status = clEnqueueNDRangeKernel(command_queue, SAD_based_patch_match_propagation, 1, NULL, 
                                                    global_work_size_pm, NULL, 0, NULL, NULL);
                    std::cout << "Propagation iteration number " << propagation_iteration_idx << " status: " << status << "\n";
                }
            }
        }


    // pass 4: debug view convert r32f back to rgb8 for image 1
    // ******
   /* if(SAD_DISPARITY_COMPUTATION){
        status = clSetKernelArg(debug_view_float_to_uchar3_conversion_cq0, 0, sizeof(cl_mem), (void *)&disparityImageBufferA);
    }else{*/
    status = clSetKernelArg(debug_view_float_to_uchar3_conversion_cq0, 0, sizeof(cl_mem), (void *)&randomDisparityMapBuffer);
   // }
    status = clSetKernelArg(debug_view_float_to_uchar3_conversion_cq0, 1, sizeof(cl_mem), (void *)&disparityVisBufferB);
    status = clSetKernelArg(debug_view_float_to_uchar3_conversion_cq0, 2, sizeof(int), (void *) &width);
    status = clSetKernelArg(debug_view_float_to_uchar3_conversion_cq0, 3, sizeof(int), (void *) &height);

    std::cout << "ENQUEUE ERROR float_to_uchar3_conversion: " << status << "\n";
    
    status = clEnqueueNDRangeKernel(command_queue, debug_view_float_to_uchar3_conversion_cq0, 2, NULL, 
                                    global_work_size, NULL, 0, NULL, NULL);
    
    std::cout << "ENQUEUE ERROR: " << status << "\n";

    /*
    //pass 5: random plane per-pixel initalization
    // ******
    const float max_disparity = 60.0;
    status = clSetKernelArg(random_plane_initialization, 0, sizeof(cl_mem), (void *)&randomPlaneImageBufferA);
    status = clSetKernelArg(random_plane_initialization, 1, sizeof(cl_mem), (void *)&disparityVisBufferA);
    status = clSetKernelArg(random_plane_initialization, 2, sizeof(int), (void *) &width);
    status = clSetKernelArg(random_plane_initialization, 3, sizeof(int), (void *) &height);
    status = clSetKernelArg(random_plane_initialization, 4, sizeof(float), (void *) &max_disparity);


    status = clEnqueueNDRangeKernel(commandQueue, random_plane_initialization, 2, NULL, 
                                    global_work_size, NULL, 0, NULL, NULL);
    
    std::cout << "ENQUEUE ERROR: " << status << "\n";

    */
    clFinish(command_queue);

    


    status = clEnqueueReadBuffer(command_queue, disparityVisBufferB, CL_TRUE, 0, 
                                 width * height * 3 * sizeof(char), output_image.data, 0, NULL, NULL);

    clFinish(command_queue);
    std::cout << "ENQUEUE ERROR: " << status << "\n";






    //Step 12: Clean the resources.
    status = clReleaseKernel(grayscale_conversion_kernel_cq0); //Release kernel.
    status = clReleaseKernel(grayscale_conversion_kernel_cq1);
    status = clReleaseKernel(debug_view_float_to_uchar3_conversion_cq0);

    status = clReleaseKernel(random_disparity_map_initialization);
    status = clReleaseKernel(patch_match_propagation);
    status = clReleaseKernel(color_RGB_to_LAB_conversion);
    status = clReleaseKernel(SAD_based_disparity_map_computation);
    status = clReleaseKernel(adaptive_support_weights_computation);

    release_buffers();

    status = clReleaseProgram(program); //Release the program object.

    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL)
    {
        free(device_id);
        device_id = NULL;
    }

    std::string num_iterations = std::to_string(NUM_PROPAGATION_ITERATIONS);
    std::string win_size = std::to_string(window_half_width);
    std::string prop_directions = "LD";
    if(switchPropagationDirection){
        prop_directions += "RD";
    }
    
    std::string const out_filename = std::string(argv[3]) + "_numItr" +
                                      num_iterations + "_ws" +
                                      win_size + "_" + prop_directions + ".png";


    cv::Mat display_mat;
    cv::hconcat(input_image_1, input_image_2, display_mat);
    cv::hconcat(display_mat, output_image, display_mat);
    cv::imshow("monitor", display_mat);
    cv::waitKey(0);

    cv::imwrite(out_filename, output_image);


	return 0;
}