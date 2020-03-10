// small utility functions from the framework
#include <core/init_opencl.h>
#include <core/utils.h>

// OpenC types
#include <CL/cl.h> //command_queues, devices, ...

// OpenCV  headers
#include <opencv2/core/core.hpp> //types
#include <opencv2/imgcodecs.hpp> //imread, imwrite, color formats
#include <opencv2/highgui/highgui.hpp> //gui elements, window

// boost for simple program options
#include <boost/program_options.hpp> 

// standard header
#include <iostream>
#include <string>
#include <vector>

//time
cl_event event;
cl_ulong start,end;

/* global variables for storing input and output images on the CPU */
cv::Mat input_image;
cv::Mat output_image;



// buffer
cl_mem cl_input_buffer = 0;
cl_mem cl_output_buffer = 0;
cl_mem cl_mask_buffer = 0;


// global variables for all resources related to OpenCL (GPU)
cl_context context = 0;
cl_device_id device_id = 0;
cl_command_queue command_queue = 0;

cl_program program = 0; 

cl_kernel kernel = 0;

std::vector<float> gauss_mask;

int maskSize = 0;

void createFilter(float sigma, int& maskSizePointer) {
    maskSize = std::ceil(2.8284271f * sigma); // inaccurate because of the roundig up
    std::cout << "maskSize:"<< maskSize << std::endl;
    int mask_length = (maskSize*2+1)*(maskSize*2+1); //calculates the length of the 2D-matrix into 1D
    std::cout << "mask_length:"<<mask_length << std::endl;
    
    gauss_mask.resize(mask_length); //creates a 1D mask 
    
    float sum = 0.0f;
    
    for(int a = -maskSize; a < maskSize+1; a++) {
      for(int b = -maskSize; b < maskSize+1; b++) {
        float temp = 1/(sqrt(2*M_PI)*sigma) * exp(-((float)(a*a+b*b)/(2*sigma*sigma))); //gaussion weights
        sum += temp;
        gauss_mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
      }  
    }
    
    //normalization
    for (int i = 0; i < mask_length; ++i) {
      //std::cout << "gauss_mask an der Stelle "<< i <<" = "<< gauss_mask[i] << std::endl;
      gauss_mask[i] = gauss_mask[i] / sum;
    }

    maskSizePointer = maskSize;
}



void initialize_cl_buffers(int image_width, int image_height, unsigned char* rgb_data_im_1);
void compile_program_and_kernel(std::string const& kernel_file_path, std::string const& main_kernel_name);
void prepare_and_run_window_based_image_binarization_kernel(int width, int height);
void cleanup_cl_resources();


int main(int argc, char** argv) {

    namespace po = boost::program_options; 
    po::options_description desc("Options");

    desc.add_options() 
      ("help,h", "Print help messages") //bool
      ("image,i", po::value<std::string>(), "Path to image 1") //std::string
      ("no-gui,n", "No GUI") //bool
    ; //end of argument list


    po::variables_map vm; 

  //  po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw 

    if ( vm.count("help")  ) { 
      std::cout << desc << std::endl; 
      return 0; 
    } 

	std::string image_path = "./images/Flowerpots/view1.png"; 

    if ( !vm["image"].empty()) { 
      image_path = vm["image"].as<std::string>();
    } else {
        std::cout << "Image path was not specified. Using default image:" << std::endl;
        std::cout << image_path << std::endl;
        std::cout << std::endl;

        std::cout << "Execute: '" << argv[0] << " -h' fore more information about the arguments" << std::endl;
        std::cout << std::endl;
    }


    bool show_gui = vm["no-gui"].empty();
 
    //load images
    input_image = cv::imread(image_path.c_str(), cv::IMREAD_COLOR);

    int width = input_image.cols;
    int height = input_image.rows;

    int window_size_x = width;
    int window_size_y = height;

    // 3x8 unsigned character per pixel. this line actually allocates memory for our default constructed
    // output image (a.k.a. cv::Mat). We ask it to store 3 channels of unsigned char data per pixel (CV_8UC3)
    output_image = cv::Mat(height, width, CV_8UC3);

    // framework call to initialize CL environment
    dsm::initialize_cl_environment(context, device_id, command_queue);


    createFilter(1.0f, maskSize);

    initialize_cl_buffers(width, height, input_image.data);

    /* This is where we store our kernel code. 
       The entire folder kernels/ in [...]/framework/kernels is copied to install/bin, such that all our
       kernel paths usually start with "./kernels/" */
    std::string const kernel_file_path = "./kernels/simple_task_kernels/gaussian_blur.cl";
    std::string const& main_kernel_name = "gaussian_blur";
    
    // helper function to compile kernel from source (see definition below main)
    compile_program_and_kernel(kernel_file_path, main_kernel_name);

  

    std::string const& window_name{"Monitor Window"};
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::resizeWindow(window_name, window_size_x, window_size_y);


    do {

        prepare_and_run_window_based_image_binarization_kernel(width, height);

        //time
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        std::cout << "time =" << (end - start) / 1000000.0 << std::endl;


        cv::Mat concatenated_result;
        // put both input images and the output image next to each other and display them 
        cv::hconcat(input_image, output_image, concatenated_result);

        cv::imshow(window_name, concatenated_result); 
        int key = cv::waitKey(1) & 0xFF;

        //escape
        if(27 == key) {
            break;
        }
    } while(show_gui);

    std::string const& executable_name = dsm::get_filename_from_path(argv[0]);

    std::string const& output_image_path = "./result_" + executable_name + ".png";

    //write out result image as png-file
    cv::imwrite(output_image_path.c_str(), output_image);

    std::cout << std::endl;
    std::cout << "Wrote output image to: " << output_image_path << std::endl;


    cleanup_cl_resources();

    if (device_id != NULL)
    {
        free(device_id);
        device_id = NULL;
    }

	return 0;
}




// DEFINITIONS OF CL-FUNCTIONS
void initialize_cl_buffers(int image_width, int image_height, unsigned char* rgb_data_im) {
  int num_channels_input_output_image = 3; //RGB, 8 bit per channel for all images

  cl_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels_input_output_image * sizeof(char), (void *)rgb_data_im, NULL);
  cl_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        image_width * image_height * num_channels_input_output_image * sizeof(char), NULL, NULL);

  cl_mask_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1), gauss_mask.data(), NULL);
}

void compile_program_and_kernel(std::string const& kernel_file_path, std::string const& main_kernel_name) {


    // we will store the ASCII kernel code in the variable source_string
    std::string source_string{""};
    //use our helper function to load the kernel code from the file into our storage variable
    dsm::load_kernel_from_file(kernel_file_path.c_str(), source_string);

    // the c_function clCreateProgramWithSource expects a c-style string
    char const* source = source_string.c_str();

    size_t source_buffer_size = source_string.size();

    program = clCreateProgramWithSource(context, 1, &source, &source_buffer_size, NULL);

    cl_int status_code = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // check whether building the program was successfull
    if (CL_SUCCESS != status_code) {
        //if it did not work, check what was going wrong (~what errors we had in our cl_kernel_code)
        char *log;
        size_t size;

        // query size of log message from cl
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        // allocate space for log message (c-style)
        log = (char *)malloc(size+1);
        if (log) {

            // query actual log message
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
            size, log, NULL);
            // null termination
            log[size] = '\0';
            printf("%s", log);
            free(log);
        }
        exit(-1);
    }
    
    // create a kernel (-> tell opencl how the main kernel of the compiled program is called; here: compute_dufference_image)
    kernel = clCreateKernel(program, main_kernel_name.c_str(), NULL);
}

void prepare_and_run_window_based_image_binarization_kernel(int width, int height) {

    // prepare kernel arguments before we launch kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_input_buffer); // set input buffers
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_output_buffer);  // set output buffer
    clSetKernelArg(kernel, 2, sizeof(int), (void*) &width); /* add width of image as kernel argument 
                                                                (is automatically copied from CPU to GPU, because it is a built-in data type)*/
    clSetKernelArg(kernel, 3, sizeof(int), (void*) &height);

    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &cl_mask_buffer); 
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&maskSize);
    


    //prepare the n-dimensional range (here: 2D, because it fits conceptually well to 2D images)
    size_t global_work_size[2] = {size_t(width), size_t(height)};

    /*submit the execution of our precompiled kernel into the command queue and tell it about the dimensionality (2) 
      and the actual global work size (total number of compute elements [e.g. threads] we will launch in this kernel)*/
    
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                           global_work_size, NULL, 0, NULL, &event);

    // calculate the number of byte in the output image (=width * height * 8bit-RGB); could be also precomputed
    std::size_t num_byte_to_read_result = width * height * 3 * sizeof(unsigned char);

    /* submit a read call from cl_output_buffer (GPU) to output_image.data (CPU) into the working queue. 
       because this request is submitted after clEnqueueNDRangeKernel in the same queue, this ensures that
       the kernel execution finishes before clEnqueueReadBuffer begins. After this call, we can immediately
       access the new result written to output_image.data, because clEnqueueReadBuffer, because the call is blocking
       (-> what does blocking mean?)*/
    int status_code = clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                                     num_byte_to_read_result, output_image.data, 0, NULL, NULL);

    /* Check, whether some cl-call along the way failed (not necessarily the clEnqueueReadBuffer call)*/
    if(CL_SUCCESS != status_code) {
        std::cout << "Encountered a cl-Error in the execution of submitted commandQueue operations." << "\n";
        std::cout << "Error Code: " << status_code << "\n";
    }
}

// release all cl resources and check for errors
void cleanup_cl_resources() {

    cl_int status_code = 0;
    // Release the CL resources
    status_code = clReleaseKernel(kernel);
    if(CL_SUCCESS != status_code) { //Release kernel and check for errors
        std::cout << "Failed to release kernel!\n";
    } 
    
    status_code = clReleaseProgram(program);
    if(CL_SUCCESS != status_code) {
      std::cout << "Failed to release program!\n";
    }  //Release the program object.
    
    status_code = clReleaseMemObject(cl_input_buffer);
    if(CL_SUCCESS != status_code) {
      std::cout << "Failed to release GPU input buffer 1\n";
    }  //Release mem object.
    
    status_code = clReleaseMemObject(cl_output_buffer);
    if(CL_SUCCESS != status_code) {
      std::cout << "Failed to release \n";
    }
    
    status_code = clReleaseCommandQueue(command_queue);
    if(CL_SUCCESS != status_code) {
      std::cout << "Failed to release \n";
    } //Release  Command queue.
    
    status_code = clReleaseContext(context);
    if(CL_SUCCESS != status_code) {
      std::cout << "Failed to release \n";
    } //Release context.
}