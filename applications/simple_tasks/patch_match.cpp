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


int main(int argc, char** argv) {

    uint32_t num_iterations = 0;

    if(argc < 2){
        std::cout << "USAGE: " << argv[0] << " <num_iterations> <left_image> <right_image> \n";
        return 1;
    }

    num_iterations = std::atoi(argv[1]);
    std::cout << "NUM ITERATIONS TO PERFORM " << num_iterations << std::endl;

  

    std::string const left_rgb_filename = argv[2];
    std::string const right_rgb_filename = argv[3];

  

   //default path
   	// std::string image_path_left = "./images/Teddy/im2.png"; 
	  // std::string image_path_right = "./images/Teddy/im6.png"; 

    // std::string image_path_left = "./images/Flowerpots/view1.png"; 
	  // std::string image_path_right = "./images/Flowerpots/view5.png";

   	//std::string image_path_left = "./images/RealWorldTestImages/left_real_image.png"; 
	  //std::string image_path_right = "./images/RealWorldTestImages/right_real_image.png"; 


    // std::string const left_rgb_filename = image_path_left;
    // std::string const right_rgb_filename = image_path_right;

    std::cout << "LEFT IMAGE FILENAME: " << left_rgb_filename << "\n";
    std::cout << "RIGHT IMAGE FILENAME: " << right_rgb_filename << "\n";
    
    //cv::Mat left_image_grayscale;
    //cv::Mat right_image_grayscale;

    cv::Mat left_image_grayscale = cv::imread(left_rgb_filename.c_str(), cv::IMREAD_GRAYSCALE);
    cv::Mat right_image_grayscale = cv::imread(right_rgb_filename.c_str(), cv::IMREAD_GRAYSCALE);


    int width = left_image_grayscale.cols;
    int height = left_image_grayscale.rows;


    cv::Mat output_image;
    output_image = cv::Mat(height, width, CV_32FC3); //CV_32FC1
    std::string const output_filename = "test_patch_match.png";
    std::cout << "OUTPUT FILENAME: " << output_filename << "\n";

    cl_context       context = 0;
    cl_device_id     device_id = 0;
    cl_command_queue command_queue = 0;
    cl_program       program = 0; 
    cl_kernel        kernel = 0;
    cl_kernel        kernel_plane = 0;
    cl_kernel        kernel_view = 0;
  
    cl_mem cl_left_input_image = 0;
    cl_mem cl_right_input_image = 0;
    cl_mem cl_plane_image_left_a = 0;
    cl_mem cl_plane_image_right_a = 0;
    cl_mem cl_plane_image_left_b = 0;
    cl_mem cl_plane_image_right_b = 0;
    cl_mem cl_plane_image_temp_left = 0;
    cl_mem cl_plane_image_temp_right = 0;
    cl_mem cl_output_buffer = 0;
    cl_mem cl_disparity_image_left = 0;
    cl_mem cl_disparity_image_right = 0;

    
    dsm::initialize_cl_environment(context, device_id, command_queue);  

    //image discriptor
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    //different image formats
    cl_image_format format_gray;
    format_gray.image_channel_order = CL_R;
    format_gray.image_channel_data_type = CL_UNORM_INT8;

    cl_image_format format_RGBA_32F;
    format_RGBA_32F.image_channel_order = CL_RGBA;
    format_RGBA_32F.image_channel_data_type = CL_FLOAT;

    
    cl_left_input_image = clCreateImage(context, CL_MEM_READ_ONLY, 
                                &format_gray, &desc,NULL, NULL);
    cl_right_input_image = clCreateImage(context, CL_MEM_READ_ONLY, 
                                &format_gray, &desc,NULL, NULL);

    cl_plane_image_left_a = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_right_a = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);                          

    cl_plane_image_left_b = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_right_b = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);                                

    cl_plane_image_temp_left = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_temp_right = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);                                

    cl_disparity_image_left = clCreateImage(context, CL_MEM_WRITE_ONLY, 
                                      &format_gray, &desc,NULL, NULL);


    cl_disparity_image_right = clCreateImage(context, CL_MEM_WRITE_ONLY, 
                                      &format_gray, &desc,NULL, NULL);

    size_t origin[3] = {0, 0, 0}; //offset within the image to copy from
    size_t region[3] = {size_t(width), size_t(height), size_t(1)}; //elements to per dimension

    int num_channels_input_output_image = 3; //RGB, 8 bit per channel for all images

    cl_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        width * height * num_channels_input_output_image * sizeof(float), NULL, NULL);


  std::string const kernel_file_path = "./kernels/simple_task_kernels/simple_patch_match.cl";
  std::string const& kernel_random_initialization = "random_initialization";
  std::string const& kernel_propagation = "propagation";
  std::string const& kernel_convert_RGBA_to_R = "convert_RGBA_to_R";
  std::string const& kernel_plane_refinement = "plane_refinement";
  std::string const& kernel_view_propagation = "view_propagation";

  std::string source_string{""};
  dsm::load_kernel_from_file(kernel_file_path.c_str(), source_string);
  char const* source = source_string.c_str();
  size_t source_buffer_size = source_string.size();
  program = clCreateProgramWithSource(context, 1, &source, &source_buffer_size, NULL);
  cl_int status_code = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  if (CL_SUCCESS != status_code) {
    char *log;
    size_t size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    log = (char *)malloc(size+1);
    if (log) {

      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
      size, log, NULL);
      log[size] = '\0';
      printf("%s", log);
      free(log);
    }
        exit(-1);
  }
  // Step 1 - RANDOM INITIALIZATION

  kernel = clCreateKernel(program, kernel_random_initialization.c_str(), NULL);

  clSetKernelArg(kernel, 0, sizeof(int), (void*) &width); 
  clSetKernelArg(kernel, 1, sizeof(int), (void*) &height);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cl_output_buffer); 
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cl_plane_image_left_a); 

  size_t global_work_size[2] = {size_t(width), size_t(height)};



  float rgb_data_result[width*height*4];

  float gray_data_result_left[width*height];
  float gray_data_result_right[width*height];

  float rgb_test_data_result[width*height*4];

  //copy the host image data to the device 
 
  status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                           global_work_size, NULL, 0, NULL, NULL);

  //cl_plane_image_right_a = cl_plane_image_left_a;
                             
  
  std::size_t num_byte_to_read_result = width * height * 3 * sizeof(float);
  //copy random values for the right image planes
  cl_plane_image_right_a = cl_plane_image_left_a;

  status_code = clEnqueueReadImage(command_queue, cl_plane_image_left_a, CL_TRUE, 
                             origin, region, 0, 0, rgb_data_result, 0, NULL, NULL);


  status_code = clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                                     num_byte_to_read_result, output_image.data, 0, NULL, NULL);

  output_image *= 255.0;
  cv::imwrite(output_filename.c_str(), output_image);

  //2. PROPAGATION

  clEnqueueWriteImage(command_queue, cl_left_input_image, CL_TRUE, origin, region, 0, 0, left_image_grayscale.data, 0, NULL, NULL);
  clEnqueueWriteImage(command_queue, cl_right_input_image, CL_TRUE, origin, region, 0, 0, right_image_grayscale.data, 0, NULL, NULL);

  kernel = clCreateKernel(program, kernel_propagation.c_str(), NULL);
  
  size_t global_work_size_horizon[2] = {1, size_t(height)};
  size_t global_work_size_vertical[2] = {size_t(width), 1};

  int compute_left_disparity_map = 0; // 0 == true

  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cl_left_input_image); 
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cl_right_input_image);  
  clSetKernelArg(kernel, 6, sizeof(int), (void*)&width); 
  clSetKernelArg(kernel, 7, sizeof(int), (void*)&height);


  int propagation_index = 0;

  for(uint32_t curr_iteration = 0; curr_iteration < num_iterations; ++curr_iteration) {
    


    
    propagation_index = curr_iteration % 4;
    
    clSetKernelArg(kernel, 8, sizeof(int), (void *)&propagation_index);

    // if(0 == curr_iteration) {
    //   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&cl_plane_image_left_a);
    //   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&cl_plane_image_right_a);
    //   clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
    //   clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cl_plane_image_right_b);      
    // }

    size_t current_iteration_work_size[2] = {0, 0};
    // (horizontal)/even iterations
    if(0 == propagation_index % 2) {
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_a);      
      clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
      clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cl_plane_image_right_b);
      
      for(uint32_t dim_idx = 0; dim_idx < 2; ++dim_idx) {
        current_iteration_work_size[dim_idx] = global_work_size_horizon[dim_idx];
      }
    } else { // (vertical)/odd iterations
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_b);      
      clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
      clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cl_plane_image_right_a);
      for(uint32_t dim_idx = 0; dim_idx < 2; ++dim_idx) {
        current_iteration_work_size[dim_idx] = global_work_size_vertical[dim_idx];
      }
    // status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
    //                                     global_work_size_vertical , NULL, 0, NULL, NULL);
    }
    


    status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                        current_iteration_work_size, NULL, 0, NULL, NULL);


    
    
    // VIEW PROPAGATION
    kernel_view = clCreateKernel(program, kernel_view_propagation.c_str(), NULL);

      clSetKernelArg(kernel_view, 2, sizeof(cl_mem), (void *)&cl_left_input_image);        
      clSetKernelArg(kernel_view, 3, sizeof(cl_mem), (void *)&cl_right_input_image);
      clSetKernelArg(kernel_view, 6, sizeof(int), (void*)&width); 
      clSetKernelArg(kernel_view, 7, sizeof(int), (void*)&height);

      if(propagation_index % 2 == 0) {
        clSetKernelArg(kernel_view, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
        clSetKernelArg(kernel_view, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_b);
        clSetKernelArg(kernel_view, 4, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
        clSetKernelArg(kernel_view, 5, sizeof(cl_mem), (void *)&cl_plane_image_right_a);        
      } else {
        clSetKernelArg(kernel_view, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
        clSetKernelArg(kernel_view, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_a);
        clSetKernelArg(kernel_view, 4, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
        clSetKernelArg(kernel_view, 5, sizeof(cl_mem), (void *)&cl_plane_image_right_b);                 
      }



      status_code = clEnqueueNDRangeKernel(command_queue, kernel_view, 2, NULL, 
                                          global_work_size, NULL, 0, NULL, NULL);
    
  

    // PLANE REFINEMENT
    kernel_plane = clCreateKernel(program, kernel_plane_refinement.c_str(), NULL);

      clSetKernelArg(kernel_plane, 2, sizeof(cl_mem), (void *)&cl_left_input_image);
      clSetKernelArg(kernel_plane, 3, sizeof(cl_mem), (void *)&cl_right_input_image);
      clSetKernelArg(kernel_plane, 4, sizeof(cl_mem), (void *)&cl_plane_image_temp_left);
      clSetKernelArg(kernel_plane, 5, sizeof(cl_mem), (void *)&cl_plane_image_temp_right);
      clSetKernelArg(kernel_plane, 6, sizeof(int), (void *)&width);
      clSetKernelArg(kernel_plane, 7, sizeof(int), (void *)&height);
      if(propagation_index % 2 == 0) {
        clSetKernelArg(kernel_plane, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
        clSetKernelArg(kernel_plane, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_b);
      } else {
        clSetKernelArg(kernel_plane, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
        clSetKernelArg(kernel_plane, 1, sizeof(cl_mem), (void *)&cl_plane_image_right_a);
      }

      status_code = clEnqueueNDRangeKernel(command_queue, kernel_plane, 2, NULL,
                                            global_work_size, NULL, 0, NULL, NULL);

      if(propagation_index % 2 == 0) {
        cl_plane_image_right_b = cl_plane_image_temp_right;
        cl_plane_image_left_b = cl_plane_image_temp_left;
      } else {
        cl_plane_image_right_a = cl_plane_image_temp_right;
        cl_plane_image_left_a = cl_plane_image_temp_left;
      }                                                  

  }



  //Convert plane image to 1 channel disparity
  kernel = clCreateKernel(program, kernel_convert_RGBA_to_R.c_str(), NULL);
  //if you want to see a first Iteration Image please use plane image b and outcomment iteration 2-4

  if(num_iterations % 2 == 0) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_b);
  } else {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_left_a);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_disparity_image_left);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&width); 
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&height);
  status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                       global_work_size, NULL, 0, NULL, NULL);
  
  status_code = clEnqueueReadImage(command_queue, cl_disparity_image_left, CL_TRUE, 
                                    origin, region, 0, 0, gray_data_result_left, 0, NULL, NULL);
  
  // clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_disparity_image_right);                                    
  // status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
  //                                      global_work_size, NULL, 0, NULL, NULL);
  // status_code = clEnqueueReadImage(command_queue, cl_disparity_image_right, CL_TRUE, 
  //                                   origin, region, 0, 0, gray_data_result_right, 0, NULL, NULL);


 //right image
  if(num_iterations % 2 == 0) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_right_a);
  } else {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_plane_image_right_b);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_disparity_image_right);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&width); 
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&height);
  status_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                       global_work_size, NULL, 0, NULL, NULL);
  
  status_code = clEnqueueReadImage(command_queue, cl_disparity_image_right, CL_TRUE, 
                                    origin, region, 0, 0, gray_data_result_right, 0, NULL, NULL);
  
  status_code = clFinish(command_queue);

  if(CL_SUCCESS != status_code) {
    std::cout << "Encountered a cl-Error in the execution of submitted commandQueue operations." << "\n";
    std::cout << "Error Code: " << status_code << "\n";
  }


  std::string const& window_name{"Monitor Window"};	
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::resizeWindow(window_name, width, height);
  do {
		//cv::Mat image_mat = cv::Mat(height, width, CV_32FC4, rgb_data_result);
    cv::Mat image_mat = cv::Mat(height, width, CV_8UC1, gray_data_result_left);

		cv::imshow(window_name, image_mat);
		int key = cv::waitKey(1) & 0xFF;
		if(27 == key) {
			break;	
		}	
	} while(true);



    return 0;
}
