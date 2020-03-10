#include <image_processing/binary/GPUStereoMatcher.h>

#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUStereoMatcher> 
GPUStereoMatcher::create(cl_context const& context, cl_device_id const& device_id,
						 cv::Vec2i const& image_dimensions, StereoMatchingMode const& stereo_matching_mode) {
	std::shared_ptr<GPUStereoMatcher> image_blender_ptr = std::make_shared<GPUStereoMatcher>(context, device_id, image_dimensions, stereo_matching_mode);
	return image_blender_ptr;
}

GPUStereoMatcher::GPUStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, StereoMatchingMode const& stereo_matching_mode)
  : GPUImageProcessorBinary(image_dims), stereo_matching_mode_(stereo_matching_mode) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);

    _init_helper_processors(context, device_id, image_dims);
  }

void GPUStereoMatcher::_init_helper_processors(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims) {
    image_converter_ptr_ = dsm::GPUImageConverter::create(context, device_id,
                                                          image_dims,
                                                          dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
}

void GPUStereoMatcher::_convert_RGB_3x8_to_grayscale_1x8(cl_command_queue const& command_queue, 
                                                         cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2) {
    
    image_converter_ptr_->set_mode(dsm::ConversionMode::BGR_3x8_TO_GRAYSCALE_1x8);
    image_converter_ptr_->process(command_queue, in_image_buffer_1, grayscale_1x8_buffer_1_);
    image_converter_ptr_->process(command_queue, in_image_buffer_2, grayscale_1x8_buffer_2_);   
}

void GPUStereoMatcher::_convert_RGB_3x8_to_LAB_3x32f(cl_command_queue const& command_queue, 
                                                     cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2) {
    
    image_converter_ptr_->set_mode(dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
    image_converter_ptr_->process(command_queue, in_image_buffer_1, lab_3x32f_image_buffer_1_);
    image_converter_ptr_->process(command_queue, in_image_buffer_2, lab_3x32f_image_buffer_2_);   
}

void GPUStereoMatcher::process(cl_command_queue const& command_queue, 
						 	  cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						      cl_mem& out_image_buffer) {

    if(StereoMatchingMode::SIMPLE_CENSUS_GRAYSCALE_1x8 == stereo_matching_mode_ ||
       StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
        _convert_RGB_3x8_to_grayscale_1x8(command_queue, in_image_buffer_1, in_image_buffer_2);
    } else if(StereoMatchingMode::SIMPLE_SAD_LAB_3x32F == stereo_matching_mode_ ||
              StereoMatchingMode::SIMPLE_ASW_LAB_3x32F == stereo_matching_mode_) {
        _convert_RGB_3x8_to_LAB_3x32f(command_queue, in_image_buffer_1, in_image_buffer_2);       
    }
    // reference to kernel used used for our color conversion;
    auto const& map_iterator = cl_kernels_per_mode_.find(stereo_matching_mode_);
    if(cl_kernels_per_mode_.end() == map_iterator) {
        std::cout << "Error: StereoMatchingMode was not defined.\n";
        throw std::exception();
    }
    cl_kernel const& active_kernel = map_iterator->second;
    const int search_window_half_size = search_window_half_size_; //TODO exchange for user-defined value
    const int maximum_disparity = maximum_disparity_; //TODO exchange for user-defined value

    std::map<int, cl_int> kernel_arg_statuses; 

    if(StereoMatchingMode::SIMPLE_CENSUS_GRAYSCALE_1x8 == stereo_matching_mode_ ||
       StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&grayscale_1x8_buffer_1_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&grayscale_1x8_buffer_2_);
    } else if(StereoMatchingMode::SIMPLE_SAD_LAB_3x32F == stereo_matching_mode_ || 
              StereoMatchingMode::SIMPLE_ASW_LAB_3x32F == stereo_matching_mode_ ) {
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&lab_3x32f_image_buffer_1_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&lab_3x32f_image_buffer_2_);       
    } else {
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&in_image_buffer_1);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&in_image_buffer_2);   
    }
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&out_image_buffer);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[0]);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &image_dims_[1]);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *) &search_window_half_size);
    kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(int), (void *) &minimum_disparity_);    
    kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(int), (void *) &maximum_disparity_);

    /*if(StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
        std::cout << "ALLOCATE LOCAL MEM\n";
        kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(unsigned char) * 100, NULL);
    }*/

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                      + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }




size_t global_work_size[2] = {image_dims_[0], image_dims_[1]};

size_t local_work_size[2] = {1, 1};

if(StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
    global_work_size[0] = 256;
    global_work_size[1] = 256;

    local_work_size[0] = 32;
    local_work_size[1] = 32;

} 


cl_int kernel_execution_status = CL_SUCCESS;
#if ENABLE_KERNEL_PROFILING
    cl_event image_blending_timer_event;
    if(StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
        kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &image_blending_timer_event);
    } else {
        kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);
    }
    register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: simple_stereomatching_kernel");
#else
    if(StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8 == stereo_matching_mode_) {
        kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, 0);
    } else {
        kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    }
#endif

  if(kernel_execution_status != CL_SUCCESS) {
    std::cout << "KERNEL EXECUTIOn FAILED: " << kernel_execution_status << "\n";
  }
}

void GPUStereoMatcher::set_mode(StereoMatchingMode const& stereo_matching_mode) {
	stereo_matching_mode_ = stereo_matching_mode;
}

void GPUStereoMatcher::set_search_window_half_size(int in_search_window_half_size) {
    search_window_half_size_ = in_search_window_half_size;
}

void GPUStereoMatcher::set_minimum_disparity(int in_maximum_disparity) {
    minimum_disparity_ = in_maximum_disparity;
}

void GPUStereoMatcher::set_maximum_disparity(int in_maximum_disparity) {
    maximum_disparity_ = in_maximum_disparity;
}

void GPUStereoMatcher::_register_kernel(cl_context const& context, cl_device_id const& device_id,
                                        StereoMatchingMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name) {
    cl_program program_to_compile = 0;
    cl_kernel kernel_to_compile = 0;

    dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                  program_to_compile, kernel_to_compile);

    // register compiled kernel in defined enum
    cl_kernels_per_mode_[mode] = kernel_to_compile;
    // assign a the name to kernel for profiling prints
    cl_kernel_names_per_mode_[mode] = kernel_function_name;

    cl_programs_.push_back(program_to_compile);
}

void GPUStereoMatcher::_init_simple_SAD_green_3x8_to_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    StereoMatchingMode const mode = StereoMatchingMode::SIMPLE_SAD_GREEN_3x8;
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_SAD_green_3x8.cl";
    std::string const kernel_function_name = "compute_disparity_simple_green_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUStereoMatcher::_init_simple_census_grayscale_1x8_to_1x8_kernel(cl_context const& context, cl_device_id const& device_id) {

    StereoMatchingMode const mode = StereoMatchingMode::SIMPLE_CENSUS_GRAYSCALE_1x8;
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_census_grayscale_1x8.cl";
    std::string const kernel_function_name = "compute_disparity_simple_grayscale_1x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}


void GPUStereoMatcher::_init_simple_SAD_grayscale_local_memory_1x8_to_1x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    StereoMatchingMode const mode = StereoMatchingMode::SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8;
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_SAD_grayscale_local_memory_1x8.cl";
    std::string const kernel_function_name = "compute_disparity_grayscale_local_memory_1x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUStereoMatcher::_init_simple_SAD_lab_3x32f_to_3x32f_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    StereoMatchingMode const mode = StereoMatchingMode::SIMPLE_SAD_LAB_3x32F;
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_SAD_lab_3x32f.cl";
    std::string const kernel_function_name = "compute_disparity_simple_lab_3x32f";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUStereoMatcher::_init_simple_ASW_lab_3x32f_to_3x32f_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    StereoMatchingMode const mode = StereoMatchingMode::SIMPLE_ASW_LAB_3x32F;
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_ASW_lab_3x32f.cl";
    std::string const kernel_function_name = "compute_disparity_simple_lab_3x32f";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}



void GPUStereoMatcher::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
    // matching of 3x8 bgr images
   _init_simple_SAD_green_3x8_to_3x8_kernel(context, device_id);
   // matching of 1x8 grayscale images
   _init_simple_census_grayscale_1x8_to_1x8_kernel(context, device_id);
    
    // matching of 1x8 grayscale images optimized with local memory usage
   _init_simple_SAD_grayscale_local_memory_1x8_to_1x8_kernel(context, device_id);

   // matching of 3x32f lab images
   _init_simple_SAD_lab_3x32f_to_3x32f_kernel(context, device_id);

   _init_simple_ASW_lab_3x32f_to_3x32f_kernel(context, device_id);
}

void GPUStereoMatcher::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
    // grayscale images
    grayscale_1x8_buffer_1_ = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                            image_dims_[0] * image_dims_[1] * 1 * sizeof(char), NULL, NULL);

    grayscale_1x8_buffer_2_ = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                            image_dims_[0] * image_dims_[1] * 1 * sizeof(char), NULL, NULL);

    // lab space images
    lab_3x32f_image_buffer_1_ = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                               image_dims_[0] * image_dims_[1] * 3 * sizeof(float), NULL, NULL);

    lab_3x32f_image_buffer_2_ = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                               image_dims_[0] * image_dims_[1] * 3 * sizeof(float), NULL, NULL);
}

void GPUStereoMatcher::_cleanup_kernels() {
    for(auto& map_kernel_iterator : cl_kernels_per_mode_) {

        if(NULL != map_kernel_iterator.second) {
            clReleaseKernel(map_kernel_iterator.second);
            map_kernel_iterator.second = NULL;
        }
    }
}
void GPUStereoMatcher::_cleanup_memory_objects() {}
	//no intermediate passes are required for the color converter
}
