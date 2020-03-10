#include <image_processing/binary/GPUStereoMatcherNCC.h>

#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUStereoMatcherNCC> 
GPUStereoMatcherNCC::create(cl_context const& context, cl_device_id const& device_id,
						 cv::Vec2i const& image_dimensions) {
	std::shared_ptr<GPUStereoMatcherNCC> image_blender_ptr = std::make_shared<GPUStereoMatcherNCC>(context, device_id, image_dimensions);
	return image_blender_ptr;
}

GPUStereoMatcherNCC::GPUStereoMatcherNCC(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims)
  : GPUImageProcessorBinary(image_dims) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
}

void GPUStereoMatcherNCC::process(cl_command_queue const& command_queue, 
						 	  cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						      cl_mem& out_image_buffer) {
        const int search_window_half_size = search_window_half_size_; //TODO exchange for user-defined value
    const int maximum_disparity = maximum_disparity_; //TODO exchange for user-defined value

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
    // perform initial conversion
    
    {
        cl_kernel const& active_kernel = ncc_conversion_gs_;

        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&in_image_buffer_1);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_gs_l_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[1]);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: conversion_bytebuffer_to_gs_image2d");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    {
        cl_kernel const& active_kernel = ncc_conversion_gs_;

        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&in_image_buffer_2);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_gs_r_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[1]);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: conversion_bytebuffer_to_gs_image2d");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    // ******* Preprocess mean
    {
        cl_kernel const& active_kernel = ncc_mean_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_gs_l_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_mean_l_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *)&search_window_half_size);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: preprocess_mean_l");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    {
        cl_kernel const& active_kernel = ncc_mean_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_gs_r_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_mean_r_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *)&search_window_half_size);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: preprocess_mean_r");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    // ******** Preprocess variance
    {
        cl_kernel const& active_kernel = ncc_variance_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_gs_l_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_mean_l_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&ncc_variance_l_);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *)&search_window_half_size);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: preprocess_variance_l");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    {
        cl_kernel const& active_kernel = ncc_variance_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_gs_r_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_mean_r_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&ncc_variance_r_);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *)&search_window_half_size);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: preprocess_variance_r");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

    // ******* Matching stage
    {
        const int minimum_disparity = 0;
        cl_kernel const& active_kernel = ncc_match_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_gs_l_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ncc_gs_r_);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&ncc_mean_l_);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&ncc_mean_r_);
        kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(cl_mem), (void *)&ncc_variance_l_);
        kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(cl_mem), (void *)&ncc_variance_r_);
        kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(cl_mem), (void *)&ncc_output_);
        kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(int), (void *)&search_window_half_size);
        kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(int), (void *)&minimum_disparity);
        kernel_arg_statuses[9] = clSetKernelArg(active_kernel, 9, sizeof(int), (void *)&maximum_disparity);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }


#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: ncc_aggregation_left_right");
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }
    // ******* Final conversion to byte buffer

    {
        cl_kernel const& active_kernel = ncc_conversion_gs_buffer_;
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ncc_output_);
        kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&out_image_buffer);
        kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[1]);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }

        size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
        std::cout << "image dims: " << image_dims_[0] << ":" << image_dims_[1] << std::endl;

#if ENABLE_KERNEL_PROFILING
        cl_event image_blending_timer_event;
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

        register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: conversion_image2d_to_bytebuffer");
        std::cout << "Performed final conversion" << std::endl;
#else
        clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

    }

}

void GPUStereoMatcherNCC::set_search_window_half_size(int in_search_window_half_size) {
    search_window_half_size_ = in_search_window_half_size;
}

void GPUStereoMatcherNCC::set_minimum_disparity(int in_maximum_disparity) {
    minimum_disparity_ = in_maximum_disparity;
}

void GPUStereoMatcherNCC::set_maximum_disparity(int in_maximum_disparity) {
    maximum_disparity_ = in_maximum_disparity;
}

void GPUStereoMatcherNCC::_init_NCC_kernels(cl_context const& context, cl_device_id const& device_id) {
    std::cout << "Initializing NCC kernels" << std::endl;

    dsm::compile_kernel_from_file(context, device_id, 
            "./kernels/image_processing/unary/conversion/bgr_3x8_to_grayscale_1x8_image_conversion.cl", 
            "convert_image_bgr_3x8_to_grayscale_1x8_image",
            ncc_conversion_gs_program_, ncc_conversion_gs_);
    
    dsm::compile_kernel_from_file(context, device_id, 
            "./kernels/image_processing/unary/conversion/image_to_buffer_conversion.cl", 
            "convert_image_to_buffer_1x32f",
            ncc_conversion_gs_buffer_program_, ncc_conversion_gs_buffer_);

    dsm::compile_kernel_from_file(context, device_id, 
            "./kernels/image_processing/unary/misc/image_mean.cl", 
            "mean",
            ncc_mean_program_, ncc_mean_);
    dsm::compile_kernel_from_file(context, device_id, 
            "./kernels/image_processing/unary/misc/image_variance.cl", 
            "variance",
            ncc_variance_program_, ncc_variance_);
    dsm::compile_kernel_from_file(context, device_id, 
            "./kernels/image_processing/binary/stereo_matching/matching_algorithms/image_ncc.cl", 
            "match",
            ncc_match_program_, ncc_match_);
}


void GPUStereoMatcherNCC::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
   // init NCC kernels
   _init_NCC_kernels(context, device_id);
}


void GPUStereoMatcherNCC::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
    // init NCC buffers
    std::cout << "Creating tmp buffers\n";
    cl_image_format image_format_grayscale = { CL_R, CL_UNORM_INT8 };
    cl_image_format image_format_mean = { CL_R, CL_UNORM_INT16 };
    cl_image_format image_format_variance = { CL_R, CL_UNORM_INT16 };
    cl_image_format image_format_disparity = { CL_R, CL_UNSIGNED_INT8 };
    cl_image_desc image_desc = {CL_MEM_OBJECT_IMAGE2D, image_dims_[0], image_dims_[1]};
    cl_int* errcode;
    ncc_gs_l_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_grayscale, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "Error occured during clCreateImage\n";
    ncc_gs_r_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_grayscale, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "Error occured during clCreateImage\n";
    ncc_mean_l_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_mean, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "Error occured during clCreateImage\n";
    ncc_mean_r_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_mean, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "error occured during clcreateimage\n";
    ncc_variance_l_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_variance, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "error occured during clcreateimage\n";
    ncc_variance_r_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_variance, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "error occured during clcreateimage\n";
    ncc_output_ = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_disparity, &image_desc, nullptr, errcode);
    if(errcode != nullptr)
        std::cout << "error occured during clcreateimage\n";
}

void GPUStereoMatcherNCC::_cleanup_kernels() {
    clReleaseKernel(ncc_conversion_gs_);
    clReleaseKernel(ncc_conversion_gs_buffer_);
    clReleaseKernel(ncc_mean_);
    clReleaseKernel(ncc_variance_);
    clReleaseKernel(ncc_match_);
}
void GPUStereoMatcherNCC::_cleanup_memory_objects() {}
	//no intermediate passes are required for the color converter
}
