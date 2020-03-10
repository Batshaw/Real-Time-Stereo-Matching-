#include "GPUConstWinStereoMatcher.h"

#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUConstWinStereoMatcher>
GPUConstWinStereoMatcher::create(cl_context const& context, cl_device_id const& device_id,
                        cv::Vec2i const& image_dimensions, ConstWinStereoMatchingMode const& stereo_matching_mode) {
    std::shared_ptr<GPUConstWinStereoMatcher> image_blender_ptr = std::make_shared<GPUConstWinStereoMatcher>(context, device_id, image_dimensions, stereo_matching_mode);
	return image_blender_ptr;
}

GPUConstWinStereoMatcher::GPUConstWinStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ConstWinStereoMatchingMode const& stereo_matching_mode)
  : GPUImageProcessorBinary(image_dims), stereo_matching_mode_(stereo_matching_mode) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
  }

void GPUConstWinStereoMatcher::process(cl_command_queue const& command_queue,
						 	                         cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4,
						                           cl_mem& out_image_buffer) {

    {
        const int search_window_half_size = search_window_half_size_; //TODO exchange for user-defined value

        std::map<int, cl_int> kernel_arg_statuses;
        kernel_arg_statuses[0] = clSetKernelArg(cl_cost_kernel_, 0, sizeof(cl_mem), (void *)&in_image_buffer_1);
        kernel_arg_statuses[1] = clSetKernelArg(cl_cost_kernel_, 1, sizeof(cl_mem), (void *)&in_image_buffer_2);
        kernel_arg_statuses[2] = clSetKernelArg(cl_cost_kernel_, 2, sizeof(cl_mem), (void *)&cl_cost_buffer_);
        kernel_arg_statuses[3] = clSetKernelArg(cl_cost_kernel_, 3, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses[4] = clSetKernelArg(cl_cost_kernel_, 4, sizeof(int), (void *) &image_dims_[1]);
        kernel_arg_statuses[5] = clSetKernelArg(cl_cost_kernel_, 5, sizeof(int), (void *) &search_window_half_size);
        kernel_arg_statuses[6] = clSetKernelArg(cl_cost_kernel_, 6, sizeof(int), (void *) &maximum_disparity_);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Cost_Computtaion: Could not set kernel argument #" + std::to_string(status_pair.first)
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }

        size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

    #if ENABLE_KERNEL_PROFILING
        cl_event timer_event;
        clEnqueueNDRangeKernel(command_queue, cl_cost_kernel_, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);

        register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: Cost Computation");
    #else
        clEnqueueNDRangeKernel(command_queue, cl_cost_kernel_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    #endif
    }

    {
        //run mean cost filter
        std::map<int, cl_int> kernel_arg_statuses_1;
        kernel_arg_statuses_1[0] = clSetKernelArg(cl_aggre_kernel_, 0, sizeof(cl_mem), (void *)&cl_cost_buffer_);
        kernel_arg_statuses_1[1] = clSetKernelArg(cl_aggre_kernel_, 1, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses_1[2] = clSetKernelArg(cl_aggre_kernel_, 2, sizeof(int), (void *) &image_dims_[1]);
        kernel_arg_statuses_1[3] = clSetKernelArg(cl_aggre_kernel_, 3, sizeof(int), (void*) &maximum_disparity_);
        kernel_arg_statuses_1[4] = clSetKernelArg(cl_aggre_kernel_, 4, sizeof(int), (void*) &search_window_half_size_);
        kernel_arg_statuses_1[5] = clSetKernelArg(cl_aggre_kernel_, 5, sizeof(cl_mem), (void *)&cl_cost_buffer_back_);

        for(auto const& status_pair : kernel_arg_statuses_1) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Cost Aggegation: Could not set kernel argument #" + std::to_string(status_pair.first)
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }

        size_t global_work_size_1[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

    #if ENABLE_KERNEL_PROFILING
        cl_event timer_event;
        clEnqueueNDRangeKernel(command_queue, cl_aggre_kernel_, 2, NULL, global_work_size_1, NULL, 0, NULL, &timer_event);

        register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: Const Win Cost Aggregation");
    #else
        clEnqueueNDRangeKernel(command_queue, cl_aggre_kernel_, 2, NULL, global_work_size_1, NULL, 0, NULL, 0);
    #endif
    }

    {
        //run winner take all
        std::map<int, cl_int> kernel_arg_statuses;
        kernel_arg_statuses[0] = clSetKernelArg(cl_wta_kernel_, 0, sizeof(cl_mem), (void *)&cl_cost_buffer_back_);
        kernel_arg_statuses[1] = clSetKernelArg(cl_wta_kernel_, 1, sizeof(cl_mem), (void *)&out_image_buffer);
        kernel_arg_statuses[2] = clSetKernelArg(cl_wta_kernel_, 2, sizeof(int), (void *) &image_dims_[0]);
        kernel_arg_statuses[3] = clSetKernelArg(cl_wta_kernel_, 3, sizeof(int), (void *) &image_dims_[1]);
        kernel_arg_statuses[4] = clSetKernelArg(cl_wta_kernel_, 4, sizeof(int), (void *) &maximum_disparity_);

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Disparity Winner Take All: Could not set kernel argument #" + std::to_string(status_pair.first)
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }

        size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

    #if ENABLE_KERNEL_PROFILING
        cl_event timer_event;
        clEnqueueNDRangeKernel(command_queue, cl_wta_kernel_, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);

        register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: Disparity Winner Take All");
    #else
        size_t global_work_size_1[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
        clEnqueueNDRangeKernel(command_queue, cl_wta_kernel_, 2, NULL, global_work_size_1, NULL, 0, NULL, 0);
    #endif
    }

}

void GPUConstWinStereoMatcher::set_mode(ConstWinStereoMatchingMode const& stereo_matching_mode) {
	stereo_matching_mode_ = stereo_matching_mode;
}

void GPUConstWinStereoMatcher::set_search_window_half_size(int in_search_window_half_size) {
    search_window_half_size_ = in_search_window_half_size;
}

void GPUConstWinStereoMatcher::set_maximum_disparity(int in_maximum_disparity) {
    maximum_disparity_ = in_maximum_disparity;
}


void GPUConstWinStereoMatcher::_init_kernels(cl_context const& context, cl_device_id const& device_id) {

    {
        std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/compute_SAD_cost_green_3x8.cl";
        std::string const kernel_function_name = "compute_SAD_cost_green_3x8";

        cl_program program_to_compile = 0;
        cl_kernel kernel_to_compile = 0;

        dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                    program_to_compile, kernel_to_compile);

        // register compiled kernel in defined enum
        cl_cost_kernel_ = kernel_to_compile;
        // assign a the name to kernel for profiling prints
        cl_cost_kernel_name_ = kernel_function_name;

        cl_programs_.push_back(program_to_compile);
    }
    //==========================================================================================================================================
    {
       std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/aggregate_cost_functions/const_win_cost_aggregate.cl";
       std::string const kernel_function_name = "const_win_cost_aggregate";

       cl_program program_to_compile = 0;
       cl_kernel kernel_to_compile = 0;

       dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                     program_to_compile, kernel_to_compile);

       // register compiled kernel in defined enum
       cl_aggre_kernel_ = kernel_to_compile;
       // assign a the name to kernel for profiling prints
       cl_aggre_kernel_name_ = kernel_function_name;

       cl_programs_.push_back(program_to_compile);
   }

   //==========================================================================================================================================
   {
       std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/best_disp_winner_take_all.cl";
       std::string const kernel_function_name = "best_disp_winner_take_all";

       cl_program program_to_compile = 0;
       cl_kernel kernel_to_compile = 0;

       dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                     program_to_compile, kernel_to_compile);

       // register compiled kernel in defined enum
       cl_wta_kernel_ = kernel_to_compile;
       // assign a the name to kernel for profiling prints
       cl_wta_kernel_name_ = kernel_function_name;

       cl_programs_.push_back(program_to_compile);
   }
}

void GPUConstWinStereoMatcher::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
    cl_int err_num;
    unsigned long cost_size = sizeof(cl_float) * image_dims_[0] * image_dims_[1] * maximum_disparity_;
    cl_cost_buffer_ = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                cost_size,
                nullptr,
                &err_num);

    dsm::check_cl_error(err_num, "GPUConstWinStereoMatcher:cl_cost_buffer_, clCreateBuffer");

  cl_cost_buffer_back_ = clCreateBuffer(
                                        context,
                                        CL_MEM_READ_WRITE,
                                        cost_size,
                                        nullptr,
                                        &err_num);

}

void GPUConstWinStereoMatcher::_cleanup_kernels() {
    clReleaseKernel(cl_cost_kernel_);
    clReleaseKernel(cl_aggre_kernel_);
    clReleaseKernel(cl_wta_kernel_);
}

void GPUConstWinStereoMatcher::_cleanup_memory_objects() {
    clReleaseMemObject(cl_cost_buffer_);
    clReleaseMemObject(cl_cost_buffer_back_);
}

}
