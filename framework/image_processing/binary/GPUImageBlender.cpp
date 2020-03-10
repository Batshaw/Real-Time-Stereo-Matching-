#include <image_processing/binary/GPUImageBlender.h>

#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUImageBlender> 
GPUImageBlender::create(cl_context const& context, cl_device_id const& device_id,
						cv::Vec2i const& image_dimensions, BlendMode const& blend_mode) {
	std::shared_ptr<GPUImageBlender> image_blender_ptr = std::make_shared<GPUImageBlender>(context, device_id, image_dimensions, blend_mode);
	return image_blender_ptr;
}

GPUImageBlender::GPUImageBlender(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, BlendMode const& blend_mode)
  : GPUImageProcessorBinary(image_dims), blend_mode_(blend_mode) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
  }

void GPUImageBlender::process(cl_command_queue const& command_queue, 
						 	  cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						      cl_mem& out_image_buffer) {

    // reference to kernel used used for our color conversion;
    auto const& map_iterator = cl_kernels_per_mode_.find(blend_mode_);
    if(cl_kernels_per_mode_.end() == map_iterator) {
        DSM_LOG_ERROR("BlendMode was not defined!");
        throw std::exception();
    }
    cl_kernel const& active_kernel = map_iterator->second;



    std::map<int, cl_int> kernel_arg_statuses; 
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&in_image_buffer_1);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&in_image_buffer_2);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&out_image_buffer);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[0]);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &image_dims_[1]);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                      + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }




size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#if ENABLE_KERNEL_PROFILING
    cl_event image_blending_timer_event;
    clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_blending_timer_event);

    register_kernel_execution_time(command_queue, image_blending_timer_event, get_filename_from_path(__FILE__) + ":: image_blending_kernel");
#else
    clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif
}

// mode setter in order to switch blend modes
void GPUImageBlender::set_mode(BlendMode const& blend_mode) {
	blend_mode_ = blend_mode;
}


// routine for compiling the kernels 
void GPUImageBlender::_register_kernel(cl_context const& context, cl_device_id const& device_id,
                                        BlendMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name) {
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


/*** the following functions all just define a kernel path for a configurd mode label (see GPUImageBlender.h)
     and also tell the kernel the name of the main kernel
***/
void GPUImageBlender::_init_addition_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    BlendMode const mode = BlendMode::ADD_3x8;
    std::string const kernel_path = "./kernels/image_processing/binary/blending/addition_3x8.cl";
    std::string const kernel_function_name = "add_images_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageBlender::_init_multiplication_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    BlendMode const mode = BlendMode::MULTIPLY_3x8;
    std::string const kernel_path = "./kernels/image_processing/binary/blending/multiplication_3x8.cl";
    std::string const kernel_function_name = "multiply_images_3x8";
    
    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageBlender::_init_difference_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    BlendMode const mode = BlendMode::DIFFERENCE_3x8;
    std::string const kernel_path = "./kernels/image_processing/binary/blending/difference_3x8.cl";
    std::string const kernel_function_name = "subtract_absolute_images_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

/* do not forget to to call the helper functions for each of your modes here! */
void GPUImageBlender::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
   _init_addition_3x8_kernel(context, device_id);
   _init_multiplication_3x8_kernel(context, device_id);
   _init_difference_3x8_kernel(context, device_id);
} 

/* you could initialize additional intermediate memory objects here in case the image processing
   task needs intermediate compute passes and therefore temporary objects */
void GPUImageBlender::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
	//no intermediate passes are required for the color converter
}

/* This cleans up kernels automatically */
void GPUImageBlender::_cleanup_kernels() {
    for(auto& map_kernel_iterator : cl_kernels_per_mode_) {

        if(NULL != map_kernel_iterator.second) {
            clReleaseKernel(map_kernel_iterator.second);
            map_kernel_iterator.second = NULL;
        }
    }
}


void GPUImageBlender::_cleanup_memory_objects() {}
	//no intermediate passes are required for the color converter
}