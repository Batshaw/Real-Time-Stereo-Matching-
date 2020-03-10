#include <image_processing/binary/GPUPointcloudReconstructor.h>

#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUPointcloudReconstructor> 
GPUPointcloudReconstructor::create(cl_context const& context, cl_device_id const& device_id,
						cv::Vec2i const& image_dimensions, ReconstructionMode const& recon_mode) {
	std::shared_ptr<GPUPointcloudReconstructor> reconstructor_ptr = std::make_shared<GPUPointcloudReconstructor>(context, device_id, image_dimensions, recon_mode);
	return reconstructor_ptr;
}

GPUPointcloudReconstructor::GPUPointcloudReconstructor(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ReconstructionMode const& recon_mode)
  : GPU3DReconstructor(image_dims), recon_mode_(recon_mode) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
  }



void 
GPUPointcloudReconstructor::process(cl_command_queue const& command_queue, 
                                    cl_mem const& in_disparity_image, cl_mem const& in_color_image, 
                                    cl_mem& out_xyz_buffer, cl_mem& out_color_buffer) {





    if(ReconstructionMode::FLOAT_DISPARITY_TO_VERTEX_COLOR_TRIANGLES == recon_mode_) {
        std::map<int, cl_int> kernel_arg_statuses; 
        kernel_arg_statuses[0] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 0, sizeof(cl_mem), (void *)&in_disparity_image);
        kernel_arg_statuses[1] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 1, sizeof(cl_mem), (void *)&out_xyz_buffer);
        kernel_arg_statuses[2] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 2, sizeof(int), (void *)&image_dims_[0]);
        kernel_arg_statuses[3] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 3, sizeof(int), (void *)&image_dims_[1]);   
        kernel_arg_statuses[4] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 4, sizeof(float), (void *)&baseline_);
        kernel_arg_statuses[5] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 5, sizeof(float), (void *)&focal_length_);    
        kernel_arg_statuses[6] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 6, sizeof(float), (void *)&disparity_scaling_);
        kernel_arg_statuses[7] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 7, sizeof(cl_mem), (void *)&in_color_image);
        kernel_arg_statuses[8] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 8, sizeof(cl_mem), (void *)&out_color_buffer);
        kernel_arg_statuses[9] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 9, sizeof(int), (void *)&min_valid_disparity_);
        
        int distance_cut_off_variable = use_distance_cut_off_ ? 1 : 0;

        kernel_arg_statuses[10] = clSetKernelArg(float_disparity_to_colored_triangles_kernel_, 10, sizeof(int), (void *) &distance_cut_off_variable);            
        


        //... andere kernel argumente

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }




    size_t global_work_size[2] = {size_t(image_dims_[0]-1), size_t(image_dims_[1]-1)};

    #if ENABLE_KERNEL_PROFILING
        cl_event recon_timer_event;
        clEnqueueNDRangeKernel(command_queue, float_disparity_to_colored_triangles_kernel_, 2, NULL, global_work_size, NULL, 0, NULL, &recon_timer_event);

        register_kernel_execution_time(command_queue, recon_timer_event, get_filename_from_path(__FILE__) + ":: recon_kernel");
    #else
        clEnqueueNDRangeKernel(command_queue, float_disparity_to_colored_triangles_kernel_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    #endif

    }

}


// mode setter in order to switch blend modes
void GPUPointcloudReconstructor::set_mode(ReconstructionMode const& recon_mode) {
	recon_mode_ = recon_mode;
}


// routine for compiling the kernels 
cl_kernel GPUPointcloudReconstructor::_register_kernel(cl_context const& context, cl_device_id const& device_id,
                                                  std::string const& kernel_path, std::string const& kernel_function_name) {
    cl_program program_to_compile = 0;
    cl_kernel kernel_to_compile = 0;

    dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                  program_to_compile, kernel_to_compile);


    cl_programs_.push_back(program_to_compile);

    return kernel_to_compile;
}


/*** the following functions all just define a kernel path for a configurd mode label (see GPUPointcloudReconstructor.h)
     and also tell the kernel the name of the main kernel
***/
void GPUPointcloudReconstructor::_init_1x32f_disparity_to_pcl_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/3d_reconstruction/reconstruct_3D_geometry_from_disparity.cl";
    std::string const kernel_function_name = "reconstruct_3D_points_GPU";

    float_disparity_to_point_cloud_kernel_ = _register_kernel(context, device_id, kernel_path, kernel_function_name);
}

void GPUPointcloudReconstructor::_init_1x32f_disparity_to_vertex_color_triangles_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/3d_reconstruction/reconstruct_3D_geometry_from_disparity.cl";
    std::string const kernel_function_name = "reconstruct_3D_colored_triangles_GPU";

    float_disparity_to_colored_triangles_kernel_ = _register_kernel(context, device_id, kernel_path, kernel_function_name);
}

void GPUPointcloudReconstructor::_init_1x32f_disparity_to_vertex_uv_triangles_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/3d_reconstruction/reconstruct_3D_geometry_from_disparity.cl";
    std::string const kernel_function_name = "reconstruct_3D_textured_triangles_GPU";

    float_disparity_to_textured_triangles_kernel_ = _register_kernel(context, device_id, kernel_path, kernel_function_name);
    
}



/* do not forget to to call the helper functions for each of your modes here! */
void GPUPointcloudReconstructor::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
   _init_1x32f_disparity_to_pcl_kernel(context, device_id);
   _init_1x32f_disparity_to_vertex_color_triangles_kernel(context, device_id);
   _init_1x32f_disparity_to_vertex_uv_triangles_kernel(context, device_id);
} 

/* you could initialize additional intermediate memory objects here in case the image processing
   task needs intermediate compute passes and therefore temporary objects */
void GPUPointcloudReconstructor::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
	//no intermediate passes are required for the color converter
}

/* This cleans up kernels automatically */
void GPUPointcloudReconstructor::_cleanup_kernels() {
    clReleaseKernel(float_disparity_to_point_cloud_kernel_);
    clReleaseKernel(float_disparity_to_colored_triangles_kernel_);
    clReleaseKernel(float_disparity_to_textured_triangles_kernel_);
}


void GPUPointcloudReconstructor::_cleanup_memory_objects() {}
	//no intermediate passes are required for the color converter
}