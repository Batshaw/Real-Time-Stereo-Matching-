#include <image_processing/binary/GPUAdCensusStereoMatcher.h>
#include <core/utils.h>

namespace dsm {
//factory function
std::shared_ptr<GPUAdCensusStereoMatcher>
    GPUAdCensusStereoMatcher::create(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dimensions) {
	std::shared_ptr<GPUAdCensusStereoMatcher> image_blender_ptr = std::make_shared<GPUAdCensusStereoMatcher>(context, device_id, image_dimensions);
	return image_blender_ptr;
}

GPUAdCensusStereoMatcher::GPUAdCensusStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims)
: GPUImageProcessorBinary(image_dims){
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
    _init_helper_processors(context, device_id, image_dims);
}

void GPUAdCensusStereoMatcher::_init_helper_processors(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims) {
    image_converter_ptr_ = dsm::GPUImageConverter::create(context, device_id,
                                                          image_dims,
                                                          dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
}

std::string GPUAdCensusStereoMatcher::_random_string()
{
    return std::to_string(std::rand());
}
void GPUAdCensusStereoMatcher::process(cl_command_queue const& command_queue,
						 	  cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						      cl_mem& out_buffer) {

    auto start_0 = std::chrono::system_clock::now();

    do_copy_buf_to_img(command_queue, cl_in_image_1_, in_image_buffer_1, image_dims_[0], image_dims_[1]);
    do_copy_buf_to_img(command_queue, cl_in_image_2_, in_image_buffer_2, image_dims_[0], image_dims_[1]);

    auto start_1 = std::chrono::system_clock::now();

    //=================================================================================================================================================

    //do_ad_census(command_queue, cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_1, -1, dMin, dMax, lambdaAD, lambdaCensus, censusWinH, censusWinW);
    //do_ad_census(command_queue, cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_1, 1, dMin, dMax, lambdaAD, lambdaCensus, censusWinH, censusWinW);
    initialize_cost(command_queue);

    //=================================================================================================================================================
    do_compute_limits(command_queue, cl_in_image_1_, cl_left_limits, tau1, tau2, L1, L2);
    do_compute_limits(command_queue, cl_in_image_2_, cl_right_limits, tau1, tau2, L1, L2);

    //=================================================================================================================================================
    aggregate_cost(command_queue);

    //=================================================================================================================================================
    do_scanline_optimize(command_queue,
            cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_1, cl_cost_volume_left_2,
            dMin, dMax,Pi1, Pi2, tauSO, 1, 1, 0);

    do_scanline_optimize(command_queue,
            cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_2, cl_cost_volume_left_1,
            dMin, dMax,Pi1, Pi2, tauSO, -1, 1, 0);

    do_scanline_optimize(command_queue,
            cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_1, cl_cost_volume_left_2,
            dMin, dMax,Pi1, Pi2, tauSO, 1, 0, 1);

    do_scanline_optimize(command_queue,
            cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_2, cl_cost_volume_left_1,
            dMin, dMax,Pi1, Pi2, tauSO, -1, 0, 1);

    do_cost_to_disparity(command_queue, cl_cost_volume_left_2, cl_disparity_left, dMin, dMax);

    //=================================================================================================================================================
    do_scanline_optimize(command_queue,
                         cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_1, cl_cost_volume_right_2,
                         dMin, dMax,Pi1, Pi2, tauSO, 1, 1, 1);

    do_scanline_optimize(command_queue,
                         cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_2, cl_cost_volume_right_1,
                         dMin, dMax,Pi1, Pi2, tauSO, -1, 1, 1);

    do_scanline_optimize(command_queue,
                         cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_1, cl_cost_volume_right_2,
                         dMin, dMax,Pi1, Pi2, tauSO, 1, 0, 1);

    do_scanline_optimize(command_queue,
                         cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_2, cl_cost_volume_right_1,
                         dMin, dMax,Pi1, Pi2, tauSO, -1, 0, 1);

    do_cost_to_disparity(command_queue, cl_cost_volume_right_1, cl_disparity_right, dMin, dMax);

    //==================================================================================================================================================
    do_outlier_detection(command_queue, cl_disparity_left, cl_disparity_right, cl_disparity_1, cl_outlier_mask_1, dMin, dMax);

    do_region_voting(command_queue,
            cl_disparity_1,
            cl_outlier_mask_1,
            cl_left_limits,
            cl_disparity_2,
            cl_outlier_mask_2,
            dMin,
            dMax,
            1,
            votingThreshold,
            votingRatioThreshold);

    do_region_voting(command_queue,
            cl_disparity_2,
            cl_outlier_mask_2,
            cl_left_limits,
            cl_disparity_1,
            cl_outlier_mask_1,
            dMin,
            dMax,
            0,
            votingThreshold,
            votingRatioThreshold);

    do_region_voting(command_queue,
              cl_disparity_1,
              cl_outlier_mask_1,
              cl_left_limits,
              cl_disparity_2,
              cl_outlier_mask_2,
              dMin,
              dMax,
              1,
              votingThreshold,
              votingRatioThreshold);

    do_region_voting(command_queue,
              cl_disparity_2,
              cl_outlier_mask_2,
              cl_left_limits,
              cl_disparity_1,
              cl_outlier_mask_1,
              dMin,
              dMax,
              0,
              votingThreshold,
              votingRatioThreshold);

    do_region_voting(command_queue,
              cl_disparity_1,
              cl_outlier_mask_1,
              cl_left_limits,
              cl_disparity_2,
              cl_outlier_mask_2,
              dMin,
              dMax,
              1,
              votingThreshold,
              votingRatioThreshold);

    do_proper_interpolation(command_queue,
                         cl_disparity_2,
                         cl_outlier_mask_2,
                         cl_in_image_1_,
                         cl_disparity_1,
                         cl_outlier_mask_1,
                         maxSearchDepth);

    do_gaussian_3x3(command_queue,
                 cl_disparity_1,
                 cl_canny_1);

    do_sobel(command_queue,
          cl_canny_1,
          cl_canny_2,
          cl_thetas);

    do_non_max_suppression(command_queue,
                        cl_canny_2,
                        cl_thetas,
                        cl_canny_1);

    do_hysteresis_thresholding(command_queue,
                            cl_canny_1,
                            cl_canny_2,
                            20,
                            70);

    do_discontinuity_adjustment(command_queue,
                             cl_disparity_1,
                             cl_outlier_mask_1,
                             cl_cost_volume_left_1,
                             cl_canny_2,
                             cl_disparity_2,
                             dMin);

    do_subpixel_enhancement(command_queue,
                         cl_disparity_2,
                         cl_cost_volume_left_1,
                         cl_float_disparity_1,
                            dMin,
                            dMax);

    do_median_3x3(command_queue,
               cl_float_disparity_1,
               cl_float_disparity_2);

    auto end_1 = std::chrono::system_clock::now();

    //==================================================================================================================================================
    //TODO: handle float disparity buffer. currently, the input buffer is unsigned char disparity buffer
    do_copy_img_float_to_buf(command_queue, cl_float_disparity_2, out_buffer, image_dims_[0], image_dims_[1]);
    clFlush(command_queue);
    
    auto end_0 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_0 = end_0-start_0;
    std::chrono::duration<double> elapsed_seconds_1 = end_1-start_1;
    std::cout << "AdCencus: time_0 = " << " " << elapsed_seconds_0.count() * 1000 << "ms" << "\ttime_1 = "<< elapsed_seconds_1.count() * 1000 << "ms" << std::endl;
}

void GPUAdCensusStereoMatcher::initialize_cost(cl_command_queue const &cqueue)
{
    bool cost_all_in_one = false;
    if (!cost_all_in_one){
        //std::cout <<"hello khanh";
        do_ci_ad_kernel_2(cqueue,cl_in_image_1_, cl_in_image_2_,
                          cl_cost_volume_ad_left, cl_cost_volume_ad_right,
                          dMin, dMax - dMin);

        do_mux_average_kernel(cqueue, cl_in_image_1_, cl_avg_left);

        do_mux_average_kernel(cqueue, cl_in_image_2_, cl_avg_right);

        do_tx_census_9x7_kernel_3(cqueue, cl_avg_left, cl_census_left);

        do_tx_census_9x7_kernel_3(cqueue, cl_avg_right, cl_census_right);

        do_ci_census_kernel_2(cqueue, cl_census_left, cl_census_right,
                              cl_cost_volume_census_left, cl_cost_volume_census_right,
                              dMin, dMax - dMin);

        float lambdaADInv = 1.0 / lambdaAD;
        float lambdaCensusInv = 1.0 / lambdaCensus;
        do_ci_adcensus_kernel(cqueue,cl_cost_volume_ad_left, cl_cost_volume_ad_right,
                              cl_cost_volume_census_left, cl_cost_volume_census_right,
                              cl_cost_volume_left_1, cl_cost_volume_right_1,
                              lambdaADInv, lambdaCensusInv, dMin, dMax - dMin);
    }
    else{
        do_ad_census(cqueue, cl_in_image_1_, cl_in_image_2_, cl_cost_volume_left_1, -1, dMin, dMax, lambdaAD, lambdaCensus, censusWinH, censusWinW);
        do_ad_census(cqueue, cl_in_image_1_, cl_in_image_2_, cl_cost_volume_right_1, 1, dMin, dMax, lambdaAD, lambdaCensus, censusWinH, censusWinW);
    }

}

void GPUAdCensusStereoMatcher::aggregate_cost(cl_command_queue const &cqueue)
{
    // left cost aggregation
    do_aggregation_hor(cqueue, cl_cost_volume_left_1, cl_cost_volume_left_2, cl_left_limits, dMin, dMax);
    do_aggregation_ver(cqueue, cl_cost_volume_left_2, cl_cost_volume_left_1, cl_left_limits, dMin, dMax);

    do_aggregation_hor(cqueue, cl_cost_volume_left_1, cl_cost_volume_left_2, cl_left_limits, dMin, dMax);
    do_aggregation_ver(cqueue, cl_cost_volume_left_2, cl_cost_volume_left_1, cl_left_limits, dMin, dMax);

    // right cost aggregation
    do_aggregation_hor(cqueue, cl_cost_volume_right_1, cl_cost_volume_right_2, cl_right_limits, dMin, dMax);
    do_aggregation_ver(cqueue, cl_cost_volume_right_2, cl_cost_volume_right_1, cl_right_limits, dMin, dMax);

    do_aggregation_hor(cqueue, cl_cost_volume_right_1, cl_cost_volume_right_2, cl_right_limits, dMin, dMax);
    do_aggregation_ver(cqueue, cl_cost_volume_right_2, cl_cost_volume_right_1, cl_right_limits, dMin, dMax);
}

void GPUAdCensusStereoMatcher::do_copy_buf_to_img(cl_command_queue const &command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height)
{
    auto active_kernel = cl_kernel_copy_buffer_to_img;
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &width);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
}

void GPUAdCensusStereoMatcher::do_copy_img_to_buf(cl_command_queue const &command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height)
{
    auto active_kernel = cl_kernel_copy_img_to_buffer;
    std::map<int, cl_int> kernel_arg_statuses;
    int d = 0;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *)&width);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *)&height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel::debug_vol_to_img");
}

void GPUAdCensusStereoMatcher::do_copy_img_float_to_buf(cl_command_queue const& command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height)
{
    auto active_kernel = cl_kernel_copy_img_float_to_buffer;
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *)&width);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *)&height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel::debug_vol_to_img");
}

void GPUAdCensusStereoMatcher::do_ad_census(cl_command_queue const &command_queue, cl_mem const &left_image, cl_mem const &right_image, cl_mem const &cost_volume,
                      const int direction, const int dMin, const int dMax,
                      const float lambdaAD, const float lambdaCensus,
                      const int censusWinH, const int censusWinW)
{
      auto active_kernel = cl_adcencus_kernels_["ad_census"];
      std::map<int, cl_int> kernel_arg_statuses;

      kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&left_image);
      kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&right_image);
      kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&cost_volume);
      kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &direction);
      kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &dMin);
      kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *) &dMax);
      kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(float), (void *) &lambdaAD);
      kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(float), (void *) &lambdaCensus);
      kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(int), (void *) &censusWinH);
      kernel_arg_statuses[9] = clSetKernelArg(active_kernel, 9, sizeof(int), (void *) &censusWinW);

      for(auto const& status_pair : kernel_arg_statuses) {
          if(CL_SUCCESS != status_pair.second) {
              std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                          + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
              DSM_LOG_ERROR(error_message);
          }
      }

      size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

      cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
      dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
}

void GPUAdCensusStereoMatcher::do_mux_average_kernel(cl_command_queue const& command_queue, cl_mem &image_in, cl_mem &image_out)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["mux_average_kernel"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_in);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image_out);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#if ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_mux_average_kernel_" + _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_ci_ad_kernel_5(cl_command_queue const& command_queue, cl_mem &img_l, cl_mem &img_r,
                       cl_mem &cost_l, cl_mem &cost_r, int zero_disp, int num_disp)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["ci_ad_kernel_5"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&img_l);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&img_r);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&cost_l);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&cost_r);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void*)&zero_disp);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void*)&num_disp);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#if ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_ci_ad_kernel_5_" + _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_ci_ad_kernel_2(cl_command_queue const& command_queue, cl_mem &left_image, cl_mem &right_image,
                       cl_mem &left_cost, cl_mem &right_cost, int zero_disp, int num_disp)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["ci_ad_kernel_2"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&left_image);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&right_image);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&left_cost);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&right_cost);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void*)&zero_disp);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void*)&num_disp);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};


#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_ci_ad_kernel_2_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_tx_census_9x7_kernel_3(cl_command_queue const& command_queue, cl_mem &img, cl_mem &census)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["tx_census_9x7_kernel_3"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&img);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&census);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_tx_census_9x7_kernel_3_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_ci_census_kernel_2(cl_command_queue const& command_queue, cl_mem &census_l, cl_mem &census_r,
                           cl_mem &cost_l, cl_mem &cost_r, int zero_disp, int num_disp)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["ci_census_kernel_2"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&census_l);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&census_r);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&cost_l);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&cost_r);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void*)&zero_disp);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void*)&num_disp);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_ci_census_kernel_2_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_ci_adcensus_kernel(cl_command_queue const& command_queue, cl_mem &ad_cost_l, cl_mem &ad_cost_r,
                           cl_mem &census_cost_l, cl_mem &census_cost_r,
                           cl_mem &adcensus_cost_l, cl_mem &adcensus_cost_r,
                           float inv_ad_coeff, float inv_census_coeff, int zero_disp, int num_disp)
{
    auto active_kernel = cl_adcencus_init_cost_kernels_["ci_adcensus_kernel"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&ad_cost_l);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&ad_cost_r);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&census_cost_l);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&census_cost_r);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(cl_mem), (void *)&adcensus_cost_l);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(cl_mem), (void *)&adcensus_cost_r);
    kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(float), (void*)&inv_ad_coeff);
    kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(float), (void*)&inv_census_coeff);
    kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(int), (void*)&zero_disp);
    kernel_arg_statuses[9] = clSetKernelArg(active_kernel, 9, sizeof(int), (void*)&num_disp);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_ci_adcensus_kernel_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_compute_limits(cl_command_queue const &command_queue, cl_mem const &input_image, cl_mem const &limits_image, const int tau1, const int tau2, const int L1, const int L2)
{
    auto active_kernel = cl_adcencus_kernels_["compute_limits"];
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&input_image);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&limits_image);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &tau1);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &tau2);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &L1);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *) &L2);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_ci_compute_limits_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_aggregation_ver(cl_command_queue const & comand_queue, cl_mem  const &cost_volume_src, cl_mem  const &cost_volume_target, cl_mem  const &limits, int dmin, int dmax)
{
    auto active_kernel = cl_adcencus_kernels_["aggregate_ver2"];
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&cost_volume_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&cost_volume_target);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&limits);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &dmin);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &dmax);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t local_size = 16;
    size_t local_work_size[2] = {local_size, local_size};

    size_t g0 = std::lround(image_dims_[0]/local_size)*local_size;
    size_t g1 = std::lround(image_dims_[1]/local_size)*local_size;
    size_t global_work_size[2] = {g0, g1};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_aggregation_ver_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_aggregation_hor(cl_command_queue const & comand_queue, cl_mem  const &cost_volume_src, cl_mem  const &cost_volume_target, cl_mem  const &limits, int dmin, int dmax)
{
    auto active_kernel = cl_adcencus_kernels_["aggregate_hor2"];
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&cost_volume_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&cost_volume_target);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&limits);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &dmin);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &dmax);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t local_size = 16;
    size_t local_work_size[2] = {local_size, local_size};

    size_t g0 = std::lround(image_dims_[0]/local_size)*local_size;
    size_t g1 = std::lround(image_dims_[1]/local_size)*local_size;
    size_t global_work_size[2] = {g0, g1};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_aggregation_hor_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}


void GPUAdCensusStereoMatcher::do_agg_normalize(cl_command_queue const &comand_queue, cl_mem  const &cost_1, cl_mem  const &cost_2, cl_mem  const &limits, int horizontal, int dmin, int dmax)
{
    //auto active_kernel = cl_kernel_agg_normalize;
    auto active_kernel = cl_adcencus_kernels_["agg_normalize"];
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&cost_1);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&cost_2);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&limits);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &horizontal);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &dmin);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *) &dmax);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_agg_normalize_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_cost_to_disparity(cl_command_queue const &command_queue, cl_mem const &cost, cl_mem const &disp, int dmin, int dmax)
{
    auto active_kernel = cl_adcencus_kernels_["cost_to_disparity"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&cost);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&disp);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &dmin);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &dmax);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_cost_to_disparity_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_scanline_optimize(cl_command_queue const &comand_queue, cl_mem const &image_1,
                                                    cl_mem const &image_2, cl_mem const &cost_volume_src,
                                                    cl_mem const &cost_volume_target, const int dMin, const int dMax,
                                                    const float Pi1, const float Pi2,
                                                    const int tauSO, const int direction, const int vertical, const int right) {

    auto active_kernel = cl_adcencus_kernels_["scanline_optimize"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_1);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image_2);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&cost_volume_src);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&cost_volume_target);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *) &dMin);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *) &dMax);
    kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(float), (void *) &Pi1);
    kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(float), (void *) &Pi2);
    kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(int), (void *) &tauSO);
    kernel_arg_statuses[9] = clSetKernelArg(active_kernel, 9, sizeof(int), (void *) &direction);
    kernel_arg_statuses[10] = clSetKernelArg(active_kernel, 10, sizeof(int), (void *) &vertical);
    kernel_arg_statuses[11] = clSetKernelArg(active_kernel, 11, sizeof(int), (void *) &right);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_scanline_optimize_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_gaussian_3x3(cl_command_queue const &command_queue, cl_mem &image_src, cl_mem &image_target)
{
    auto active_kernel = cl_adcencus_kernels_["gaussian_3x3"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image_target);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_gaussian_3x3_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_sobel(cl_command_queue const &command_queue,  cl_mem &image_src, cl_mem &image_target, cl_mem &theta)
{
    auto active_kernel = cl_adcencus_kernels_["sobel"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image_target);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&theta);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_sobel_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_non_max_suppression(cl_command_queue const &command_queue,  cl_mem &image_src, cl_mem &theta, cl_mem &image_target)
{
    auto active_kernel = cl_adcencus_kernels_["non_max_suppression"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&theta);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&image_target);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_non_max_suppression_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_hysteresis_thresholding(cl_command_queue const &command_queue, cl_mem &image_src,  cl_mem &image_target, uint32_t low, uint32_t  high)
{
    auto active_kernel = cl_adcencus_kernels_["hysteresis_thresholding"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&image_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&image_target);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(uint32_t), (void *)&low);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(uint32_t), (void *)&high);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_hysteresis_thresholding_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_discontinuity_adjustment(cl_command_queue const &command_queue,
        cl_mem &disparity_src,  cl_mem &outlier_mask, cl_mem &cost_volume,
        cl_mem &edges, cl_mem &disparity_target, int dMin)
{
    auto active_kernel = cl_adcencus_kernels_["discontinuity_adjustment"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&outlier_mask);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&cost_volume);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&edges);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(cl_mem), (void *)&disparity_target);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *)&dMin);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_discontinuity_adjustment_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_subpixel_enhancement(cl_command_queue const &command_queue,
        cl_mem &disparity_src,  cl_mem &cost_volume, cl_mem &disparity_float_target, int dMin, int dMax)
{
    auto active_kernel = cl_adcencus_kernels_["subpixel_enhancement"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&cost_volume);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&disparity_float_target);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *)&dMin);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *)&dMax);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_subpixel_enhancement_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_median_3x3(cl_command_queue const &command_queue, cl_mem &disparity_float_src,  cl_mem &disparity_float_target)
{
    auto active_kernel = cl_adcencus_kernels_["median_3x3"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_float_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&disparity_float_target);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_median_3x3_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_outlier_detection(cl_command_queue const &comand_queue,
                          cl_mem const &disparity_left, cl_mem const &disparity_right,
                          cl_mem const &disparity_image, cl_mem const &outlier_mask,
                          int dmin, int dmax)
{
    auto active_kernel = cl_adcencus_kernels_["outlier_detection"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_left);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&disparity_right);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&disparity_image);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&outlier_mask);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(int), (void *)&dmin);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *)&dmax);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_outlier_detection_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_region_voting(cl_command_queue const &comand_queue,
                      cl_mem const &disparity_src, cl_mem const &outlier_mask_src,
                      cl_mem const &limits_image, cl_mem const &disparity_target,  cl_mem const &outlier_mask_target,
                      int dmin, int dmax, int horizontal, int votingThreshold, float votingRatioThreshold)
{
    auto active_kernel = cl_adcencus_kernels_["region_voting"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&outlier_mask_src);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&limits_image);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&disparity_target);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(cl_mem), (void *)&outlier_mask_target);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *)&dmin);
    kernel_arg_statuses[6] = clSetKernelArg(active_kernel, 6, sizeof(int), (void *)&dmax);
    kernel_arg_statuses[7] = clSetKernelArg(active_kernel, 7, sizeof(int), (void *)&horizontal);
    kernel_arg_statuses[8] = clSetKernelArg(active_kernel, 8, sizeof(int), (void *)&votingThreshold);
    kernel_arg_statuses[9] = clSetKernelArg(active_kernel, 9, sizeof(float), (void *)&votingRatioThreshold);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};

#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(comand_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_region_voting_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(comand_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::do_proper_interpolation(cl_command_queue const &command_queue,
                             cl_mem const & disparity_src, cl_mem const &outlier_mask_src, cl_mem const &left_image,
                             cl_mem const &disparity_target, cl_mem const &outlier_mask_target, int max_search_depth)
{
    auto active_kernel = cl_adcencus_kernels_["proper_interpolation"];

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&disparity_src);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&outlier_mask_src);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(cl_mem), (void *)&left_image);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(cl_mem), (void *)&disparity_target);
    kernel_arg_statuses[4] = clSetKernelArg(active_kernel, 4, sizeof(cl_mem), (void *)&outlier_mask_target);
    kernel_arg_statuses[5] = clSetKernelArg(active_kernel, 5, sizeof(int), (void *)&max_search_depth);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#ifdef ENABLE_KERNEL_PROFILING
    cl_event timer_event;
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &timer_event);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
    register_kernel_execution_time(command_queue, timer_event, get_filename_from_path(__FILE__) + ":: adcensus_proper_interpolation_"+ _random_string());
#else
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");
#endif
}

void GPUAdCensusStereoMatcher::set_minimum_disparity(cl_context const& context, cl_device_id const& device_id, int in_maximum_disparity) {
    //this is not allowed not. minimum disparity is always zero.
    //we are not sure about the effect of changing minimum disparity on kernels
    //dMin = in_maximum_disparity;
}

void GPUAdCensusStereoMatcher::set_maximum_disparity(cl_context const& context, cl_device_id const& device_id, int in_maximum_disparity) {
    if (in_maximum_disparity != dMax){
        assert(dMax > dMin);
        dMax = in_maximum_disparity;
        std::cout << "Clean up memory and kernels" << std::endl;
        _cleanup_kernels();
        _cleanup_memory_objects();
        std::cout << "Initialize kernels and memory objects" << std::endl;
        _init_kernels(context, device_id);
        _init_memory_objects(context, device_id);
    }
}

void GPUAdCensusStereoMatcher::set_parameters(Params const& params)
{
    lambdaAD = params.lambdaAD;
    lambdaCensus = params.lambdaCensus;
    tau1 = params.tau1;
    tau2 = params.tau2;
    tauSO = params.tauSO;
    Pi1 = params.Pi1;
    Pi2 = params.Pi2;
    votingThreshold = params.votingThreshold;
    maxSearchDepth = params.maxSearchDepth;
}


void GPUAdCensusStereoMatcher::_init_kernels(cl_context const& context, cl_device_id const& device_id)
{

    std::string ad_census_prog_path = "./kernels/image_processing/binary/stereo_matching/adcensus.cl";

    std::string defines = "#define NUM_DISPARITIES " + std::to_string(dMax - dMin) + "\n";
    for (const std::string &kernel_name : kernel_names_)
    {
        std::cout << "init kernel: " << kernel_name << std::endl;
        cl_kernel kernel;
        cl_program program_to_compile = 0;
        dsm::compile_kernel_from_file(context, device_id, ad_census_prog_path, kernel_name,
                                      program_to_compile, kernel, defines);
        cl_adcencus_kernels_.insert(std::make_pair(kernel_name, kernel));
    }



    std::string ad_census_init_cost_prog_path = "./kernels/image_processing/binary/stereo_matching/adcensus_cost_init.cl";
    for (const std::string &kernel_name : kernel_init_cost_names_)
    {
        std::cout << "init kernel: " << kernel_name << std::endl;
        cl_kernel kernel;
        cl_program program_to_compile = 0;
        dsm::compile_kernel_from_file(context, device_id, ad_census_init_cost_prog_path, kernel_name,
                                      program_to_compile, kernel);
        cl_adcencus_init_cost_kernels_.insert(std::make_pair(kernel_name, kernel));
    }

    std::string kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_buffer_image_copy.cl";
    {
        std::cout << "init copy_bgr_3x8_buffer_to_image kernel" << std::endl;
        cl_program program_to_compile = 0;
        std::string kernel_function_name = "copy_bgr_3x8_buffer_to_image";
        dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                      program_to_compile, cl_kernel_copy_buffer_to_img);
    }

    {
        std::cout << "init copy_image_to_buffer_1x8_buffer kernel" << std::endl;
        cl_program program_to_compile = 0;
        std::string kernel_function_name = "copy_image_to_buffer_1x8_buffer";
        dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                      program_to_compile, cl_kernel_copy_img_to_buffer);
    }

    {
        std::cout << "init copy_image_float_to_buffer_1x8_buffer kernel" << std::endl;
        cl_program program_to_compile = 0;
        std::string kernel_function_name = "copy_image_float_to_buffer_1x32f_buffer";
        dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                      program_to_compile, cl_kernel_copy_img_float_to_buffer);
    }

    {
        std::cout << "init cl_kernel_debug_vol_to_buffer_ kernel" << std::endl;
        cl_program program_to_compile = 0;
        std::string kernel_function_name = "copy_3D_image_to_buffer_1x8_buffer";
        dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                      program_to_compile, cl_kernel_debug_vol_to_buffer_);
    }

}

void GPUAdCensusStereoMatcher::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {

    {
        cl_image_format format;
        format.image_channel_order = CL_RGBA;
        format.image_channel_data_type = CL_UNSIGNED_INT8;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = 1;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_in_image_1_ = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");

        cl_in_image_2_ = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
    }

    {
        cl_image_format format;
        format.image_channel_order = CL_RGBA;
        format.image_channel_data_type = CL_UNSIGNED_INT8;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = 1;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_left_limits = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");

        cl_right_limits = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
    }

    {
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_UNSIGNED_INT8;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = 1;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_disparity_left = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_disparity_right = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_disparity_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_disparity_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");

        cl_canny_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_canny_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_thetas = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");

        cl_outlier_mask_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_outlier_mask_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");

        cl_avg_left  = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        cl_avg_right = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
    }

    {
        cl_image_format format;
        format.image_channel_order = CL_RG;
        format.image_channel_data_type = CL_UNSIGNED_INT32;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = 1;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_census_left  = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_census_right = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
    }


    {
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = 1;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_float_disparity_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
        cl_float_disparity_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage");
    }

    {
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;

        cl_image_desc desc;
        memset(&desc, '\0', sizeof(cl_image_desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE3D;
        desc.image_width = image_dims_[0];
        desc.image_height = image_dims_[1];
        desc.image_depth = dMax;
        desc.image_array_size = 1;
        desc.mem_object= NULL; //or some buffer;
        cl_int err;
        cl_cost_volume_left_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume_right_1");

        cl_cost_volume_left_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume_right_1");

        cl_cost_volume_right_1 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume_right_1");

        cl_cost_volume_right_2 = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume_right_1");

        cl_cost_volume_ad_left = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume");
        cl_cost_volume_ad_right = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume");
        cl_cost_volume_census_left = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume");
        cl_cost_volume_census_right = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        dsm::check_cl_error(err, "clCreateImage:cl_cost_volume");
    }
}

void GPUAdCensusStereoMatcher::_cleanup_kernels() {
    cl_int  status;
    for(auto& map_kernel_iterator : cl_adcencus_kernels_) {
        if(NULL != map_kernel_iterator.second) {
            status = clReleaseKernel(map_kernel_iterator.second);
            dsm::check_cl_error(status, "clReleaseKernel");
            map_kernel_iterator.second = NULL;
        }
    }
    cl_adcencus_kernels_.clear();

    for(auto& map_kernel_iterator : cl_adcencus_init_cost_kernels_) {
        if(NULL != map_kernel_iterator.second) {
            status = clReleaseKernel(map_kernel_iterator.second);
            dsm::check_cl_error(status, "clReleaseKernel");
            map_kernel_iterator.second = NULL;
        }
    }
    cl_adcencus_init_cost_kernels_.clear();

    status = clReleaseKernel(cl_kernel_copy_buffer_to_img); dsm::check_cl_error(status, "clReleaseMemObject"); cl_kernel_copy_buffer_to_img = 0;
    status = clReleaseKernel(cl_kernel_copy_img_to_buffer); dsm::check_cl_error(status, "clReleaseMemObject"); cl_kernel_copy_img_to_buffer = 0;
    status = clReleaseKernel(cl_kernel_copy_img_float_to_buffer); dsm::check_cl_error(status, "clReleaseMemObject"); cl_kernel_copy_img_float_to_buffer = 0;
    status = clReleaseKernel(cl_kernel_debug_vol_to_buffer_); dsm::check_cl_error(status, "clReleaseMemObject"); cl_kernel_debug_vol_to_buffer_ = 0;
}

void GPUAdCensusStereoMatcher::_cleanup_memory_objects(){
    cl_int  status;
    status = clReleaseMemObject(cl_in_image_1_); dsm::check_cl_error(status, "clReleaseMemObject"); cl_in_image_1_ = 0;
    status = clReleaseMemObject(cl_in_image_2_); dsm::check_cl_error(status, "clReleaseMemObject"); cl_in_image_2_ = 0;
    status = clReleaseMemObject(cl_cost_volume_left_1); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_left_1 = 0;
    status = clReleaseMemObject(cl_cost_volume_left_2); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_left_2 = 0;
    status = clReleaseMemObject(cl_cost_volume_right_1); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_right_1 = 0;
    status = clReleaseMemObject(cl_cost_volume_right_2); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_right_2 = 0;
    status = clReleaseMemObject(cl_left_limits); dsm::check_cl_error(status, "clReleaseMemObject"); cl_left_limits = 0;
    status = clReleaseMemObject(cl_right_limits); dsm::check_cl_error(status, "clReleaseMemObject"); cl_right_limits = 0;
    status = clReleaseMemObject(cl_disparity_left); dsm::check_cl_error(status, "clReleaseMemObject"); cl_disparity_left = 0;
    status = clReleaseMemObject(cl_disparity_right); dsm::check_cl_error(status, "clReleaseMemObject");cl_disparity_right = 0;
    status = clReleaseMemObject(cl_disparity_1); dsm::check_cl_error(status, "clReleaseMemObject"); cl_disparity_1 = 0;
    status = clReleaseMemObject(cl_disparity_2); dsm::check_cl_error(status, "clReleaseMemObject"); cl_disparity_2 = 0;
    status = clReleaseMemObject(cl_float_disparity_1); dsm::check_cl_error(status, "clReleaseMemObject"); cl_float_disparity_1 = 0;
    status = clReleaseMemObject(cl_float_disparity_2); dsm::check_cl_error(status, "clReleaseMemObject"); cl_float_disparity_2 = 0;

    status = clReleaseMemObject(cl_outlier_mask_1); dsm::check_cl_error(status, "clReleaseMemObject");cl_outlier_mask_1  = 0;
    status = clReleaseMemObject(cl_outlier_mask_2); dsm::check_cl_error(status, "clReleaseMemObject");cl_outlier_mask_2  = 0;
    status = clReleaseMemObject(cl_canny_1);        dsm::check_cl_error(status, "clReleaseMemObject"); cl_canny_1 = 0;
    status = clReleaseMemObject(cl_canny_2);        dsm::check_cl_error(status, "clReleaseMemObject"); cl_canny_2 = 0;
    status = clReleaseMemObject(cl_thetas);         dsm::check_cl_error(status, "clReleaseMemObject"); cl_thetas = 0;


    status = clReleaseMemObject(cl_cost_volume_ad_left); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_ad_left = 0;
    status = clReleaseMemObject(cl_cost_volume_ad_right); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_ad_right = 0;
    status = clReleaseMemObject(cl_cost_volume_census_left); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_census_left = 0;
    status = clReleaseMemObject(cl_cost_volume_census_right); dsm::check_cl_error(status, "clReleaseMemObject"); cl_cost_volume_census_right  = 0;
    status = clReleaseMemObject(cl_avg_left); dsm::check_cl_error(status, "clReleaseMemObject"); cl_avg_left = 0;
    status = clReleaseMemObject(cl_avg_right); dsm::check_cl_error(status, "clReleaseMemObject"); cl_avg_right = 0;
    status = clReleaseMemObject(cl_census_right); dsm::check_cl_error(status, "clReleaseMemObject"); cl_census_right = 0;
    status = clReleaseMemObject(cl_census_left); dsm::check_cl_error(status, "clReleaseMemObject"); cl_census_left = 0;
}
}