#ifndef DSM_GPU_AD_CENSUS_STEREO_MATCHER_H
#define DSM_GPU_AD_CENSUS_STEREO_MATCHER_H

#include <image_processing/base/GPUImageProcessorBinary.h>

#include <image_processing/unary/GPUImageConverter.h>
#include <map>
#include <chrono>
#include <ctime>

namespace dsm {

class GPUAdCensusStereoMatcher : public GPUImageProcessorBinary {
public:

    //TODO: opencv.createTrackbar doens't allow float value, so we have to use Int type for all fields
    struct Params{
        int lambdaAD = 10;
        int lambdaCensus = 30;
        int tau1 = 20;
        int tau2 = 6;
        int tauSO = 15;
        int Pi1 = 1;
        int Pi2 = 3;
        int votingThreshold = 20;
        int maxSearchDepth = 20;
    };

	//factory function
	static std::shared_ptr<GPUAdCensusStereoMatcher> create(cl_context const& context, cl_device_id const& device_id,
												   cv::Vec2i const& image_dimensions);

    GPUAdCensusStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims);
	~GPUAdCensusStereoMatcher() {};

	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						 cl_mem& out_buffer) override;


	void set_search_window_half_size(int in_search_window_half_size);
	void set_minimum_disparity(cl_context const& context, cl_device_id const& device_id, int in_minimum_disparity);
	void set_maximum_disparity(cl_context const& context, cl_device_id const& device_id, int in_maximum_disparity);
	void set_parameters(Params const& params);
private:
    void aggregate_cost(cl_command_queue const &command_queue);
    void initialize_cost(cl_command_queue const &command_queue);

    void do_copy_buf_to_img(cl_command_queue const& command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height);
    void do_copy_img_to_buf(cl_command_queue const& command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height);
    void do_copy_img_float_to_buf(cl_command_queue const& command_queue, cl_mem const &image, cl_mem const &buffer, int width, int height);
    void do_ad_census(cl_command_queue const &command_queue, cl_mem const &left_image, cl_mem const &right_image, cl_mem const &cost_volume,
                      const int direction, const int dMin, const int dMax,
                      const float lambdaAD, const float lambdaCensus,
                      const int censusWinH, const int censusWinW);

    void do_mux_average_kernel(cl_command_queue const& command_queue, cl_mem &image_in, cl_mem &image_out);
    void do_ci_ad_kernel_5(cl_command_queue const& command_queue, cl_mem &img_l, cl_mem &img_r,
            cl_mem &cost_l, cl_mem &cost_r, int zero_disp, int num_disp);
    void do_ci_ad_kernel_2(cl_command_queue const& command_queue, cl_mem &left_image, cl_mem &right_image,
            cl_mem &left_cost, cl_mem &right_cost, int zero_disp, int num_disp);
    void do_tx_census_9x7_kernel_3(cl_command_queue const& command_queue, cl_mem &img, cl_mem &census);
    void do_ci_census_kernel_2(cl_command_queue const& command_queue, cl_mem &census_l, cl_mem &census_r,
                               cl_mem &cost_l, cl_mem &cost_r, int zero_disp, int num_disp);
    void do_ci_adcensus_kernel(cl_command_queue const& command_queue, cl_mem &ad_cost_l, cl_mem &ad_cost_r,
                               cl_mem &census_cost_l, cl_mem &census_cost_r,
                               cl_mem &adcensus_cost_l, cl_mem &adcensus_cost_r,
                               float inv_ad_coeff, float inv_census_coeff, int zero_disp, int num_disp);

    void do_compute_limits(cl_command_queue const &command_queue, cl_mem const &input_image, cl_mem const &limits_image, const int tau1, const int tau2, const int L1, const int L2);
    void do_cost_to_disparity(cl_command_queue const &command_queue, cl_mem const &cost, cl_mem const &disp, int dmin, int dmax);
    void do_aggregation_hor(cl_command_queue const &comand_queue, cl_mem const &cost_volume_src, cl_mem  const &cost_volume_target, cl_mem  const &limits, int dmin, int dmax);
    void do_aggregation_ver(cl_command_queue const &comand_queue, cl_mem const &cost_volume_src, cl_mem  const &cost_volume_target, cl_mem  const &limits, int dmin, int dmax);

    void do_agg_normalize(cl_command_queue const &comand_queue, cl_mem  const &cost_1, cl_mem  const &cost_2, cl_mem  const &limits, int horizontal, int dmin, int dmax);
    void do_scanline_optimize(cl_command_queue const &comand_queue, cl_mem const &image_1, cl_mem const &image_2, cl_mem const &cost_volume_src, cl_mem const &cost_volume_target,
                              const int dMin, const int dMax,
                              const float Pi1, const float Pi2, const int tauSO,
                              const int direction, const int vertical, const int right);

    void do_outlier_detection(cl_command_queue const &comand_queue,
            cl_mem const &disparity_left, cl_mem const &disparity_right,
            cl_mem const &disparity_image, cl_mem const &outlier_mask,
            int dmin, int dmax);

    void do_region_voting(cl_command_queue const &comand_queue,
            cl_mem const &disparity_src, cl_mem const &outlier_mask_src,
            cl_mem const &limits_image, cl_mem const &disparity_target,  cl_mem const &outlier_mask_target,
            int dmin, int dmax, int horizontal, int votingThreshold, float votingRatioThreshold);

    void do_proper_interpolation(cl_command_queue const &command_queue,
            cl_mem const & disparity_src, cl_mem const &outlier_mask_src, cl_mem const &left_image,
             cl_mem const &disparity_target, cl_mem const &outlier_mask_target, int max_search_depth);

    void do_gaussian_3x3(cl_command_queue const &command_queue, cl_mem &image_src, cl_mem &image_target);

    void do_sobel(cl_command_queue const &command_queue,  cl_mem &image_src, cl_mem &image_target, cl_mem &theta);

    void do_non_max_suppression(cl_command_queue const &command_queue,  cl_mem &image_src, cl_mem &theta, cl_mem &image_target);

    void do_hysteresis_thresholding(cl_command_queue const &command_queue, cl_mem &image_src,  cl_mem &image_target, uint32_t low, uint32_t  high);

    void do_discontinuity_adjustment(cl_command_queue const &command_queue,
            cl_mem &disparity_src,  cl_mem &outlier_mask, cl_mem &cost_volume, cl_mem &edges, cl_mem &disparity_target, int dMin);

    void do_subpixel_enhancement(cl_command_queue const &command_queue, cl_mem &disparity_src,  cl_mem &cost_volume, cl_mem &disparity_float_target, int dMin, int dMax);

    void do_median_3x3(cl_command_queue const &command_queue, cl_mem &disparity_float_src,  cl_mem &disparity_float_target);

    virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:
	// initialize converter and other auxiliary processors needed for this class
    void _init_helper_processors(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims);
	std::string _random_string();
// member variables
private:
	
	std::shared_ptr<dsm::GPUImageConverter> image_converter_ptr_ = nullptr;

    std::vector<std::string> kernel_names_ = {"ad_census", "compute_limits", "aggregate_hor2", "aggregate_ver2", "agg_normalize", "scanline_optimize",
                                   "outlier_detection", "region_voting", "proper_interpolation", "gaussian_3x3", "sobel",
                                   "non_max_suppression", "hysteresis_thresholding", "discontinuity_adjustment", "subpixel_enhancement", "median_3x3", "cost_to_disparity"};
    std::map<std::string, cl_kernel> cl_adcencus_kernels_;

    std::vector<std::string> kernel_init_cost_names_ = {"mux_average_kernel", "ci_ad_kernel_5", "ci_ad_kernel_2", "tx_census_9x7_kernel_3", "ci_census_kernel_2", "ci_adcensus_kernel"};
    std::map<std::string, cl_kernel> cl_adcencus_init_cost_kernels_;

    cl_kernel  cl_kernel_copy_buffer_to_img;
    cl_kernel  cl_kernel_copy_img_to_buffer;
    cl_kernel  cl_kernel_copy_img_float_to_buffer;
    cl_kernel  cl_kernel_debug_vol_to_buffer_;


    int dMin = 0;
    int dMax = 60;

    int censusWinH = 9;
    int censusWinW = 7;
    float lambdaAD = 10.0;
    float lambdaCensus = 30.0;
    int L1 = 32;
    int L2 = 17;
    int tau1 = 20;
    int tau2 = 6;
    int tauSO = 15;
    float Pi1 = 1.0;
    float Pi2 = 3.0;
    int votingThreshold = 20;
    float votingRatioThreshold = 0.4;
    int maxSearchDepth = 20;
//intermediate buffers
private:
	cl_mem cl_in_image_1_ = 0;
	cl_mem cl_in_image_2_ = 0;

	cl_mem cl_cost_volume_left_1 = 0;
    cl_mem cl_cost_volume_left_2 = 0;
    cl_mem cl_cost_volume_right_1 = 0;
    cl_mem cl_cost_volume_right_2 = 0;

    cl_mem  cl_cost_volume_ad_left = 0;
    cl_mem  cl_cost_volume_ad_right = 0;
    cl_mem  cl_cost_volume_census_left = 0;
    cl_mem  cl_cost_volume_census_right = 0;
    cl_mem  cl_avg_left = 0;
    cl_mem  cl_avg_right = 0;
    cl_mem  cl_census_right = 0;
    cl_mem  cl_census_left = 0;

    cl_mem cl_left_limits = 0;
    cl_mem cl_right_limits = 0;

    cl_mem cl_disparity_left = 0;
    cl_mem cl_disparity_right = 0;
    cl_mem cl_disparity_1 = 0, cl_disparity_2 = 0;
    cl_mem cl_float_disparity_1 = 0, cl_float_disparity_2 = 0;
    cl_mem cl_outlier_mask_1 = 0, cl_outlier_mask_2 = 0;
    cl_mem cl_canny_1 = 0, cl_canny_2 = 0, cl_thetas = 0;
};


} ///namespace dsm


#endif // DSM_GPU_SIMPLE_STEREO_MATCHER_H