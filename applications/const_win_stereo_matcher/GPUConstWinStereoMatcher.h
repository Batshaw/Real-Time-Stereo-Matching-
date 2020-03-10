#ifndef DSM_GPU_CONSTANT_WIN_STEREO_MATCHER_H
#define DSM_GPU_CONSTANT_WIN_STEREO_MATCHER_H

#include <image_processing/base/GPUImageProcessorBinary.h>

#include <map>

namespace dsm {

enum class ConstWinStereoMatchingMode {
	SIMPLE_SAD_GREEN_3x8,
	SIMPLE_SAD_GRAYSCALE_1x8,
	SIMPLE_SAD_LAB_3x16F,
	SIMPLE_SAD_LAB_3x32F,
	SIMPLE_ASW_LAB_3x32F,

	UNDEFINED
};

class GPUConstWinStereoMatcher : public GPUImageProcessorBinary {
public:


	//factory function
    static std::shared_ptr<GPUConstWinStereoMatcher> create(cl_context const& context, cl_device_id const& device_id,
                                                   cv::Vec2i const& image_dimensions, ConstWinStereoMatchingMode const& similarity_measure_mode);
	
    GPUConstWinStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ConstWinStereoMatchingMode const& similarity_measure_mode);
    ~GPUConstWinStereoMatcher() {};

	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4,
						 cl_mem& out_image_buffer) override;


    void set_mode(ConstWinStereoMatchingMode const& similarity_measure_mode);
	void set_search_window_half_size(int in_search_window_half_size);
	void set_maximum_disparity(int in_maximum_disparity);
private:
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// member variables
private:
    cl_mem cl_cost_buffer_;
    cl_mem cl_cost_buffer_back_;

    cl_kernel cl_cost_kernel_;
    std::string cl_cost_kernel_name_;

    cl_kernel cl_aggre_kernel_;
    std::string cl_aggre_kernel_name_;

    cl_kernel cl_wta_kernel_; //winner take all
    std::string cl_wta_kernel_name_;


    ConstWinStereoMatchingMode stereo_matching_mode_ = ConstWinStereoMatchingMode::UNDEFINED;
	int search_window_half_size_ = 10;
	int maximum_disparity_ = 60;
};


} ///namespace dsm


#endif // DSM_GPU_SIMPLE_STEREO_MATCHER_H
