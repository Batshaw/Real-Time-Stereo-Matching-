#ifndef DSM_GPU_STEREO_MATCHER_NCC_H
#define DSM_GPU_STEREO_MATCHER_NCC_H

#include <image_processing/base/GPUImageProcessorBinary.h>
#include <image_processing/unary/GPUImageConverter.h>
#include <map>

namespace dsm {

class GPUStereoMatcherNCC : public GPUImageProcessorBinary {
public:


	//factory function
	static std::shared_ptr<GPUStereoMatcherNCC> create(cl_context const& context, cl_device_id const& device_id,
												   cv::Vec2i const& image_dimensions);
	
	GPUStereoMatcherNCC(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims);
	~GPUStereoMatcherNCC() {};

	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						 cl_mem& out_image_buffer) override;


	void set_search_window_half_size(int in_search_window_half_size);
	void set_minimum_disparity(int in_minimum_disparity);
	void set_maximum_disparity(int in_maximum_disparity);
private:
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:
    // NCC matching
	void _init_NCC_kernels(cl_context const& context, cl_device_id const& device_id);

// member variables
private:
	std::shared_ptr<dsm::GPUImageConverter> image_converter_ptr_ = nullptr;

	int search_window_half_size_ = 10;
	int minimum_disparity_ = 0;
	int maximum_disparity_ = 60;

//intermediate buffers
private:
    cl_mem ncc_gs_l_;
    cl_mem ncc_gs_r_;
    cl_mem ncc_mean_l_;
    cl_mem ncc_mean_r_;
    cl_mem ncc_variance_l_;
    cl_mem ncc_variance_r_;
    cl_mem ncc_output_;
    cl_kernel ncc_conversion_gs_;
    cl_program ncc_conversion_gs_program_;
    cl_kernel ncc_conversion_gs_buffer_;
    cl_program ncc_conversion_gs_buffer_program_;
    cl_kernel ncc_mean_;
    cl_program ncc_mean_program_;
    cl_kernel ncc_variance_;
    cl_program ncc_variance_program_;
    cl_kernel ncc_match_;
    cl_program ncc_match_program_;
};


} ///namespace dsm


#endif // DSM_GPU_SIMPLE_STEREO_MATCHER_H
