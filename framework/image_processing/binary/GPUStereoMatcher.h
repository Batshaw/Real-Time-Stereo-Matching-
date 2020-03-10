#ifndef DSM_GPU_SIMPLE_STEREO_MATCHER_H
#define DSM_GPU_SIMPLE_STEREO_MATCHER_H

#include <image_processing/base/GPUImageProcessorBinary.h>

#include <image_processing/unary/GPUImageConverter.h>
#include <map>

namespace dsm {

enum class StereoMatchingMode {
	SIMPLE_SAD_GREEN_3x8,
	SIMPLE_CENSUS_GRAYSCALE_1x8,
	SIMPLE_SAD_GRAYSCALE_LOCAL_MEMORY_1x8,
	SIMPLE_SAD_LAB_3x32F,
	SIMPLE_ASW_LAB_3x32F,
	UNDEFINED
};

class GPUStereoMatcher : public GPUImageProcessorBinary {
public:


	//factory function
	static std::shared_ptr<GPUStereoMatcher> create(cl_context const& context, cl_device_id const& device_id,
												   cv::Vec2i const& image_dimensions, StereoMatchingMode const& similarity_measure_mode);
	
	GPUStereoMatcher(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, StereoMatchingMode const& similarity_measure_mode);
	~GPUStereoMatcher() {};

	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						 cl_mem& out_image_buffer) override;

	void set_mode(StereoMatchingMode const& similarity_measure_mode);
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
    void _register_kernel(cl_context const& context, cl_device_id const& device_id,
		     			  StereoMatchingMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name);
	

	void _convert_RGB_3x8_to_grayscale_1x8(cl_command_queue const& command_queue, 
	                                       cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2);

	void _convert_RGB_3x8_to_LAB_3x32f(cl_command_queue const& command_queue, 
	                                   cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2);


	// initialize converter and other auxiliary processors needed for this class
    void _init_helper_processors(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims);

	// SAD matching to 1x8 disparity
	// 3x8 bgr
	void _init_simple_SAD_green_3x8_to_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	// 1x8 grayscale
	void _init_simple_census_grayscale_1x8_to_1x8_kernel(cl_context const& context, cl_device_id const& device_id);
	// 1x8 grayscale using local memory
	void _init_simple_SAD_grayscale_local_memory_1x8_to_1x8_kernel(cl_context const& context, cl_device_id const& device_id);
	// 3x32f lab
	void _init_simple_SAD_lab_3x32f_to_3x32f_kernel(cl_context const& context, cl_device_id const& device_id);

	// ASW matching
	void _init_simple_ASW_lab_3x32f_to_3x32f_kernel(cl_context const& context, cl_device_id const& device_id);

    // NCC matching
	void _init_NCC_kernels(cl_context const& context, cl_device_id const& device_id);

// member variables
private:
	
	std::shared_ptr<dsm::GPUImageConverter> image_converter_ptr_ = nullptr;

	std::map<StereoMatchingMode, cl_kernel> cl_kernels_per_mode_;
	std::map<StereoMatchingMode, std::string> cl_kernel_names_per_mode_;

	StereoMatchingMode stereo_matching_mode_ = StereoMatchingMode::UNDEFINED;
	int search_window_half_size_ = 10;
	int minimum_disparity_ = 0;
	int maximum_disparity_ = 60;

//intermediate buffers
private:
	//if matching is performed on grayscale images, 3x8
	cl_mem grayscale_1x8_buffer_1_ = 0;
	cl_mem grayscale_1x8_buffer_2_ = 0;
	
	//if matching is performed on LAB images, 3x32F 
	cl_mem lab_3x32f_image_buffer_1_ = 0;
	cl_mem lab_3x32f_image_buffer_2_ = 0;

};


} ///namespace dsm


#endif // DSM_GPU_SIMPLE_STEREO_MATCHER_H
