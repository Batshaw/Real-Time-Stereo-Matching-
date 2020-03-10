#ifndef DSM_GPU_IMAGE_CONVERTER_H
#define DSM_GPU_IMAGE_CONVERTER_H

#include <image_processing/base/GPUImageProcessorUnary.h>

#include <map>

namespace dsm {

enum class ConversionMode {
	// unnormalized conversions
	//3x8 -> 3x8 modes:
	BGR_3x8_TO_GRAYSCALE_3x8,
	BGR_3x8_TO_BINARY_3x8,
	BGR_3x8_TO_RGB_3x8,
	//3x8 -> 1x8 modes:
	BGR_3x8_TO_GRAYSCALE_1x8,
	//3x8 -> 3x16f modes:
	BGR_3x8_TO_LAB_3x16F,
	//3x8 -> 3x32f modes:
	BGR_3x8_TO_LAB_3x32F,

	//1x32f -> 3x8 modes:
	FLOAT_1x32F_TO_RGB_3x8,
	UNDEFINED
};

class GPUImageConverter : public GPUImageProcessorUnary {
public:


	//factory function
	static std::shared_ptr<GPUImageConverter> create(cl_context const& context, cl_device_id const& device_id,
														  cv::Vec2i const& image_dimensions, ConversionMode const& conversion_mode);
	
	GPUImageConverter(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ConversionMode const& conversion_mode);
	virtual ~GPUImageConverter();

	virtual void process(cl_command_queue const& command_queue, cl_mem const& in_image_buffer, cl_mem& out_image_buffer) override;


	void set_mode(ConversionMode const& in_conversion_mode);
private:
    void _register_kernel(cl_context const& context, cl_device_id const& device_id,
		     			  ConversionMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name);
	
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:
	//3x8 -> 3x8 modes
	void _init_bgr_3x8_to_grayscale_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_bgr_3x8_to_binary_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_bgr_3x8_to_rgb_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	//3x8 -> 1x8 modes	
	void _init_bgr_3x8_to_grayscale_1x8_kernel(cl_context const& context, cl_device_id const& device_id);
	//3x8 -> 3x16f modes
	void _init_bgr_3x8_to_lab_3x16f_kernel(cl_context const& context, cl_device_id const& device_id);
	//3x8 -> 3x32f modes
	void _init_bgr_3x8_to_lab_3x32f_kernel(cl_context const& context, cl_device_id const& device_id);
// member variables
private:
	std::map<ConversionMode, cl_kernel> cl_kernels_per_mode_;
	std::map<ConversionMode, std::string> cl_kernel_names_per_mode_;

	ConversionMode conversion_mode_ = ConversionMode::UNDEFINED;
};


} ///namespace dsm


#endif // DSM_GPU_IMAGE_COLOR_CONVERTER_H