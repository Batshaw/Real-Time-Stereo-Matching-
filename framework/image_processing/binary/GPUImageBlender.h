#ifndef DSM_GPU_IMAGE_BLENDER_H
#define DSM_GPU_IMAGE_BLENDER_H

#include <image_processing/base/GPUImageProcessorBinary.h>

#include <map>

/* Derived Class  
 */

namespace dsm {

enum class BlendMode {
	ADD_3x8,
	DIFFERENCE_3x8,
	MULTIPLY_3x8,

	UNDEFINED
};

class GPUImageBlender : public GPUImageProcessorBinary {
public:


	//factory function
	static std::shared_ptr<GPUImageBlender> create(cl_context const& context, cl_device_id const& device_id,
												   cv::Vec2i const& image_dimensions, BlendMode const& blend_mode);
	
	GPUImageBlender(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, BlendMode const& blend_mode);
	~GPUImageBlender() {};

	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						 cl_mem& out_image_buffer) override;


	void set_mode(BlendMode const& blend_mode);

private:
    void _register_kernel(cl_context const& context, cl_device_id const& device_id,
		     			  BlendMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name);
	
	
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:

	void _init_addition_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_difference_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_multiplication_3x8_kernel(cl_context const& context, cl_device_id const& device_id);
// member variables
private:
	
	std::map<BlendMode, cl_kernel> cl_kernels_per_mode_;
	std::map<BlendMode, std::string> cl_kernel_names_per_mode_;

	BlendMode blend_mode_ = BlendMode::UNDEFINED;
};


} ///namespace dsm


#endif // DSM_GPU_IMAGE_BLENDER_H