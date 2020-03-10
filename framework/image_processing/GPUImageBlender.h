#ifndef DSM_GPU_IMAGE_BLENDER_H
#define DSM_GPU_IMAGE_BLENDER_H

#include <image_processing/base/GPUImageProcessorBinary.h>

#include <unordered_map>

namespace dsm {

enum class BlendMode {
	ADD,
	SUBTRACT,
	MULTIPLY,

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
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2,
						 cl_mem& out_image_buffer) const override;


	void set_mode(BlendMode const& blend_mode);

private:
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:

	void _init_addition_kernel(cl_context const& context, cl_device_id const& device_id, BlendMode const& blend_mode);
	void _init_subtraction_kernel(cl_context const& context, cl_device_id const& device_id, BlendMode const& blend_mode);
	void _init_multiplication_kernel(cl_context const& context, cl_device_id const& device_id, BlendMode const& blend_mode);
// member variables
private:
	std::unordered_map<BlendMode, cl_kernel> cl_kernels_per_mode_;
	std::unordered_map<BlendMode, std::string> cl_kernel_names_per_mode_;

	BlendMode blend_mode_ = BlendMode::UNDEFINED;
};


} ///namespace dsm


#endif // DSM_GPU_IMAGE_BLENDER_H