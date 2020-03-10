#ifndef DSM_GPU_POINTCLOUD_RECONSTRUCTOR_H
#define DSM_GPU_POINTCLOUD_RECONSTRUCTOR_H

#include <image_processing/base/GPU3DReconstructor.h>

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types

#include <map>

/* Derived Class  
 */

namespace dsm {

enum class ReconstructionMode {
	FLOAT_DISPARITY_TO_XYZRGB_POINTS,
	FLOAT_DISPARITY_TO_VERTEX_COLOR_TRIANGLES,
	FLOAT_DISPARITY_TO_VERTEX_UV_TRIANGLES_POINTS,
	UNDEFINED
};


struct cl_commnd_queue;

class GPUPointcloudReconstructor : public GPU3DReconstructor {
public:


	//factory function
	static std::shared_ptr<GPUPointcloudReconstructor> create(cl_context const& context, cl_device_id const& device_id,
												   			  cv::Vec2i const& image_dimensions, ReconstructionMode const& recon_mode = ReconstructionMode::FLOAT_DISPARITY_TO_VERTEX_COLOR_TRIANGLES);
	
	GPUPointcloudReconstructor(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ReconstructionMode const& recon_mode);
	~GPUPointcloudReconstructor() {};

	virtual void process(cl_command_queue const& command_queue, 
				  		 cl_mem const& in_disparity_image, cl_mem const& in_color_image, 
				  		 cl_mem& out_xyz_buffer, cl_mem& out_color_buffer) override;


	void set_mode(ReconstructionMode const& recon_mode);

private:
    cl_kernel _register_kernel(cl_context const& context, cl_device_id const& device_id,
		     			  std::string const& kernel_path, std::string const& kernel_function_name);
	
	
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) override;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) override;

	virtual void _cleanup_kernels() override;
	virtual void _cleanup_memory_objects() override;

// non-overriding private member functions
private:

	void _init_1x32f_disparity_to_pcl_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_1x32f_disparity_to_vertex_color_triangles_kernel(cl_context const& context, cl_device_id const& device_id);
	void _init_1x32f_disparity_to_vertex_uv_triangles_kernel(cl_context const& context, cl_device_id const& device_id);
// member variables
private:
	
	cl_kernel float_disparity_to_point_cloud_kernel_ = 0;
	cl_kernel float_disparity_to_colored_triangles_kernel_ = 0;
	cl_kernel float_disparity_to_textured_triangles_kernel_ = 0;

	ReconstructionMode recon_mode_ = ReconstructionMode::UNDEFINED;

	
};


} ///namespace dsm


#endif // DSM_GPU_POINTCLOUD_RECONSTRUCTOR_H