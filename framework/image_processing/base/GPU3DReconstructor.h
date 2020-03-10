#ifndef DSM_GPU_3D_RECONSTRUCTOR_H
#define DSM_GPU_3D_RECONSTRUCTOR_H

#include "GPUBaseImageProcessor.h"

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types

/* Derived class for defining 3D reconstruction operations which take
   two images (1 disparity and 1 color) as input and creates 2 output buffers (3x float xyz and 3x unsigned char rgb)
 */

namespace dsm {
class GPU3DReconstructor: public GPUBaseImageProcessor{
public:
	//actual functionality of the image processor, override in subclass
	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_disparity_image, cl_mem const& in_color_image, 
						 cl_mem& out_xyz_buffer, cl_mem& out_color_buffer) = 0;


	virtual ~GPU3DReconstructor();

	void set_baseline(float baseline);
	void set_focal_length(float focal_length);
	void set_disparity_scaling(float disparity_scaling);

	void set_min_valid_disparity(int min_valid_disparity);

	void set_enable_distance_cut_off(bool enable);
protected:
	GPU3DReconstructor(cv::Vec2i const& image_dims);

	float focal_length_      = 0.0f;
	float baseline_          = 0.0f;
	float disparity_scaling_  = 1.0f;

	int min_valid_disparity_ = 0;


	bool use_distance_cut_off_ = false;
};

} //namespace dsm
#endif //DSM_GPU_3D_RECONSTRUCTOR_H