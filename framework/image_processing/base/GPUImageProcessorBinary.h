#ifndef DSM_GPU_IMAGE_PROCESSOR_BINARY_H
#define DSM_GPU_IMAGE_PROCESSOR_BINARY_H

#include "GPUBaseImageProcessor.h"

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types

/* Derived class for defining image processing operations which take
   two images as input and one image as output
 */

namespace dsm {
class GPUImageProcessorBinary : public GPUBaseImageProcessor{
public:
	//actual functionality of the image processor, override in subclass
	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						 cl_mem& out_image_buffer) = 0;

	virtual ~GPUImageProcessorBinary();


protected:
	GPUImageProcessorBinary(cv::Vec2i const& image_dims);
};

} //namespace dsm
#endif //DSM_GPU_IMAGE_PROCESSOR_BINARY_H