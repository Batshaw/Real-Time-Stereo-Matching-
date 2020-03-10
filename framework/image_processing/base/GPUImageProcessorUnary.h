#ifndef DSM_GPU_IMAGE_PROCESSOR_UNARY_H
#define DSM_GPU_IMAGE_PROCESSOR_UNARY_H

#include "GPUBaseImageProcessor.h"

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types

/* Base class for defining image processing operations which take
   one image as input and one image as output 
 */

namespace dsm {
class GPUImageProcessorUnary : public GPUBaseImageProcessor{
public:
	//actual functionality of the image processor, override in subclass
	virtual void process(cl_command_queue const& command_queue, 
						 cl_mem const& in_image_buffer, cl_mem& out_image_buffer) = 0;

	virtual ~GPUImageProcessorUnary();


protected:
	GPUImageProcessorUnary(cv::Vec2i const& image_dims);
};

} //namespace dsm
#endif //DSM_GPU_IMAGE_PROCESSOR_UNARY_H