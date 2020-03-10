#include "GPUImageProcessorBinary.h"

namespace dsm {

GPUImageProcessorBinary::GPUImageProcessorBinary(cv::Vec2i const& image_dims) 
	: GPUBaseImageProcessor(image_dims) {
}

GPUImageProcessorBinary::~GPUImageProcessorBinary() {}

}