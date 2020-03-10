#include "GPUImageProcessorUnary.h"

namespace dsm {
	GPUImageProcessorUnary::GPUImageProcessorUnary(cv::Vec2i const& image_dims) : GPUBaseImageProcessor{image_dims}{
	}

	GPUImageProcessorUnary::~GPUImageProcessorUnary() {
	}
}