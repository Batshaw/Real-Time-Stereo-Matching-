#include "GPU3DReconstructor.h"

namespace dsm {

GPU3DReconstructor::GPU3DReconstructor(cv::Vec2i const& image_dims) 
	: GPUBaseImageProcessor(image_dims) {
}

GPU3DReconstructor::~GPU3DReconstructor() {}

void GPU3DReconstructor::set_baseline(float baseline) {
	baseline_ = baseline;
}
void GPU3DReconstructor::set_focal_length(float focal_length) {
	focal_length_ = focal_length;
}

void GPU3DReconstructor::set_disparity_scaling(float disparity_scaling) {
	disparity_scaling_ = disparity_scaling;
}

void GPU3DReconstructor::set_min_valid_disparity(int min_valid_disparity) {
	min_valid_disparity_ = min_valid_disparity;
}

void GPU3DReconstructor::set_enable_distance_cut_off(bool enable_distance_cut_off) {
	use_distance_cut_off_ = enable_distance_cut_off;
}

}