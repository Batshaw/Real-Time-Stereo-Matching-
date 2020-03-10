#include "rgbd_stream.h"

#include <exception>
#include <iostream>
namespace dsm {

rgbd_stream::rgbd_stream(int num_cameras, cv::Vec2i const& rgb_res, cv::Vec2i const& depth_res) :
num_cameras_(num_cameras), rgb_image_resolution_(rgb_res), depth_image_resolution_(depth_res) {

	// RGB8 -> 3 bytes per pixel of a color image
	num_bytes_per_color_image_ = 3 * sizeof(char) * rgb_image_resolution_[0] * rgb_image_resolution_[1];
	num_bytes_per_depth_image_ =     sizeof(float) * depth_image_resolution_[0] * depth_image_resolution_[1];

	num_bytes_per_frame_ = num_cameras_ * (num_bytes_per_depth_image_ + num_bytes_per_color_image_);

	//allocate buffer for stroring exactly one complete point in time (frame) in
	current_frame_.resize(num_bytes_per_frame_);
}

void
rgbd_stream::open_stream(std::string const& path_to_stream_file) {
	//open stream file in as input file(std::ios::in), binary mode (std::ios::binary) and at the end (std::ios::ate)
	rgbd_stream_handle_ = std::make_unique<std::ifstream>(path_to_stream_file, std::ios::in | std::ios::binary | std::ios::ate);


	if(!rgbd_stream_handle_->is_open() ) {
		std::cout << "Could not open rgbd_stream. Please check the file path.\n";
		throw std::exception();
	}

	std::size_t total_num_bytes_in_stream = rgbd_stream_handle_->tellg();

	//reset file cursor to zero
	rgbd_stream_handle_->seekg(0);

	total_num_frames_in_stream_ = total_num_bytes_in_stream / num_bytes_per_frame_;
}



void 
rgbd_stream::close_stream() {
	if(!rgbd_stream_handle_->is_open() ) {
		std::cout << "Stream to be closed was not open.\n";
		throw std::exception();
	}

	rgbd_stream_handle_->close();
}

char* 
rgbd_stream::read_frame(std::size_t frame_idx) {

	if(frame_idx >= total_num_frames_in_stream_) {
		std::cout << "Frame idx " << frame_idx << "is invalid. Returning frame 0\n";
		frame_idx = 0;
	}

	std::size_t requested_frame_byte_offset = frame_idx * num_bytes_per_frame_;

	//set read cursor to computed offset
	rgbd_stream_handle_->seekg(requested_frame_byte_offset);

	rgbd_stream_handle_->read(current_frame_.data(), num_bytes_per_frame_);

	return current_frame_.data();
}

}