#ifndef DSM_RGBD_STREAM_H
#define DSM_RGBD_STREAM_H

#include <opencv2/opencv.hpp>

#include <fstream> //ofstream
#include <string>  //strings
#include <memory>  //smart ptr
#include <vector>


namespace dsm {

class rgbd_stream{
public:
	//default constructor & destructor
	rgbd_stream(int num_cameras, 
				cv::Vec2i const& rgb_image_resolution, 
				cv::Vec2i const& depth_image_resolution);
	~rgbd_stream() = default;

	//opens stream from *.stream file and computes num frames based on file size
	void open_stream(std::string const& path_to_stream_file);
	void close_stream();

	char* read_frame(std::size_t frame_idx);

private:
	//num devices used to record stream such as realsense or kinect
	int num_cameras_ = 0;
	cv::Vec2i rgb_image_resolution_{-1, -1};
	cv::Vec2i depth_image_resolution_{-1, -1};

	std::size_t num_bytes_per_color_image_ = 0;
	std::size_t num_bytes_per_depth_image_ = 0;
	std::size_t num_bytes_per_frame_ = 0;
	std::size_t total_num_frames_in_stream_ = 0;

	std::unique_ptr<std::ifstream> rgbd_stream_handle_ = nullptr;

	//contains data of all cameras for rgb and depth images for one point in time
	std::vector<char> current_frame_;
//	void load_entire_stream_to_memory();
//	void flush_memory();

};

}

#endif //DSM_RGBD_STREAM_H