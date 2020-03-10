#include <stereo_camera.h>

#include<mutex>

namespace dsm {

void StereoCamera::init(bool init_rgb) {

  image_capturing_mutex_ = std::make_shared<std::mutex>();

  baumer_system_ = std::make_shared<baumer::BSystem>();

  baumer_system_->init();

   num_baumer_cams_ = baumer_system_->getNumCameras();

   camera_descriptors_.resize(num_baumer_cams_);
   detected_baumer_cams_.resize(num_baumer_cams_, nullptr);

   captured_image_buffers_.resize(num_baumer_cams_, std::vector<uint8_t>());
   captured_image_buffers_back_.resize(num_baumer_cams_, std::vector<uint8_t>());

  for(uint32_t camera_idx = 0; camera_idx < num_baumer_cams_; ++camera_idx) {
    detected_baumer_cams_[camera_idx] = baumer_system_->getCamera(camera_idx, init_rgb);

  	auto& current_cam_descriptor = camera_descriptors_[camera_idx];
  	current_cam_descriptor.width        = detected_baumer_cams_[camera_idx]->getWidth();
  	current_cam_descriptor.height       = detected_baumer_cams_[camera_idx]->getHeight();
  	current_cam_descriptor.num_channels = detected_baumer_cams_[camera_idx]->getNumChannels();

    uint32_t current_image_buffer_size =   current_cam_descriptor.width
                                         * current_cam_descriptor.height
                                         * current_cam_descriptor.num_channels;
    
  captured_image_buffers_[camera_idx].resize(current_image_buffer_size);
	captured_image_buffers_back_[camera_idx].resize(current_image_buffer_size);

	//hack for now
	cam_width_ = current_cam_descriptor.width;
	cam_height_ = current_cam_descriptor.height;
  }

}

void StereoCamera::capture_images() {

	std::unique_lock<std::mutex> cam_capturing_lock{*image_capturing_mutex_};

  	for(uint32_t cam_idx = 0; cam_idx < num_baumer_cams_; ++cam_idx) {
        //std::cout << " Trying to memcpy " << baumer_image_buffers[0].size() << "\n";
        boost::mutex::scoped_lock individual_camera_lock(detected_baumer_cams_[cam_idx]->getMutexLock());
        if(0 != detected_baumer_cams_[cam_idx]->capture()){
            memcpy(captured_image_buffers_back_[cam_idx].data(), detected_baumer_cams_[cam_idx]->capture(), captured_image_buffers_back_[cam_idx].size());
        }
    }

    std::swap(captured_image_buffers_back_, captured_image_buffers_);
}

std::vector<std::vector<uint8_t> > StereoCamera::get_images() {
	std::unique_lock<std::mutex> cam_capturing_lock{*image_capturing_mutex_};
	return captured_image_buffers_;
}

void StereoCamera::set_gain(float in_gain) {
  float clamped_gain = std::max(1.0f, std::min(10.0f, in_gain));

  for(uint32_t camera_idx = 0; camera_idx < num_baumer_cams_; ++camera_idx) {

    detected_baumer_cams_[camera_idx]->setGain(clamped_gain);
  }
}

void StereoCamera::set_exposure_time(float exposure_time) {
  for(uint32_t camera_idx = 0; camera_idx < num_baumer_cams_; ++camera_idx) {

    detected_baumer_cams_[camera_idx]->setExposureTime(exposure_time);
  } 
}

uint32_t StereoCamera::get_cam_width() const {
  return cam_width_;
}
uint32_t StereoCamera::get_cam_height() const {
  return cam_height_;
}

}