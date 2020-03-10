#ifndef DSM_STEREO_CAMERA_H
#define DSM_STEREO_CAMERA_H

#include <BSystem.h>
#include <BCamera.h>

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types
#include <mutex>

/* Derived class for defining 3D reconstruction operations which take
   two images (1 disparity and 1 color) as input and creates 2 output buffers (3x float xyz and 3x unsigned char rgb)
 */


namespace dsm {

struct camera_descriptor_t {
	int width = 0;
	int height = 0;
	int num_channels = 0;
};

class StereoCamera {
public:
    StereoCamera() = default;

	virtual ~StereoCamera() {};
	void init(bool init_rgb = false);
	void capture_images();
	std::vector<std::vector<uint8_t> > get_images();
	uint32_t get_cam_width() const;
	uint32_t get_cam_height() const;

	void set_gain(float gain);

	void set_exposure_time(float exposure_time);
private:
	std::vector<camera_descriptor_t> camera_descriptors_;
	std::shared_ptr<baumer::BSystem> baumer_system_  = nullptr;
    std::vector<baumer::BCamera*> detected_baumer_cams_;

    std::vector<std::vector<uint8_t> > captured_image_buffers_;
    std::vector<std::vector<uint8_t> > captured_image_buffers_back_;

    uint32_t num_baumer_cams_ = 0;
    uint32_t cam_width_ = 0;
    uint32_t cam_height_ = 0;

    std::shared_ptr<std::mutex> image_capturing_mutex_ = nullptr;
};

} //namespace dsm

#endif //DSM_STEREO_CAMERA_H