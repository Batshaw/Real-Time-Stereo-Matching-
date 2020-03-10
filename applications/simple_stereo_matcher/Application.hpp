#ifndef SIMPLE_STEREO_MATCHING_APPLICATION
#define SIMPLE_STEREO_MATCHING_APPLICATION

#include <image_processing/GPUImageProcessors.h>

#include <image_processing/binary/GPUPatchMatchStereoMatcher.h>

#include <core/init_opencl.h>
#include <core/utils.h>

#include <core/stereo_camera.h>

#include <io/GeometrySender.h>

#include <CL/cl.h> //command_queues, devices, ...

// OpenCV 
#include <opencv2/core/core.hpp> //types
#include <opencv2/imgcodecs.hpp> //imread, imwrite, color formats
#include <opencv2/highgui/highgui.hpp> //gui elements, window
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <boost/program_options.hpp> 

#include <GLFW/glfw3.h>
#include <GL/glu.h>

// standard header
#include <iostream>
#include <string>
#include <algorithm>
#include <iostream>
#include <image_processing/binary/GPUAdCensusStereoMatcher.h>



namespace po = boost::program_options;

struct StereoInfo
{
    cv::Mat R, T, R1, R2, P1, P2, Q;
    cv::Mat cam_mats[2], dist_coeffs[2];
    cv::Rect valid_roi[2];
    cv::Mat rmap[2][2];

    double base_line; //in meter
    double focal_length; //in meter
};

class Application {
public:
    enum STEREO_METHOD{
            PATCH_MATCH = 1,
            NCC = 2,
            AD_CENSUS = 3
    };
  Application(int argc, char** argv);
  ~Application();
  
  void define_command_line_arguments();
  // gui layout definition
  void define_GUI(STEREO_METHOD str_method);

  void process_frame();
  void visualize_geometry();
  void visualize_cv_window();
  void run_mainloop();

  void send_data();
  void initialize_cl_buffers(int image_width, int image_height, int num_channels, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2);
  void initialize_cl_buffers(int image_width, int image_height, int num_channels, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2, unsigned char* rgb_data_im_3, unsigned char* rgb_data_im_4);

  void load_camera_parameters(const std::string &path, uint32_t img_width, uint32_t img_height);
  void rectify_images();

  // forward declaration of GUI events. definition is at the end of the file

  //static void on_button_SAD_green_similarity(int state, void* userdata);
  //static void on_button_SAD_grayscale_similarity(int state, void* userdata);
  //static void on_button_SAD_grayscale_local_memory_similarity(int state, void* userdata);


  //static void on_button_SAD_lab_32f_similarity(int state, void* userdata);

  //static void on_button_ASW_lab_similarity(int state, void* userdata);


  // CV TRACKBAR EVENTS
  static void on_search_window_size_trackbar(int state, void* userdata);
  static void on_minimum_disparity_trackbar(int state, void* userdata);
  static void on_maximum_disparity_trackbar(int state, void* userdata);
  static void on_disparity_vis_scaling_trackbar(int state, void* userdata);
  //static void on_focal_length_trackbar(int state, void* userdata);
  static void on_distance_to_cam_trackbar(int state, void* userdata);

  static void on_center_of_rotatio_distance(int state, void* userdata);
  static void on_pm_iterations_trackbar(int state, void* userdata);
  static void on_pm_temporal_propagation(int state, void* userdata);
  static void on_pm_view_propagation(int state, void* userdata);
  static void on_pm_plane_refine_steps(int state, void* userdata);
  static void on_pm_slanted_or_fronto(int state, void* userdata);
  static void on_gain_offset_trackbar(int state, void* userdata);

  static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

  static void onMouse(int button, int state, int x, int y);

  static void onMotion(int x, int y);



  static uint32_t window_width;// = 3200;
  static uint32_t window_height;// = 2400;  


private:
    STEREO_METHOD m_stereo_method = STEREO_METHOD::NCC;


	uint32_t m_original_image_width = 0;
	uint32_t m_original_image_height = 0;
	uint32_t m_original_image_num_channels = 0;

	uint32_t m_downsampled_image_width = 0;
	uint32_t m_downsampled_image_height = 0;

	uint32_t m_use_distance_cut_off = 1;

	int32_t rot_vertical = 0;
	int32_t rot_horizontal = 0;

    StereoInfo _stereo_info;

    static int stereo_rig_gain_offset ;

    int distance_to_vcam_cm = 121;

	float last_recorded_timestamp = 0.0f;
	static bool m_accumulate_time;
	float accumulated_time = 0.0f;

	GLFWwindow* window = nullptr;

	cl_mem cl_input_buffer_1 = 0;
	cl_mem cl_input_buffer_2 = 0;
	cl_mem cl_input_buffer_3 = 0; //good initial guess input
	cl_mem cl_input_buffer_4 = 0; //good initial guess input

	cl_mem cl_disparity_image_output_buffer = 0;



	cl_mem cl_point_output_buffer = 0;
	cl_mem cl_color_output_buffer = 0;

	cl_context context = 0;
	cl_device_id device_id = 0;
	cl_command_queue command_queue = 0;
	cl_kernel kernel = 0;

	// global objects which will help us load and save images
	cv::Mat input_image_1_bgr_1x8;
	cv::Mat input_image_2_bgr_1x8;
	cv::Mat input_image_3_bgr_1x8;
	cv::Mat input_image_4_bgr_1x8;

	cv::Mat input_image_1_bgr_3x8;
	cv::Mat input_image_2_bgr_3x8;
	cv::Mat input_image_3_bgr_3x8; //initial guess input
	cv::Mat input_image_4_bgr_3x8; //initial guess input

    cv::UMat input_image_1_bgr_3x8_cl;
    cv::UMat input_image_2_bgr_3x8_cl;
    cv::Mat input_image_1_bgr_3x8_rectified;
    cv::Mat input_image_2_bgr_3x8_rectified;
	cv::Mat output_image_grayscale_1x8;
	cv::Mat output_image_grayscale_1x32f;
	// reference disparity image we may load
	cv::Mat reference_disparity_image_3x8;



	// some parameters limiting the amount of 
	int const MAX_CONFIGURABLE_WINDOW_SIZE = 30;
	int const MIN_CONFIGURABLE_DISPARITY = 0;
	int const MAX_CONFIGURABLE_DISPARITY = 255;
	int const MAX_CONFIGURABLE_DISPARITY_VIS_SCALING = 10;


	po::options_description desc{"Options"};

  	// if no other image is specified, ./images/Flowerpots/view1.png
	
	std::string m_image_1_path = "./images/Teddy/im2.png"; 
	std::string m_image_2_path = "./images/Teddy/im6.png"; 
	std::string m_image_3_path = "./images/Teddy/disp2_n.png"; //noisy ground truth
	std::string m_image_4_path = "./images/Teddy/disp6_n.png"; //noisy ground truth
    std::string reference_disparity_image_path = "";

    std::string m_output_ply_path = "./out.ply";

	std::string const& m_window_name = "Simple Stereo Matching";
	
	int m_window_size_x = 0;
	int m_window_size_y = 0;


	bool m_show_gui = false;
    bool m_use_adcensus = false;
	bool m_use_patch_match = false;
	bool m_input_initial_guess_3 = false;
	bool m_input_initial_guess_4 = false;
	bool m_camera_parameters_available = false;

	bool m_use_stereo_camera = false;
	bool m_run_cameras_in_color_mode    = false;


	bool m_are_images_downsampled = false;
    float m_downsampling_factor = 1.0f;

    dsm::GPUAdCensusStereoMatcher::Params m_param_adcensus;
	std::shared_ptr<dsm::GPUStereoMatcher> stereo_matcher_ptr = nullptr;
    std::shared_ptr<dsm::GPUAdCensusStereoMatcher> stereo_ad_census_matcher_ptr = nullptr;
	std::shared_ptr<dsm::GPUStereoMatcherNCC> stereo_matcher_NCC_ptr = nullptr;
	std::shared_ptr<dsm::GPUPatchMatchStereoMatcher> stereo_PATCH_MATCH_matcher__ptr = nullptr;

	std::shared_ptr<dsm::GPUPointcloudReconstructor> reconstructor_ptr = nullptr;

	dsm::ReconstructionMode active_reconstruction_mode = dsm::ReconstructionMode::FLOAT_DISPARITY_TO_VERTEX_COLOR_TRIANGLES;

	static std::shared_ptr<dsm::StereoCamera> stereo_camera_ptr_;

	bool m_send_geometry_data = false;
	bool m_show_model = true;
	std::shared_ptr<dsm::sys::GeometrySender> geometry_sender_ptr = nullptr;

	std::vector<xyz_rgb_point> xyz_rgb_points_for_ply_writer;
	std::vector<float> point_positions_vec; 
	std::vector<unsigned char> point_color_vec;

	std::vector<unsigned char> m_combined_geometry_and_texture_data;

	static dsm::StereoMatchingMode active_stereo_matching_mode;
	static bool m_changed_matching_state;
	static int m_search_window_half_size;
	static int m_minimum_disparity;
	static int m_maximum_disparity;
	static int m_disparity_vis_scaling;
	float m_focal_length = 600.0f;
	float m_baseline = 0.10f;
	static int m_center_of_rotation_distance_cm;
	static int m_num_iteration_propagation; // for PatchMatch
	static int m_num_temporal_propagation; //  for PatchMatch
	static int m_switch_view_prop;
	static int m_switch_outlier_detection; //  for PatchMatch
	static int m_plane_refine_steps;
	static int m_slanted_or_fronto;
};

#endif
