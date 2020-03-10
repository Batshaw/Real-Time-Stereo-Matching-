#include "Application.hpp"


#include <sgtp/SGTP.h>

#include <chrono>

//glm::mat4 view = glm::lookAt( glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0., 0., 0.), glm::vec3(0., 1., 0.) );


// initilization of static variables
dsm::StereoMatchingMode Application::active_stereo_matching_mode = dsm::StereoMatchingMode::SIMPLE_CENSUS_GRAYSCALE_1x8;
bool Application::m_changed_matching_state = true;


int Application::m_search_window_half_size = 5;
int Application::m_minimum_disparity = 0;
int Application::m_maximum_disparity = 60;
int Application::m_disparity_vis_scaling = 4;
int Application::m_switch_outlier_detection= 0;

int Application::m_num_iteration_propagation = 4;
int Application::m_num_temporal_propagation = 1;
int Application::m_switch_view_prop = 1;
int Application::m_plane_refine_steps = 2;
int Application::m_slanted_or_fronto = 0;

int Application::m_center_of_rotation_distance_cm = 173;

bool Application::m_accumulate_time = false;

int Application::stereo_rig_gain_offset = 0;

std::shared_ptr<dsm::StereoCamera> Application::stereo_camera_ptr_ = nullptr;

uint32_t Application::window_width = 800;
uint32_t Application::window_height = 600;  



int last_mx = 0, last_my = 0, cur_mx = 0, cur_my = 0;
int arcball_on = false;

Application::Application(int argc, char** argv) {

    std::string const usage_line = std::string("USAGE: ") + argv[0] + " <kernel_source_filename> [options] (-h or --help for more info)\n";


    define_command_line_arguments();

    po::variables_map vm; 

    po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw 

    if ( vm.count("help")  ) { 
      std::cout << usage_line << std::endl
                << desc << std::endl; 
      exit(-1);
    } 

    if ( !vm["image_1"].empty() && !vm["image_2"].empty()) { 
      m_image_1_path = vm["image_1"].as<std::string>();
      m_image_2_path = vm["image_2"].as<std::string>();
      //          << desc << std::endl; 
      //return 0; 
    } else {
        std::cout << "Image paths were not specified. Using default images: \n"; 
        std::cout << m_image_1_path << "\n";
        std::cout << m_image_2_path << "\n";
    }

    m_input_initial_guess_3 = !vm["input-initial-guess-3"].empty(); 
    m_input_initial_guess_4 = !vm["input-initial-guess-4"].empty(); 
    //std::cout << "Initial Guess Input 3 empty?  " << m_input_initial_guess_3 << "\n"; 
    //std::cout << "Initial Guess Input 4 empty?  " << m_input_initial_guess_4 << "\n"; 
    if ( m_input_initial_guess_3 && m_input_initial_guess_4){
      m_image_3_path = vm["input-initial-guess-3"].as<std::string>();
      m_image_4_path = vm["input-initial-guess-4"].as<std::string>();

      std::cout << "Initial guess images: \n"; 
      std::cout << m_image_3_path << "\n";
      std::cout << m_image_4_path << "\n";
    }

    if ( !vm["reference"].empty()) {
      reference_disparity_image_path = vm["reference"].as<std::string>();

      reference_disparity_image_3x8 = cv::imread(reference_disparity_image_path.c_str(), cv::IMREAD_COLOR);
    }

    if ( !vm["output_path"].empty()) {
      m_output_ply_path = vm["output_path"].as<std::string>();
    }

    m_show_gui = vm["no-gui"].empty();
    //load image


    

    std::string const downsampling_factor_cmd_line_name = "downsampling_factor";

    m_are_images_downsampled = !vm[downsampling_factor_cmd_line_name].empty();
    if(m_are_images_downsampled) {
      m_downsampling_factor = vm[downsampling_factor_cmd_line_name].as<float>();
    }

    m_send_geometry_data = !vm["tcp_socket_out"].empty();

    if(m_send_geometry_data) {
      std::string send_socket_port_as_string = vm["tcp_socket_out"].as<std::string>();
      geometry_sender_ptr = std::make_shared<dsm::sys::GeometrySender>(send_socket_port_as_string.c_str());

      m_show_model = false;
    }

    m_use_stereo_camera = !vm["use_stereo_camera"].empty();
    if(m_use_stereo_camera) {
      stereo_camera_ptr_ = std::make_shared<dsm::StereoCamera>(dsm::StereoCamera());


      m_run_cameras_in_color_mode = m_send_geometry_data = !vm["color"].empty();
      stereo_camera_ptr_->init(m_run_cameras_in_color_mode);


      m_original_image_width = stereo_camera_ptr_->get_cam_width();
      m_original_image_height = stereo_camera_ptr_->get_cam_height(); 



      input_image_1_bgr_1x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC1);
      input_image_2_bgr_1x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC1);

      cv::cvtColor(input_image_1_bgr_1x8, input_image_1_bgr_3x8, cv::COLOR_GRAY2BGR);
      cv::cvtColor(input_image_2_bgr_1x8, input_image_2_bgr_3x8, cv::COLOR_GRAY2BGR);

      input_image_1_bgr_3x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC3);
      input_image_2_bgr_3x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC3);

      if(m_input_initial_guess_3 && m_input_initial_guess_4){
        input_image_3_bgr_1x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC1);
        input_image_4_bgr_1x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC1);
        cv::cvtColor(input_image_3_bgr_1x8, input_image_3_bgr_3x8, cv::COLOR_GRAY2BGR);
        cv::cvtColor(input_image_4_bgr_1x8, input_image_4_bgr_3x8, cv::COLOR_GRAY2BGR);

        input_image_3_bgr_3x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC3);
        input_image_4_bgr_3x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC3);
      }


      m_original_image_num_channels = input_image_1_bgr_3x8.channels();



      m_downsampled_image_width = m_downsampling_factor * m_original_image_width;
      m_downsampled_image_height = m_downsampling_factor * m_original_image_height;
    } else {
      input_image_1_bgr_3x8 = cv::imread(m_image_1_path.c_str(), cv::IMREAD_COLOR);
      input_image_2_bgr_3x8 = cv::imread(m_image_2_path.c_str(), cv::IMREAD_COLOR);
      if(m_input_initial_guess_3 && m_input_initial_guess_4){
        input_image_3_bgr_3x8 = cv::imread(m_image_3_path.c_str(), cv::IMREAD_COLOR);
        input_image_4_bgr_3x8 = cv::imread(m_image_4_path.c_str(), cv::IMREAD_COLOR);
        //  input_image_3_bgr_1x8 = cv::imread(m_image_3_path.c_str(), cv::IMREAD_GRAYSCALE);
        //  input_image_4_bgr_1x8 = cv::imread(m_image_4_path.c_str(), cv::IMREAD_GRAYSCALE);        
      }

      m_original_image_width = input_image_1_bgr_3x8.cols;
      m_original_image_height = input_image_1_bgr_3x8.rows;
      m_original_image_num_channels = input_image_1_bgr_3x8.channels();

      if(m_are_images_downsampled) {

        m_downsampled_image_width = m_downsampling_factor * m_original_image_width;
        m_downsampled_image_height = m_downsampling_factor * m_original_image_height;   

        /*
          std::cout << "Downsampling from [" << m_original_image_width << ", " << m_original_image_height << "] pixels to [" 
                                             << m_downsampled_image_width << ", " << m_downsampled_image_height << "] pixels" << std::endl; 
        */

        //in place conversion
        cv::resize(input_image_1_bgr_3x8, 
                   input_image_1_bgr_3x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
                   0, 0, cv::INTER_LINEAR);
        cv::resize(input_image_2_bgr_3x8, 
                   input_image_2_bgr_3x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
                   0, 0, cv::INTER_LINEAR);
        if(m_input_initial_guess_3 && m_input_initial_guess_4){
            cv::resize(input_image_3_bgr_3x8, 
                      input_image_3_bgr_3x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
                      0, 0, cv::INTER_LINEAR);
            cv::resize(input_image_4_bgr_3x8, 
                      input_image_4_bgr_3x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
                      0, 0, cv::INTER_LINEAR);
            // cv::resize(input_image_3_bgr_1x8, 
            //           input_image_3_bgr_1x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
            //           0, 0, cv::INTER_LINEAR);
            // cv::resize(input_image_4_bgr_1x8, 
            //           input_image_4_bgr_1x8, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
            //           0, 0, cv::INTER_LINEAR);           
        }
        
      } else {
        m_downsampled_image_width = m_downsampling_factor * m_original_image_width;
        m_downsampled_image_height = m_downsampling_factor * m_original_image_height;   
      }

    }



    //input_image_1_bgr_3x8_cl = input_image_1_bgr_3x8.getUMat(cv::ACCESS_READ);
    //input_image_2_bgr_3x8_cl = input_image_2_bgr_3x8.getUMat(cv::ACCESS_READ);

    
    uint32_t max_num_points = m_downsampled_image_width*m_downsampled_image_height;
    point_positions_vec.resize(max_num_points * 3 * 3 * 2);
    point_color_vec.resize(max_num_points * 3 * 3 * 2);

    

    m_window_size_x = m_original_image_width;
    m_window_size_y = m_original_image_height;

    std::vector<int> window_size;
    if (!vm["win_size"].empty() && (window_size = vm["win_size"].as<std::vector<int> >()).size() == 2) {
      // valid window size
        m_window_size_x = window_size[0];
        m_window_size_y = window_size[1];       
    }

    m_camera_parameters_available = false;
    std::string path_to_camera_parameters = "";

    if(!vm["camera"].empty()) {
    	path_to_camera_parameters = vm["camera"].as<std::string>();
        m_camera_parameters_available = true;
    }
    
    if(m_camera_parameters_available) {
      load_camera_parameters(vm["camera"].as<std::string>(), m_original_image_width, m_original_image_height);
    }

    m_use_adcensus = !vm["use_adcensus"].empty();
    m_use_patch_match = !vm["use_patch_match"].empty();

    if (m_use_adcensus)
        m_stereo_method = STEREO_METHOD::AD_CENSUS;
    else if(m_use_patch_match)
        m_stereo_method = STEREO_METHOD::PATCH_MATCH;
    else
        m_stereo_method = STEREO_METHOD::NCC;
 
    std::vector<int> iter_and_disp_data;
    if ( !vm["use_patch_match"].empty() && (iter_and_disp_data = vm["use_patch_match"].as<std::vector<int> >()).size() == 6) {
        m_num_iteration_propagation = iter_and_disp_data[0];
        m_minimum_disparity = iter_and_disp_data[1];
        m_maximum_disparity = iter_and_disp_data[2];
        m_search_window_half_size = iter_and_disp_data[3];
        m_plane_refine_steps = iter_and_disp_data[4];
        m_slanted_or_fronto = iter_and_disp_data[5];
        //default temp_prop & view == OFF / 1
    }
    if ( !vm["use_patch_match"].empty() && (iter_and_disp_data = vm["use_patch_match"].as<std::vector<int> >()).size() == 8) {
        m_num_iteration_propagation = iter_and_disp_data[0];
        m_minimum_disparity = iter_and_disp_data[1];
        m_maximum_disparity = iter_and_disp_data[2];
        m_search_window_half_size = iter_and_disp_data[3];
        m_plane_refine_steps = iter_and_disp_data[4];
        m_slanted_or_fronto = iter_and_disp_data[5];
        m_switch_view_prop = iter_and_disp_data[6]; // 1 == OFF , 0 == ON 
        m_num_temporal_propagation = iter_and_disp_data[7]; // 1 == OFF , 0 == ON
        
    }
    //3x8 unsigned character per pixel

    if(!m_are_images_downsampled) {
      output_image_grayscale_1x8 = cv::Mat(m_original_image_height, m_original_image_width, CV_8UC1);
      output_image_grayscale_1x32f = cv::Mat(m_original_image_height, m_original_image_width, CV_32FC1);
    } else {
      output_image_grayscale_1x8 = cv::Mat(m_downsampled_image_height, m_downsampled_image_width, CV_8UC1);
      output_image_grayscale_1x32f = cv::Mat(m_downsampled_image_height, m_downsampled_image_width, CV_32FC1);     
    }

    if(m_show_gui) {
	    if (!glfwInit())
	    {
	      std::cout << "GLFW Initialization Failed! Exiting!\n";
	        // Initialization failed
	      exit(-1);
	    }


	    glfwWindowHint(GLFW_DEPTH_BITS, 24);
	    window = glfwCreateWindow(window_width, window_height, "PointCloud Output", NULL, NULL);


	    glfwMakeContextCurrent(window);


      glfwSetKeyCallback(window, glfw_key_callback);


	}

    dsm::initialize_cl_environment(context, device_id, command_queue, false);

  
    if(!m_are_images_downsampled) {
      if(m_input_initial_guess_3 && m_input_initial_guess_4){
        initialize_cl_buffers(m_original_image_width, m_original_image_height, m_original_image_num_channels, input_image_1_bgr_3x8.data, input_image_2_bgr_3x8.data, input_image_3_bgr_3x8.data, input_image_4_bgr_3x8.data);

      }else{
        initialize_cl_buffers(m_original_image_width, m_original_image_height, m_original_image_num_channels, input_image_1_bgr_3x8.data, input_image_2_bgr_3x8.data);
      }
    } else {
      if(m_input_initial_guess_3 && m_input_initial_guess_4){
        initialize_cl_buffers(m_downsampled_image_width, m_downsampled_image_height, m_original_image_num_channels, input_image_1_bgr_3x8.data, input_image_2_bgr_3x8.data, input_image_3_bgr_3x8.data, input_image_4_bgr_3x8.data);
      }else{
        initialize_cl_buffers(m_downsampled_image_width, m_downsampled_image_height, m_original_image_num_channels, input_image_1_bgr_3x8.data, input_image_2_bgr_3x8.data);
      }
      
    }

    {
        define_GUI(m_stereo_method);
    }

    if(!m_are_images_downsampled) {

    reconstructor_ptr = dsm::GPUPointcloudReconstructor::create(context, device_id,
                                                                cv::Vec2i{m_original_image_width, 
                                                                          m_original_image_height},
                                                                          active_reconstruction_mode);
    } else {
      reconstructor_ptr = dsm::GPUPointcloudReconstructor::create(context, device_id,
                                                                  cv::Vec2i{m_downsampled_image_width, 
                                                                            m_downsampled_image_height},
                                                                            active_reconstruction_mode);      
    }

/*
    reconstructor_ptr->set_baseline(0.0475);
    reconstructor_ptr->set_focal_length(m_downsampling_factor * 950.0);
    reconstructor_ptr->set_disparity_scaling(1.0);

    std::cout << "Before mainloop" << std::endl;
*/
    if (m_camera_parameters_available){
        m_baseline = _stereo_info.base_line;
        m_focal_length = m_downsampling_factor * _stereo_info.focal_length;
        reconstructor_ptr->set_baseline(m_baseline);
        reconstructor_ptr->set_focal_length(m_focal_length);
        reconstructor_ptr->set_disparity_scaling(1.0);
        m_use_distance_cut_off = 1;
    }
    else{
        m_baseline = 0.1;
        m_focal_length = m_downsampling_factor * 600.0;
        reconstructor_ptr->set_baseline(m_baseline);
        reconstructor_ptr->set_focal_length(m_focal_length);
        reconstructor_ptr->set_disparity_scaling(1.0);
        m_use_distance_cut_off = 0;
    }

    // blocking in GUI mode until escape is pressed
    run_mainloop();


}

Application::~Application() {

    int status = 0;
    // Clean the app resources.
    status = clReleaseMemObject(cl_input_buffer_1); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_2); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_3); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_4); //Release mem object.
    status = clReleaseMemObject(cl_disparity_image_output_buffer);

    status = clReleaseMemObject(cl_point_output_buffer);
    status = clReleaseMemObject(cl_color_output_buffer);

    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL) {
        free(device_id);
        device_id = NULL;
    }
}


// loads 
void Application::initialize_cl_buffers(int image_width, int image_height, int num_channels, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2, unsigned char* rgb_data_im_3, unsigned char* rgb_data_im_4) {

  /* 3x8 bgr, flag CL_MEM_READ_ONLY (since these are input images) and tell cl to fill the buffer with host data
     rgb_data_im_1 and rgb_data_im_2, respectively (CL_MEM_COPY_HOST_PTR). 

     The function needs to know how many byte it should allocate (here: width*height*3). The factor 3 is because we have
     3x 8bit for the red, green and blue channel of the input images
  */

  std::cout << "Initializing cl buffers with width and height: " << image_width << ", " << image_height << std::endl;

  cl_input_buffer_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_1, NULL);
  cl_input_buffer_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_2, NULL);
  cl_input_buffer_3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_3, NULL);
  cl_input_buffer_4 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_4, NULL);
  // 1x8 8 bit out image, flag WRITE_ONLY (since this is an output image) and tell cl to not fill the buffer with any data
  // stores unsigned integer CL_MEM_WRITE_ONLY up to 255 (= 2^8)

  cl_disparity_image_output_buffer =  clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                                     image_width * image_height * 1 * sizeof(float), NULL, NULL);

  //output of 3D reconstruction
  //position = 3 floats
  cl_point_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                           3 * 2 * (image_width-1) * (image_height-1) * 3 * sizeof(float), NULL, NULL);

  //color = 3 unsigned chars
  cl_color_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                           3 * 2 * (image_width-1) * (image_height-1) * num_channels * sizeof(unsigned char), NULL, NULL);
}

void Application::initialize_cl_buffers(int image_width, int image_height, int num_channels, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2) {

  /* 3x8 bgr, flag CL_MEM_READ_ONLY (since these are input images) and tell cl to fill the buffer with host data
     rgb_data_im_1 and rgb_data_im_2, respectively (CL_MEM_COPY_HOST_PTR). 

     The function needs to know how many byte it should allocate (here: width*height*3). The factor 3 is because we have
     3x 8bit for the red, green and blue channel of the input images
  */

  std::cout << "Initializing cl buffers with width and height: " << image_width << ", " << image_height << std::endl;

  cl_input_buffer_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_1, NULL);
  cl_input_buffer_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), (void *)rgb_data_im_2, NULL);
  cl_input_buffer_3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), NULL, NULL);
  cl_input_buffer_4 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels * sizeof(char), NULL, NULL);;
  // 1x8 8 bit out image, flag WRITE_ONLY (since this is an output image) and tell cl to not fill the buffer with any data
  // stores unsigned integer CL_MEM_WRITE_ONLY up to 255 (= 2^8)

  cl_disparity_image_output_buffer =  clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                                     image_width * image_height * 1 * sizeof(float), NULL, NULL);

  //output of 3D reconstruction
  //position = 3 floats
  cl_point_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                           3 * 2 * (image_width-1) * (image_height-1) * 3 * sizeof(float), NULL, NULL);

  //color = 3 unsigned chars
  cl_color_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                           3 * 2 * (image_width-1) * (image_height-1) * num_channels * sizeof(unsigned char), NULL, NULL);
}

void Application::process_frame() {



  if(m_use_stereo_camera) { 
    stereo_camera_ptr_->capture_images();
    auto image_buffers = stereo_camera_ptr_->get_images();

    if(m_run_cameras_in_color_mode) {
      memcpy(input_image_1_bgr_3x8.data, image_buffers[0].data(), image_buffers[0].size() );
      memcpy(input_image_2_bgr_3x8.data, image_buffers[1].data(), image_buffers[1].size() );


      cv::cvtColor(input_image_1_bgr_3x8, input_image_1_bgr_3x8, cv::COLOR_BGR2RGB);
      cv::cvtColor(input_image_2_bgr_3x8, input_image_2_bgr_3x8, cv::COLOR_BGR2RGB);
    } else {
      memcpy(input_image_1_bgr_1x8.data, image_buffers[0].data(), image_buffers[0].size() );
      memcpy(input_image_2_bgr_1x8.data, image_buffers[1].data(), image_buffers[1].size() );
    
      cv::cvtColor(input_image_1_bgr_1x8, input_image_1_bgr_3x8, cv::COLOR_GRAY2BGR);
      cv::cvtColor(input_image_2_bgr_1x8, input_image_2_bgr_3x8, cv::COLOR_GRAY2BGR);
    }

  }


  if(m_camera_parameters_available) {
    //input_image_1_bgr_3x8_cl = input_image_1_bgr_3x8.getUMat(cv::ACCESS_READ);
    //input_image_2_bgr_3x8_cl = input_image_2_bgr_3x8.getUMat(cv::ACCESS_READ);

  	rectify_images();




    //input_image_1_bgr_3x8_rectified
  }

  auto start = std::chrono::system_clock::now();

  std::size_t num_byte_to_write = 0;

  if(m_are_images_downsampled) {
    num_byte_to_write = m_downsampled_image_width * m_downsampled_image_height * m_original_image_num_channels * sizeof(char);
  } else {
    num_byte_to_write = m_original_image_width * m_original_image_height * m_original_image_num_channels * sizeof(char);
  }

  if(m_camera_parameters_available) {
    //std::cout << "WRITING " << std::endl;
    clEnqueueWriteBuffer(command_queue, cl_input_buffer_1, CL_TRUE, 0, 
                         num_byte_to_write, input_image_1_bgr_3x8_rectified.data, 0, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, cl_input_buffer_2, CL_TRUE, 0, 
                         num_byte_to_write, input_image_2_bgr_3x8_rectified.data, 0, NULL, NULL);
  }


  cv::Vec2i stereo_matcher_image_dims = cv::Vec2i{0, 0};

  if(m_are_images_downsampled) {
    stereo_matcher_image_dims = cv::Vec2i{m_downsampled_image_width, m_downsampled_image_height};
  } else {
    stereo_matcher_image_dims = cv::Vec2i{m_original_image_width, m_original_image_height};
  }

  std::cout << "Allocating stereo matchers with resolution: " << stereo_matcher_image_dims[0] << ", "
                                                              << stereo_matcher_image_dims[1] << std::endl;

  if (m_use_adcensus) {
      if ( not stereo_ad_census_matcher_ptr) {
          stereo_ad_census_matcher_ptr = dsm::GPUAdCensusStereoMatcher::create(context, device_id,
                                                                               stereo_matcher_image_dims);
      }
      //for changing disparity, kernels need to be recompiled, which require context and device.
      stereo_ad_census_matcher_ptr->set_minimum_disparity(context, device_id, m_minimum_disparity);
      stereo_ad_census_matcher_ptr->set_maximum_disparity(context, device_id, m_maximum_disparity);
      stereo_ad_census_matcher_ptr->set_parameters(m_param_adcensus);
      stereo_ad_census_matcher_ptr->process(command_queue, cl_input_buffer_1, cl_input_buffer_2, cl_input_buffer_3, cl_input_buffer_4,
                                            cl_disparity_image_output_buffer);
  }
  else if(m_use_patch_match){
      if (not stereo_PATCH_MATCH_matcher__ptr) {
          stereo_PATCH_MATCH_matcher__ptr = dsm::GPUPatchMatchStereoMatcher::create(context, device_id,
                                                                               stereo_matcher_image_dims);
                                                                               //dsm::AdStereoMatchingMode::SIMPLE_ASW_LAB_3x32F);
      }

      stereo_PATCH_MATCH_matcher__ptr->set_num_iterations(m_num_iteration_propagation);
      stereo_PATCH_MATCH_matcher__ptr->set_plane_refine_steps(m_plane_refine_steps);
      stereo_PATCH_MATCH_matcher__ptr->set_slanted_or_fronto(m_slanted_or_fronto);
      stereo_PATCH_MATCH_matcher__ptr->set_temp_propagation(m_num_temporal_propagation);
      stereo_PATCH_MATCH_matcher__ptr->set_view_propagation(m_switch_view_prop);
      stereo_PATCH_MATCH_matcher__ptr->set_outlier_switch(m_switch_outlier_detection);
      stereo_PATCH_MATCH_matcher__ptr->set_minimum_disparity(context, device_id, m_minimum_disparity);
      stereo_PATCH_MATCH_matcher__ptr->set_maximum_disparity(context, device_id, m_maximum_disparity);
      stereo_PATCH_MATCH_matcher__ptr->set_search_window_half_size(m_search_window_half_size);
      stereo_PATCH_MATCH_matcher__ptr->process(command_queue, cl_input_buffer_1, cl_input_buffer_2, cl_input_buffer_3, cl_input_buffer_4,
                                               cl_disparity_image_output_buffer);

  }
  else{
      if (not stereo_matcher_NCC_ptr) {
          stereo_matcher_NCC_ptr = dsm::GPUStereoMatcherNCC::create(context, device_id,
                                                                    stereo_matcher_image_dims);
      }
      // set mode of matcher
      //stereo_matcher_ptr->set_mode(active_stereo_matching_mode);
      // pass new kernel arguments based on sliders
      stereo_matcher_NCC_ptr->set_search_window_half_size(m_search_window_half_size);
      stereo_matcher_NCC_ptr->set_minimum_disparity(m_minimum_disparity);
      stereo_matcher_NCC_ptr->set_maximum_disparity(m_maximum_disparity);

      stereo_matcher_NCC_ptr->process(command_queue, cl_input_buffer_1, cl_input_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_disparity_image_output_buffer);
  }

  reconstructor_ptr->set_min_valid_disparity(m_minimum_disparity);
  //CALL RECONSTRUCTOR PROCESS WITH BUFFERS HERE

  reconstructor_ptr->set_enable_distance_cut_off(m_use_distance_cut_off);
  reconstructor_ptr->set_focal_length(m_focal_length);

/*
  if(m_send_geometry_data) {
    reconstrutor_ptr->set_mode(dsm::ReconstructionMode::FLOAT_DISPARITY_TO_VERTEX_UV_TRIANGLES_POINTS);
  } else {
    reconstrutor_ptr->set_mode(dsm::ReconstructionMode::FLOAT_DISPARITY_TO_VERTEX_UV_TRIANGLES_POINTS);
  }
*/
  reconstructor_ptr->process(command_queue, cl_disparity_image_output_buffer, cl_input_buffer_1, cl_point_output_buffer, cl_color_output_buffer);




#if ENABLE_KERNEL_PROFILING
    dsm::GPUBaseImageProcessor::print_and_clear_kernel_execution_times();
#endif



  std::size_t num_byte_to_read_result = stereo_matcher_image_dims[0] * stereo_matcher_image_dims[1]* 1 * sizeof(float);

  /* READBACK: GPU -> CPU.
     cl_output_buffer contains is our cl_mem object. (GPU data handle)
     output_image_grayscale_1x8 is a cv::Mat which was pre-allocated with a size of of width*height*1 
     (because we need one byte for disparities up to 255)
  */

  //clEnqueueReadBuffer(command_queue, cl_disparity_image_output_buffer, CL_TRUE, 0, 
  //                    num_byte_to_read_result, output_image_grayscale_1x8.data, 0, NULL, NULL);

  clEnqueueReadBuffer(command_queue, cl_disparity_image_output_buffer, CL_TRUE, 0, 
                      num_byte_to_read_result, output_image_grayscale_1x32f.data, 0, NULL, NULL);


  uint32_t max_num_points = 2*3*(stereo_matcher_image_dims[0]-1)*(stereo_matcher_image_dims[1]-1);
  uint32_t num_byte_to_read_pos_result = max_num_points * 3 * sizeof(float);
  
  cl_int status_code = clEnqueueReadBuffer(command_queue, cl_point_output_buffer, CL_TRUE, 0, 
                                           num_byte_to_read_pos_result, point_positions_vec.data(), 0, NULL, NULL);

  uint32_t num_byte_to_read_col_result = max_num_points * 3 * sizeof(unsigned char);
  status_code = clEnqueueReadBuffer(command_queue, cl_color_output_buffer, CL_TRUE, 0, 
                                    num_byte_to_read_col_result, point_color_vec.data(), 0, NULL, NULL);



  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  	clFlush(command_queue);

  std::cout << "stereo matching took  " << " " << elapsed_seconds.count() * 1000 << "ms\n";
}

void Application::send_data() {

  
    size_t num_texels_to_send_x = m_are_images_downsampled ? m_downsampled_image_width : m_original_image_width;
    size_t num_texels_to_send_y = m_are_images_downsampled ? m_downsampled_image_height : m_original_image_height;

    size_t total_num_texture_byte_to_send = num_texels_to_send_y * num_texels_to_send_x * 3;

    SGTP::send_package_t sgtp_package;



    uint32_t actually_num_reconstructed_triangles = point_positions_vec.size() / (3*3);
    uint32_t num_triangles_to_send = std::min(168750u, actually_num_reconstructed_triangles);

    if(true) {
      std::cout << "Num Triangles to send: " << num_triangles_to_send << "\n";
    }

    //1. number of triangles we want to send ( = num vertices / 3)
    sgtp_package.header.num_textured_triangles = std::min(168750u,num_triangles_to_send);

    //2. bounding rectangle of our generated texture ( in our case almost always {0, 0}, {num_texels_to_send_x - 1, num_texels_to_send_y - 1}) !!!
    sgtp_package.header.tex_bounding_box[0] = { {0, 0}, {num_texels_to_send_x - 1, num_texels_to_send_y - 1}  };

    //3. number of bytes for geometry data = num vertices * size of one vertex (here: 5x float, as described in SGTP)
    sgtp_package.header.geometry_payload_size = (num_triangles_to_send*3) * SGTP::TEXTURED_VERTEX_UNCOMPRESSED_SIZE;
    //4. number of bytes for texture data = num vertices * size of one vertex (here: 5x float, as described in SGTP)
    sgtp_package.header.texture_payload_size = total_num_texture_byte_to_send;
    
    uint64_t total_payload = sgtp_package.header.texture_payload_size + sgtp_package.header.geometry_payload_size;

    uint32_t max_num_creatable_triangles = 2 * (num_texels_to_send_x - 1) *  (num_texels_to_send_y - 1);
    uint32_t max_num_triangle_byte_size = max_num_creatable_triangles * 3 * SGTP::TEXTURED_VERTEX_UNCOMPRESSED_SIZE;

    uint64_t max_possible_payload = total_num_texture_byte_to_send + max_num_triangle_byte_size; 


    if(m_combined_geometry_and_texture_data.empty() ) {
        m_combined_geometry_and_texture_data.resize(max_possible_payload);
    }

    bool skip_sending = false;
    if(num_triangles_to_send == 0 ) {
      skip_sending = true;
    }
    


    auto remaining_jobs_start = std::chrono::system_clock::now();
    


    std::vector<float> m_triangle_vertices(sgtp_package.header.num_textured_triangles * 3 * 5);

    for(int triangle_index = 0; triangle_index < num_triangles_to_send; ++triangle_index) {
      uint32_t triangle_base_offset_in = triangle_index * 3 * 3; //input buffer only stores xyz, so 3 vertices and 3 floats per tri
      uint32_t triangle_base_offset_out = triangle_index * 3 * 5; //input buffer combines xyz and uvs, so 3 vertices and 5 floats per tri
      for(int local_vertex_id = 0; local_vertex_id < 3; ++local_vertex_id) {
        m_triangle_vertices[triangle_base_offset_out + local_vertex_id * 5 + 0] = point_positions_vec[triangle_base_offset_in + local_vertex_id * 3 + 0];
        m_triangle_vertices[triangle_base_offset_out + local_vertex_id * 5 + 1] = point_positions_vec[triangle_base_offset_in + local_vertex_id * 3 + 1];
        m_triangle_vertices[triangle_base_offset_out + local_vertex_id * 5 + 2] = point_positions_vec[triangle_base_offset_in + local_vertex_id * 3 + 2];
      }

    }

    //if(!skip_sending) {
 

      // create vector containing geometry and texture data tightly packed in one contiguous chunk of memory (first geometry, then texture data)
      //std::vector<unsigned char> combined_geometry_and_texture_data(total_payload);

      uint32_t message_write_offset = 0;
      //copy geometry data from m_triangle_vertices to beginning of combined vector (note: write offset = 0)
      memcpy((char*)m_combined_geometry_and_texture_data.data() + message_write_offset, m_triangle_vertices.data(), sgtp_package.header.geometry_payload_size );
      //increase write offset by amount of byte in geometry data
      message_write_offset += sgtp_package.header.geometry_payload_size;
      //copy texture data from m_texture_data_bgr to second part of combined vector (note: write offset = sgtp_package.header.geometry_payload_siz)
      memcpy((char*)m_combined_geometry_and_texture_data.data() + message_write_offset, input_image_1_bgr_3x8.data, sgtp_package.header.texture_payload_size );

      // get the pointer to the data field in combined_geometry_and_texture_data and store it in sgtp message (will be copied by the geometry sender once again)
      sgtp_package.message = (uint8_t*) m_combined_geometry_and_texture_data.data();

      // the geometry sender class opens a zmq socket inside and prepares the zmq message to send 
      geometry_sender_ptr->send_data(sgtp_package);

      std::cout << "Sending data " << "\n";
    //}

  

}

// runs an endless loop until someone presses Escape (keycodes: http://www.foreui.com/articles/Key_Code_Table.htm)
void Application::run_mainloop() {


   do{
      // the process_frame function is called only if someone changed setting
      //if(m_changed_matching_state) {
        process_frame();

        m_changed_matching_state = false;
      //}
      if(m_show_gui) {

        if(m_show_model) {
      	 visualize_geometry();
        }
      }
      
      visualize_cv_window();


      //if(m_send_geometry_data) {
      //  send_data();
      //}

   } while(m_show_gui);
}

void Application::load_camera_parameters(const std::string &path , uint32_t img_width, uint32_t img_height) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if( fs.isOpened())
    {
        fs["M1"] >> _stereo_info.cam_mats[0];
        fs["M2"] >> _stereo_info.cam_mats[1];
        fs["D1"] >> _stereo_info.dist_coeffs[0];
        fs["D2"] >> _stereo_info.dist_coeffs[1];
        fs["R"] >> _stereo_info.R;
        fs["T"] >> _stereo_info.T;
        fs["R1"] >> _stereo_info.R1;
        fs["R2"] >> _stereo_info.R2;
        fs["P1"] >> _stereo_info.P1;
        fs["P2"] >> _stereo_info.P2;
        fs["Q"] >> _stereo_info.Q;
        fs["ValidRoi1"] >> _stereo_info.valid_roi[0];
        fs["ValidRoi2"] >> _stereo_info.valid_roi[1];
        fs.release();
    }
    else{
        std::cout << "Cannot open camera file. " << path << std::endl;
    }

    _stereo_info.base_line = std::abs(_stereo_info.T.at<double>(0,0));
    //TODO: while the pixel sizes along x,y are equal, the focal length fx, fy are different?
    //TODO: while base line is measure in meters, focal length are measured in pixel units?
    double fc_in_px_units = _stereo_info.cam_mats[0].at<double>(0,0);
    _stereo_info.focal_length = fc_in_px_units;

    auto &s = _stereo_info;
    cv::Size img_size (img_width, img_height);
    cv::initUndistortRectifyMap(s.cam_mats[0], s.dist_coeffs[0], s.R1, s.P1, img_size, CV_16SC2, s.rmap[0][0], s.rmap[0][1]);
    cv::initUndistortRectifyMap(s.cam_mats[1], s.dist_coeffs[1], s.R2, s.P2, img_size, CV_16SC2, s.rmap[1][0], s.rmap[1][1]);
}

void Application::rectify_images(){
    cv::remap(input_image_1_bgr_3x8, input_image_1_bgr_3x8_rectified, _stereo_info.rmap[0][0], _stereo_info.rmap[0][1], cv::INTER_LINEAR);
    cv::remap(input_image_2_bgr_3x8, input_image_2_bgr_3x8_rectified, _stereo_info.rmap[1][0], _stereo_info.rmap[1][1], cv::INTER_LINEAR);
  
    if(m_are_images_downsampled) {

      //std::cout << "Resizing images to: " << m_downsampled_image_width << ", " << m_downsampled_image_height << std::endl; 
      cv::resize(input_image_1_bgr_3x8_rectified, 
                 input_image_1_bgr_3x8_rectified, cv::Size(m_downsampled_image_width, m_downsampled_image_height),
                 0, 0, cv::INTER_LINEAR);
      cv::resize(input_image_2_bgr_3x8_rectified, 
                 input_image_2_bgr_3x8_rectified, cv::Size(m_downsampled_image_width  , m_downsampled_image_height),
                 0, 0, cv::INTER_LINEAR);
    }
}


