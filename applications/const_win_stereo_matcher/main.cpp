#include <image_processing/GPUImageProcessors.h>
#include "GPUConstWinStereoMatcher.h"
#include <core/init_opencl.h>
#include <core/utils.h>

#include <CL/cl.h> //command_queues, devices, ...

// OpenCV 
#include <opencv2/core/core.hpp> //types
#include <opencv2/imgcodecs.hpp> //imread, imwrite, color formats
#include <opencv2/highgui/highgui.hpp> //gui elements, window

#include <boost/program_options.hpp> 

// standard header
#include <iostream>
#include <string>

/* 
   global variables which will contain handles to our GPU memory
   the image blender app operates on 2 input images in order to create 1 output disparity image.
   As
*/
cl_mem cl_input_buffer_1 = 0;
cl_mem cl_input_buffer_2 = 0;
cl_mem cl_input_buffer_3 = 0;
cl_mem cl_input_buffer_4 = 0;

cl_mem cl_grayscale_1x8_buffer_1 = 0;
cl_mem cl_grayscale_1x8_buffer_2 = 0;

cl_mem cl_lab_image_buffer_1 = 0;
cl_mem cl_lab_image_buffer_2 = 0;

cl_mem cl_output_buffer = 0;

cl_context context = 0;
cl_device_id device_id = 0;
cl_command_queue command_queue = 0;
cl_kernel kernel = 0;

// global objects which will help us load and save images
cv::Mat input_image_1_bgr_3x8;
cv::Mat input_image_2_bgr_3x8;

cv::Mat output_image_grayscale_1x8;

// reference disparity image we may load
cv::Mat reference_disparity_image_3x8;

bool changed_matching_state = true;

// some parameters limiting the amount of 
int const MAX_CONFIGURABLE_WINDOW_SIZE = 30;
int const MAX_CONFIGURABLE_DISPARITY = 75;
int const MAX_CONFIGURABLE_DISPARITY_VIS_SCALING = 10;

int search_window_half_size = 5;
int maximum_disparity = 60;
int disparity_vis_scaling = 4;

namespace po = boost::program_options; 
po::options_description desc("Options");

std::string const& window_name = "Constant Stereo Matching";
int window_size_x = 0;
int window_size_y = 0;

bool show_gui = false;

std::shared_ptr<dsm::GPUImageConverter> image_converter_ptr = nullptr;
dsm::ConversionMode active_conversion_mode = dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F;

std::shared_ptr<dsm::GPUConstWinStereoMatcher> stereo_matcher_ptr = nullptr;
dsm::ConstWinStereoMatchingMode active_stereo_matching_mode = dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_GREEN_3x8;



void define_command_line_arguments();
// gui layout definition
void define_GUI();

void process_frame();

void run_mainloop();



// forward declaration of GUI events. definition is at the end of the file
void on_button_SAD_green_similarity(int state, void* userdata);
void on_button_SAD_grayscale_similarity(int state, void* userdata);

void on_button_SAD_lab_32f_similarity(int state, void* userdata);

void on_button_ASW_lab_similarity(int state, void* userdata);

void on_search_window_size_trackbar(int state, void* userdata);
void on_maximum_disparity_trackbar(int state, void* userdata);
void on_disparity_vis_scaling_trackbar(int state, void* userdata);


//void on_button_multiplication(int state, void* userdata);

void initialize_cl_buffers(int image_width, int image_height, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2);


int main(int argc, char** argv) {

    std::string const usage_line = std::string("USAGE: ") + argv[0] + " <kernel_source_filename> [options] (-h or --help for more info)\n";


    define_command_line_arguments();

    po::variables_map vm; 

    po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw 

    if ( vm.count("help")  ) { 
      std::cout << usage_line << std::endl
                << desc << std::endl; 
      return 0; 
    } 

  // if no other image is specified, ./images/Flowerpots/view1.png
	std::string image_1_path = "./images/Flowerpots/view1.png"; 
	std::string image_2_path = "./images/Flowerpots/view5.png"; 

  std::string reference_disparity_image_path = "";

    if ( !vm["image_1"].empty() && !vm["image_2"].empty()) { 
      image_1_path = vm["image_1"].as<std::string>();
      image_2_path = vm["image_2"].as<std::string>();
      //          << desc << std::endl; 
      //return 0; 
    } else {
        std::cout << "Image paths were not specified. Using default images: \n"; 
        std::cout << image_1_path << "\n";
        std::cout << image_2_path << "\n";
    }

    if ( !vm["reference"].empty()) {
      reference_disparity_image_path = vm["reference"].as<std::string>();

      reference_disparity_image_3x8 = cv::imread(reference_disparity_image_path.c_str(), cv::IMREAD_COLOR);
    }

    show_gui = vm["no-gui"].empty();
    //load image
    int width, height, num_channels;
   
    input_image_1_bgr_3x8 = cv::imread(image_1_path.c_str(), cv::IMREAD_COLOR);
    input_image_2_bgr_3x8 = cv::imread(image_2_path.c_str(), cv::IMREAD_COLOR);

    width = input_image_1_bgr_3x8.cols;
    height = input_image_1_bgr_3x8.rows;
    num_channels = 3;

    window_size_x = width;
    window_size_y = height;

    std::vector<int> window_size;
    if (!vm["win_size"].empty() && (window_size = vm["win_size"].as<std::vector<int> >()).size() == 2) {
      // valid window size
        window_size_x = window_size[0];
        window_size_y = window_size[1];       
    }


    //3x8 unsigned character per pixel
    output_image_grayscale_1x8 = cv::Mat(height, width, CV_8UC1);

    dsm::initialize_cl_environment(context, device_id, command_queue);

  
    initialize_cl_buffers(width, height, input_image_1_bgr_3x8.data, input_image_2_bgr_3x8.data);

    if(show_gui) {
        // creates the gui
        define_GUI();
    }

    image_converter_ptr = dsm::GPUImageConverter::create(context, device_id,
                                                         cv::Vec2i{input_image_1_bgr_3x8.cols, input_image_1_bgr_3x8.rows},
                                                         dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
    stereo_matcher_ptr = dsm::GPUConstWinStereoMatcher::create(context, device_id,
                                                       cv::Vec2i{input_image_1_bgr_3x8.cols, input_image_1_bgr_3x8.rows}, 
                                                       active_stereo_matching_mode);



    // blocking in GUI mode until escape is pressed
    run_mainloop();

    cv::imwrite("./write_testX.png", output_image_grayscale_1x8);

    int status = 0;
    // Clean the app resources.
    status = clReleaseMemObject(cl_input_buffer_1); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_2); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_3); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_4); //Release mem object.
    status = clReleaseMemObject(cl_output_buffer);
    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL) {
        free(device_id);
        device_id = NULL;
    }

	return 0;
}

// loads 
void initialize_cl_buffers(int image_width, int image_height, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2) {

  /* 3x8 bgr, flag CL_MEM_READ_ONLY (since these are input images) and tell cl to fill the buffer with host data
     rgb_data_im_1 and rgb_data_im_2, respectively (CL_MEM_COPY_HOST_PTR). 

     The function needs to know how many byte it should allocate (here: width*height*3). The factor 3 is because we have
     3x 8bit for the red, green and blue channel of the input images
  */

  cl_input_buffer_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * 3 * sizeof(char), (void *)rgb_data_im_1, NULL);
  cl_input_buffer_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * 3 * sizeof(char), (void *)rgb_data_im_2, NULL);

  /* 1x8 grayscale, flag CL_MEM_READ_WRITE (since we need to create the grayscale image ourselves).
     We tell CL to not fill it with data from the CPU (see NULL argument compared to input buffers) 

     The function needs to know how many byte it should allocate (here: width*height*3). The factor 3 is because we have
     3x 8bit for the red, green and blue channel of the input images
  */
  // 1x8 grayscale
  cl_grayscale_1x8_buffer_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                              image_width * image_height * 1 * sizeof(char), NULL, NULL);
  cl_grayscale_1x8_buffer_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                              image_width * image_height * 1 * sizeof(char), NULL, NULL);

  /* similar to 1x8 grayscale, but uses different format encoded into 3 floats (which is why it is 32 bit = sizeof(float), flag CL_MEM_READ_WRITE (since we need to create the grayscale image ourselves).
     We tell CL to not fill it with data from the CPU (see NULL argument compared to input buffers) 
  */

  // 3x32f
  cl_lab_image_buffer_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                            image_width * image_height * 3 * sizeof(float), NULL, NULL);
  cl_lab_image_buffer_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                            image_width * image_height * 3 * sizeof(float), NULL, NULL);

  // 1x8 8 bit out image, flag WRITE_ONLY (since this is an output image) and tell cl to not fill the buffer with any data
  // stores unsigned integer CL_MEM_WRITE_ONLY up to 255 (= 2^8)

  cl_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        image_width * image_height * 1 * sizeof(char), NULL, NULL);
}

//define command line arguments
void define_command_line_arguments() {
    desc.add_options() 
      ("help,h", "Print help messages") //bool
      ("image_1,i", po::value<std::string>(), "Path to image 1") //std::string
      ("image_2,j", po::value<std::string>(), "Path to image 2") //std::string
      ("reference,r", po::value<std::string>(), "Path to reference disparity 1 to 2") //std::string
      ("no-gui,n", "No GUI") //bool
      ("win_size,w", po::value<std::vector<int>>()->multitoken(), "2D size of window in gui x y"); //2 ints
}

// create gui elemenets
void define_GUI() {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, window_size_x, window_size_y);

    // parameters for trackbars: element name, window name which it is associated with, address of parameter we need to change, limit, callback function name
    cv::createTrackbar( "Search Window Half Size", window_name, &search_window_half_size, MAX_CONFIGURABLE_WINDOW_SIZE, on_search_window_size_trackbar );
    cv::createTrackbar( "Maximum Disparity", window_name, &maximum_disparity, MAX_CONFIGURABLE_DISPARITY, on_maximum_disparity_trackbar );
    
    cv::createTrackbar( "Disparity Vis Scaling", window_name, &disparity_vis_scaling, MAX_CONFIGURABLE_DISPARITY_VIS_SCALING, on_disparity_vis_scaling_trackbar );
        
  // parameters for buttons: element name, callback name, NULL, QT-Element Type
    //cv::createButton("SAD_green", on_button_SAD_green_similarity, NULL, cv::QT_RADIOBOX);
    //cv::createButton("SAD_gray", on_button_SAD_grayscale_similarity, NULL, cv::QT_RADIOBOX);

    //cv::createButton("SAD_lab_32f", on_button_SAD_lab_32f_similarity, NULL, cv::QT_RADIOBOX);
    //cv::createButton("ASW_lab", on_button_ASW_lab_similarity, NULL, cv::QT_RADIOBOX);
}


void process_frame() {

  // set mode of matcher
  stereo_matcher_ptr->set_mode(active_stereo_matching_mode);

  // pass new kernel arguments based on sliders
  stereo_matcher_ptr->set_search_window_half_size(search_window_half_size);
  stereo_matcher_ptr->set_maximum_disparity(maximum_disparity);
  
  // call different process routines for different selected modes
  switch(active_stereo_matching_mode) {
      case dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_GREEN_3x8: {
          //no prior conversion needed
          stereo_matcher_ptr->process(command_queue, cl_input_buffer_1, cl_input_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_output_buffer);
          break;
      }
      case dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_GRAYSCALE_1x8: {
          //convert bgr 3x8 to lab 3x32f
          image_converter_ptr->set_mode(dsm::ConversionMode::BGR_3x8_TO_GRAYSCALE_1x8);
          //convert image 1 bgr -> lab
          image_converter_ptr->process(command_queue, cl_input_buffer_1, cl_grayscale_1x8_buffer_1);
          //convert image 2 bgr -> lab
          image_converter_ptr->process(command_queue, cl_input_buffer_2, cl_grayscale_1x8_buffer_2);
          stereo_matcher_ptr->process(command_queue, cl_grayscale_1x8_buffer_1, cl_grayscale_1x8_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_output_buffer);
          break;
      }
      case dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_LAB_3x16F: {
          //convert bgr 3x8 to lab 3x16f
          image_converter_ptr->set_mode(dsm::ConversionMode::BGR_3x8_TO_LAB_3x16F);
          //convert image 1 bgr -> lab16f
          image_converter_ptr->process(command_queue, cl_input_buffer_1, cl_lab_image_buffer_1);
          //convert image 2 bgr -> lab16f
          image_converter_ptr->process(command_queue, cl_input_buffer_2, cl_lab_image_buffer_2);
          //stereo_matcher_ptr->process(command_queue, cl_lab_image_buffer_1, cl_lab_image_buffer_2, cl_output_buffer);
          break;
      }
      case dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_LAB_3x32F: {
          //convert bgr 3x8 to lab 3x32f
          image_converter_ptr->set_mode(dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
          //convert image 1 bgr -> lab32f
          image_converter_ptr->process(command_queue, cl_input_buffer_1, cl_lab_image_buffer_1);
          //convert image 2 bgr -> lab32f
          image_converter_ptr->process(command_queue, cl_input_buffer_2, cl_lab_image_buffer_2);
          stereo_matcher_ptr->process(command_queue, cl_lab_image_buffer_1, cl_lab_image_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_output_buffer);
          break;
      }
      case dsm::ConstWinStereoMatchingMode::SIMPLE_ASW_LAB_3x32F: {
          //convert bgr 3x8 to lab 3x32f
          image_converter_ptr->set_mode(dsm::ConversionMode::BGR_3x8_TO_LAB_3x32F);
          //convert image 1 bgr -> lab
          image_converter_ptr->process(command_queue, cl_input_buffer_1, cl_lab_image_buffer_1);
          //convert image 2 bgr -> lab
          image_converter_ptr->process(command_queue, cl_input_buffer_2, cl_lab_image_buffer_2);
          stereo_matcher_ptr->process(command_queue, cl_lab_image_buffer_1, cl_lab_image_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_output_buffer);
          break;
      }
      default:
        DSM_LOG_ERROR("Stereo Matching Mode is undefined");
        exit(1);
          break;
  }

#if ENABLE_KERNEL_PROFILING
    dsm::GPUBaseImageProcessor::print_and_clear_kernel_execution_times();
#endif



  std::size_t num_byte_to_read_result = input_image_1_bgr_3x8.cols * input_image_1_bgr_3x8.rows * 1 * sizeof(char);

  /* READBACK: GPU -> CPU.
     cl_output_buffer contains is our cl_mem object. (GPU data handle)
     output_image_grayscale_1x8 is a cv::Mat which was pre-allocated with a size of of width*height*1 
     (because we need one byte for disparities up to 255)
  */

  clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                      num_byte_to_read_result, output_image_grayscale_1x8.data, 0, NULL, NULL);

}


// runs an endless loop until someone presses Escape (keycodes: http://www.foreui.com/articles/Key_Code_Table.htm)
void run_mainloop() {


    do{
        // the process_frame function is called only if someone changed setting
        if(changed_matching_state) {
          process_frame();
          changed_matching_state = false;
        }

        // everything in the show_gui braces is purely meant for the visualization of the results
        // this is completely on the cpu side and helps you to verify whether your code is doing
        // what it is supposed to do
        if(show_gui) {            
            cv::Mat grayscale_3x8_mat(output_image_grayscale_1x8.cols, output_image_grayscale_1x8.rows, CV_8UC3);
            
            cv::cvtColor(output_image_grayscale_1x8, grayscale_3x8_mat, cv::COLOR_GRAY2BGR);

            cv::Mat scaled_out_mat_3x8(output_image_grayscale_1x8.cols, output_image_grayscale_1x8.rows, CV_8UC3);

            scaled_out_mat_3x8 = grayscale_3x8_mat * disparity_vis_scaling;

            cv::Mat concatenated_result;
            cv::hconcat(input_image_1_bgr_3x8, input_image_2_bgr_3x8, concatenated_result);
            cv::hconcat(concatenated_result, scaled_out_mat_3x8, concatenated_result);
            
            if(reference_disparity_image_3x8.dims == 2) {
              cv::hconcat(concatenated_result, reference_disparity_image_3x8, concatenated_result);

              cv::Mat absolute_difference_image;
              cv::absdiff(scaled_out_mat_3x8, reference_disparity_image_3x8, absolute_difference_image);

              

              cv::hconcat(concatenated_result, absolute_difference_image, concatenated_result);
  
            }

            // shows the image which we created from images of the same dimensionality
            // imshow just shows one image, which we created in the local matrix named "concatenated result"
            cv::imshow(window_name, concatenated_result); 
            // waits one millisecond and returns keycodes if a key was pressed
            int keycode = cv::waitKey(1) & 0xFF;

            //escape
            if(27 == keycode) {
                break;
            }
            //r or R 
            if(114 == keycode || 82 == keycode) {
              image_converter_ptr->reload_kernels(context, device_id);
              stereo_matcher_ptr->reload_kernels(context, device_id);
              std::cout << "Reloaded kernels\n";
              changed_matching_state = true;
            }
        } else {
            break;
        }

    } while(show_gui);
}



//callback functions for radio buttons
void on_button_SAD_green_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_GREEN_3x8;
    changed_matching_state = true;
}

void on_button_SAD_grayscale_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_GRAYSCALE_1x8;
    changed_matching_state = true;
}

void on_button_SAD_lab_32f_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::ConstWinStereoMatchingMode::SIMPLE_SAD_LAB_3x32F;
    changed_matching_state = true;
}

void on_button_ASW_lab_similarity(int state, void* userdata) {
    active_stereo_matching_mode = dsm::ConstWinStereoMatchingMode::SIMPLE_ASW_LAB_3x32F;
    changed_matching_state = true;
}


// callback functions for sliders
void on_search_window_size_trackbar(int state, void* userdata) {
    changed_matching_state = true;
}

void on_maximum_disparity_trackbar(int state, void* userdata) {
    changed_matching_state = true;
}

void on_disparity_vis_scaling_trackbar(int state, void* userdata) {
  disparity_vis_scaling = std::max(1, disparity_vis_scaling);
  std::cout << "Changed disparity vis scaling to " << disparity_vis_scaling << std::endl;
}
