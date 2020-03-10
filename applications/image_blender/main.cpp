#include <image_processing/GPUImageProcessors.h> //

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
   the image blender app operates on 2 input images and generates 1 output image
*/
cl_mem cl_input_buffer_1 = 0;
cl_mem cl_input_buffer_2 = 0;
cl_mem cl_input_buffer_3 = 0;
cl_mem cl_input_buffer_4 = 0;

cl_mem cl_output_buffer = 0;

/* 
   global variables containing some handles to often used cl objects 
*/

cl_context context = 0; //cl context coupled with the platform
cl_device_id device_id = 0; //cl device coupled with our GPU (in our case)
cl_command_queue command_queue = 0; // command queue couped with a compute unit
                                    // in here, we enqueue our kernels and memory transfer operation

//note: in this app, the kernel is already built into the image processing framework


// we do not want to deal with image encoding and decoding, so we let opencv do this by loading images
// into cv::Mat objects 
// Note: Here they are still uninitialized
cv::Mat input_image_1;
cv::Mat input_image_2;

cv::Mat output_image;

// we want to be able to pass some command line parameters in an easily understandable way
// ->boost program options
namespace po = boost::program_options; 
po::options_description desc("Options");

std::string const& window_name = "Image Blender";
int window_size_x = 0;
int window_size_y = 0;

bool show_gui = false;

std::shared_ptr<dsm::GPUImageBlender> image_blender_ptr = nullptr;
dsm::BlendMode active_mode = dsm::BlendMode::ADD_3x8;


void define_command_line_arguments();
// gui layout definition
void define_GUI();

void run_mainloop();



// forward declaration of GUI events. definition is at the end of the file
void on_button_addition(int state, void* userdata);
void on_button_difference(int state, void* userdata);
void on_button_multiplication(int state, void* userdata);

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

	std::string image_1_path = "./images/Flowerpots/view1.png"; 
	std::string image_2_path = "./images/Flowerpots/view5.png"; 

    if ( !vm["image_1"].empty() && !vm["image_2"].empty()) { 
      image_1_path = vm["image_1"].as<std::string>();
      image_2_path = vm["image_2"].as<std::string>();
      //          << desc << std::endl; 
      //return 0; 
    } else {
        std::cout <<"Image paths were not specified. Using default images:\n";
        std::cout << image_1_path << "\n";
        std::cout << image_2_path << "\n";
    }


    show_gui = vm["no-gui"].empty();
    //load image
    int width, height;
   
    input_image_1 = cv::imread(image_1_path.c_str(), cv::IMREAD_COLOR);
    input_image_2 = cv::imread(image_2_path.c_str(), cv::IMREAD_COLOR);

    width = input_image_1.cols;
    height = input_image_1.rows;

    window_size_x = width;
    window_size_y = height;

    std::vector<int> window_size;
    if (!vm["win_size"].empty() && (window_size = vm["win_size"].as<std::vector<int> >()).size() == 2) {
      // valid window size
        window_size_x = window_size[0];
        window_size_y = window_size[1];       
    }


    //3x8 unsigned character per pixel
    output_image = cv::Mat(height, width, CV_8UC3);

    dsm::initialize_cl_environment(context, device_id, command_queue);

  
    initialize_cl_buffers(width, height, input_image_1.data, input_image_2.data);

    if(show_gui) {
        // creates the gui
        define_GUI();
    }


    image_blender_ptr = dsm::GPUImageBlender::create(context, device_id, 
                                                     cv::Vec2i{input_image_1.cols, input_image_1.rows}, 
                                                     active_mode);



    // blocking in GUI mode until escape is pressed
    run_mainloop();

    std::string const& executable_name = dsm::get_filename_from_path(argv[0]);

    std::string const& output_image_path = "./result_" + executable_name + ".png";

    cv::imwrite(output_image_path, output_image);

    std::cout << std::endl;
    std::cout << "Wrote output image to: " << output_image_path << std::endl;

    int status = 0;
    // Clean the app resources.
    status = clReleaseMemObject(cl_input_buffer_1); //Release mem object.
    status = clReleaseMemObject(cl_input_buffer_2); //Release mem object.
    status = clReleaseMemObject(cl_output_buffer);
    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL) {
        free(device_id);
        device_id = NULL;
    }

	return 0;
}

void initialize_cl_buffers(int image_width, int image_height, unsigned char* rgb_data_im_1, unsigned char* rgb_data_im_2) {
  int num_channels_input_output_image = 3; //RGB, 8 bit per channel for all images

  cl_input_buffer_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels_input_output_image * sizeof(char), (void *)rgb_data_im_1, NULL);
  cl_input_buffer_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         image_width * image_height * num_channels_input_output_image * sizeof(char), (void *)rgb_data_im_2, NULL);

  cl_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        image_width * image_height * num_channels_input_output_image * sizeof(char), NULL, NULL);
}

//define 
void define_command_line_arguments() {
    desc.add_options() 
      ("help,h", "Print help messages") //bool
      ("image_1,i", po::value<std::string>(), "Path to image 1") //std::string
      ("image_2,j", po::value<std::string>(), "Path to image 2") //std::string
      ("no-gui,n", "No GUI") //bool
      ("win_size,w", po::value<std::vector<int>>()->multitoken(), "2D size of window in gui x y"); //2 ints


}

void define_GUI() {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, window_size_x, window_size_y);

    cv::createButton("Add", on_button_addition, NULL, cv::QT_RADIOBOX);
    cv::createButton("Difference", on_button_difference, NULL, cv::QT_RADIOBOX);
    cv::createButton("Multiply", on_button_multiplication, NULL, cv::QT_RADIOBOX);
}


void run_mainloop() {

    dsm::BlendMode active_mode_last_frame = active_mode;

    do{
        if(active_mode != active_mode_last_frame) {
            image_blender_ptr->set_mode(active_mode);
            active_mode_last_frame = active_mode;
        }

        image_blender_ptr->process(command_queue, cl_input_buffer_1, cl_input_buffer_2, cl_input_buffer_3, cl_input_buffer_4, cl_output_buffer);

        std::size_t num_byte_to_read_result = input_image_1.cols * input_image_1.rows * 3 * sizeof(char);

        clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                            num_byte_to_read_result, output_image.data, 0, NULL, NULL);

        #if ENABLE_KERNEL_PROFILING
            dsm::GPUBaseImageProcessor::print_and_clear_kernel_execution_times();
        #endif

        if(show_gui) {

            cv::Mat concatenated_result;
            cv::hconcat(input_image_1, input_image_2, concatenated_result);
            cv::hconcat(concatenated_result, output_image, concatenated_result);
            
            cv::imshow(window_name, concatenated_result); 
            int keycode = cv::waitKey(1) & 0xFF;

            //escape
            if(27 == keycode) {
                break;
            }
            //r or R 
            if(114 == keycode || 82 == keycode) {
              image_blender_ptr->reload_kernels(context, device_id);
              std::cout << "Reloaded kernels\n";
            }
            
        } else {
            break;
        }


    } while(show_gui);
}


void on_button_addition(int state, void* userdata) {
    active_mode = dsm::BlendMode::ADD_3x8;
}

void on_button_difference(int state, void* userdata) {
    active_mode = dsm::BlendMode::DIFFERENCE_3x8;
}

void on_button_multiplication(int state, void* userdata) {
    active_mode = dsm::BlendMode::MULTIPLY_3x8;

}