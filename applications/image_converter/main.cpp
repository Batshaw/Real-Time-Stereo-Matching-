#include <image_processing/GPUImageProcessors.h>

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

cl_mem cl_input_buffer = 0;
cl_mem cl_output_buffer = 0;

cl_context context = 0;
cl_device_id device_id = 0;
cl_command_queue command_queue = 0;


cv::Mat input_image;
cv::Mat output_image;

bool show_gui = false;
std::string const& window_name = "Converted_Image";

cl_int status = 0;


dsm::ConversionMode active_mode = dsm::ConversionMode::BGR_3x8_TO_GRAYSCALE_3x8;
std::shared_ptr<dsm::GPUImageConverter> gpu_image_converter_ptr = nullptr;

// gui layout definition
void define_GUI();

void run_mainloop();


// forward declaration of GUI events. definition is at the end of the file
void on_button_bgr_to_grayscale(int state, void* userdata);
void on_button_bgr_to_binary(int state, void* userdata);
void on_button_bgr_to_rgb(int state, void* userdata);


int main(int argc, char** argv) {
    std::string const usage_line = std::string("USAGE: ") + argv[0] + " <kernel_source_filename> [options] (-h or --help for more info)\n";


    namespace po = boost::program_options; 
    po::options_description desc("Options");

    desc.add_options() 
      ("help,h", "Print help messages") //bool
      ("image,i", po::value<std::string>(), "Path to image 1") //std::string
      ("no-gui,n", "No GUI") //bool
    ;


    po::variables_map vm; 

    po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw 

    if ( vm.count("help")  ) { 
      std::cout << usage_line << std::endl
                << desc << std::endl; 
      return 0; 
    } 

	std::string image_path = "./images/Flowerpots/view1.png"; 


    if ( !vm["image"].empty()) { 
      image_path = vm["image"].as<std::string>();
      //          << desc << std::endl; 
      //return 0; 
    } else {
        std::cout << "Image paths were not specified. Using default images: \n"; 
        std::cout << image_path << "\n";
    }

    show_gui = vm["no-gui"].empty();
    //load image
    int width, height, num_channels;
   
    input_image = cv::imread(image_path.c_str(), cv::IMREAD_COLOR);

    width = input_image.cols;
    height = input_image.rows;
    num_channels = 3;



    //3x8 unsigned character per pixel
    output_image = cv::Mat(height, width, CV_8UC3);


    dsm::initialize_cl_environment(context, device_id, command_queue);


    cl_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                         width * height * num_channels * sizeof(char), (void *)input_image.data, &status);

    if(CL_SUCCESS != status) {
        std::cout << "clCreateBuffer Status: " << status << "\n";
    }


    cl_output_buffer =  clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        width * height * num_channels * sizeof(char), NULL, &status);


    gpu_image_converter_ptr =
        dsm::GPUImageConverter::create(context, device_id, cv::Vec2i{input_image.cols, input_image.rows}, dsm::ConversionMode::BGR_3x8_TO_GRAYSCALE_3x8);


    if(show_gui) {
        define_GUI();
    }


    // blocking in GUI mode until escape is pressed
    run_mainloop();

    std::string const& executable_name = dsm::get_filename_from_path(argv[0]);

    std::string const& output_image_path = "./result_" + executable_name + ".png";

    cv::imwrite(output_image_path, output_image);

    std::cout << std::endl;
    std::cout << "Wrote output image to: " << output_image_path << std::endl;


    status = clReleaseMemObject(cl_input_buffer); //Release mem object.
    status = clReleaseMemObject(cl_output_buffer);
    status = clReleaseCommandQueue(command_queue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (device_id != NULL)
    {
        free(device_id);
        device_id = NULL;
    }

	return 0;
}

void define_GUI() {
    cv::namedWindow(window_name);

    cv::createButton("RGB -> Grayscale", on_button_bgr_to_grayscale, NULL, cv::QT_RADIOBOX);
    cv::createButton("RGB -> Binary", on_button_bgr_to_binary, NULL, cv::QT_RADIOBOX);
    cv::createButton("RGB -> BGR", on_button_bgr_to_rgb, NULL, cv::QT_RADIOBOX);
}

void run_mainloop() {
    dsm::ConversionMode active_mode_last_frame = active_mode;
    do {

        if(active_mode_last_frame != active_mode) {
            gpu_image_converter_ptr->set_mode(active_mode);
            active_mode_last_frame = active_mode;
        }

        gpu_image_converter_ptr->process(command_queue, cl_input_buffer, cl_output_buffer);
        
        std::size_t num_byte_to_read_result = input_image.cols * input_image.rows * 3 * sizeof(char);

        status = clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                                     num_byte_to_read_result, output_image.data, 0, NULL, NULL);
        if(CL_SUCCESS != status) {
            std::cout << "EnqueueReadBufferFailed Status: " << status << "\n";
        }

        #if ENABLE_KERNEL_PROFILING
            dsm::GPUBaseImageProcessor::print_and_clear_kernel_execution_times();
        #endif

        if(show_gui) {
            cv::Mat concatenated_result;
            cv::hconcat(input_image, output_image, concatenated_result);

            cv::imshow(window_name, concatenated_result);
            int keycode = cv::waitKey(1) & 0xFF; 

            if(27 == keycode) {
                break;
            }
            //r or R 
            if(114 == keycode || 82 == keycode) {
              gpu_image_converter_ptr->reload_kernels(context, device_id);
              std::cout << "Reloaded kernels\n";
            }

        } else {
            // gui was not specified, process pass only once
            break;
        }
    } while(true);
}


void on_button_bgr_to_grayscale(int state, void* userdata)
{
    active_mode = dsm::ConversionMode::BGR_3x8_TO_GRAYSCALE_3x8;
}

void on_button_bgr_to_binary(int state, void* userdata)
{
    active_mode = dsm::ConversionMode::BGR_3x8_TO_BINARY_3x8;
}

void on_button_bgr_to_rgb(int state, void* userdata) {
    active_mode = dsm::ConversionMode::BGR_3x8_TO_RGB_3x8;
}