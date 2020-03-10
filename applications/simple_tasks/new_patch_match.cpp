// small utility functions from the framework
#include <core/init_opencl.h>
#include <core/utils.h>

// OpenC types
#include <CL/cl.h> //command_queues, devices, ...

// OpenCV  headers
#include <opencv2/core/core.hpp> //types
#include <opencv2/imgcodecs.hpp> //imread, imwrite, color formats
#include <opencv2/highgui/highgui.hpp> //gui elements, window

// boost for simple program options
#include <boost/program_options.hpp> 

// standard header
#include <iostream>
#include <string>
#include <vector>

#include "patch_match.h"


#define COMPUTE_LEFT_DISPARITY 0
#define COMPUTE_RIGHT_DISPARITY 1

#define WITH_TEMP_PROP 0
#define WITHOUT_TEMP_PROP 1

using namespace dsm;


int main(int argc, char** argv) {

    uint32_t num_iterations = 0;

    if(argc < 7){
        std::cout << "USAGE: " << argv[0] << " <num_iterations> <min_disp> <max_disp> <win_radius> <left_image> <right_image> [ <inital_guess_left> <inital_guess_right> ]\n";
        return 1;
    }

    num_iterations = std::atoi(argv[1]);
    std::cout << "NUM ITERATIONS TO PERFORM " << num_iterations << std::endl;

    float min_disparity = std::atof(argv[2]);
    float max_disparity = std::atof(argv[3]);
    int window_radius = std::atoi(argv[4]);
    //std::cout << "MIN DISPARITY " << min_disparity << std::endl;
    //std::cout << "MAX DISPARITY " << max_disparity << std::endl;

    std::string const left_img_filename = argv[5];
    std::string const right_img_filename = argv[6];

    std::cout << "LEFT IMAGE FILENAME: " << left_img_filename << "\n";
    std::cout << "RIGHT IMAGE FILENAME: " << right_img_filename << "\n";

    cv::Mat left_image_grayscale = cv::imread(left_img_filename.c_str(), cv::IMREAD_GRAYSCALE);
    cv::Mat right_image_grayscale = cv::imread(right_img_filename.c_str(), cv::IMREAD_GRAYSCALE);

    cv::Mat left_image_color = cv::imread(left_img_filename.c_str(), cv::IMREAD_COLOR);
    cv::Mat right_image_color = cv::imread(right_img_filename.c_str(), cv::IMREAD_COLOR);

    int height = left_image_grayscale.cols;
    int width = left_image_grayscale.rows;

	
    //default black guess
    cv::Mat initial_guess_left; 
    cv::Mat initial_guess_right;
    //optional initial guess input
    if(argc == 9){
        std::cout << "INITIAL GUESS INPUT \n";
        std::string const left_in_guess_filename = argv[7];
        std::string const right_in_guess_filename = argv[8];
        initial_guess_left = cv::imread(left_in_guess_filename.c_str(), cv::IMREAD_GRAYSCALE);
        initial_guess_right = cv::imread(right_in_guess_filename.c_str(), cv::IMREAD_GRAYSCALE);
    }

    cl_context       context = 0;
    cl_device_id     device_id = 0;
    cl_command_queue command_queue = 0;
    cl_program       program = 0;

    dsm::initialize_cl_environment(context, device_id, command_queue); 
  
    //Patch_match pm(context, device_id, command_queue, cv::Vec2i {width, height});

    //pm.set_height(height);
    //pm.set_width(width);



    int num_channels_input_output_image_RGB = 3;
    int num_channels_input_output_image_R = 3;
    std::size_t num_byte_to_read_RGB = width * height * num_channels_input_output_image_RGB * sizeof(unsigned char);
    std::size_t num_byte_to_read_R = width * height * num_channels_input_output_image_R * sizeof(unsigned char);
    cl_mem  in_buffer_1 = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_to_read_RGB, 
                                        left_image_color.data, NULL);
    cl_mem  in_buffer_2 = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_to_read_RGB,
                                        right_image_color.data, NULL);
    cl_mem  in_buffer_3 = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_to_read_RGB, 
                                        initial_guess_left.data, NULL);
    cl_mem  in_buffer_4 = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_byte_to_read_RGB,
                                        initial_guess_right.data, NULL);
    cl_mem  out_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, num_byte_to_read_R,
                                        initial_guess_right.data, NULL);


    std::shared_ptr<dsm::Patch_match> stereo_PATCH_MATCH_matcher__ptr = nullptr;
    cv::Vec2i stereo_matcher_image_dims {width, height};
    
    stereo_PATCH_MATCH_matcher__ptr = dsm::Patch_match::create(context, device_id, stereo_matcher_image_dims);
    stereo_PATCH_MATCH_matcher__ptr->set_width(width);
    stereo_PATCH_MATCH_matcher__ptr->set_height(height);                                                                          
    stereo_PATCH_MATCH_matcher__ptr->set_num_iterations(num_iterations);
    //stereo_PATCH_MATCH_matcher__ptr->set_temp_propagation(m_num_temporal_propagation);
    stereo_PATCH_MATCH_matcher__ptr->set_minimum_disparity(context, device_id, min_disparity, width, height);
    stereo_PATCH_MATCH_matcher__ptr->set_maximum_disparity(context, device_id, max_disparity, width, height);
    stereo_PATCH_MATCH_matcher__ptr->set_search_window_half_size(window_radius);
    stereo_PATCH_MATCH_matcher__ptr->process(command_queue, in_buffer_1, in_buffer_2, in_buffer_3, in_buffer_4,
                                               out_buffer);
    //TO DO
    /*
    kernel init funktionen
    input init guess funktioniert ?
    optional view prop
    prepare temp prop for demo

    */


    /**********************************************************************************************/
    //result disparity image

    cv::Mat output_image;
    output_image = cv::Mat(width, height, CV_32FC3); //CV_32FC1
    
    std::size_t num_byte_to_read_R_32f = width * height * num_channels_input_output_image_R * sizeof(float);
    clEnqueueReadBuffer(command_queue, out_buffer, CL_TRUE, 0, 
                                     num_byte_to_read_R_32f, output_image.data, 0, NULL, NULL);

    std::string const output_filename = "test_patch_match.png";
    std::cout << "OUTPUT FILENAME: " << output_filename << "\n";
    cv::imwrite(output_filename.c_str(), output_image);

    //float rgb_data_result_left[width*height*4];
    //float gray_data_result_left[width*height];

    //show result disparity image
    std::string const& window_name{"Monitor Window"};	
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::resizeWindow(window_name, width, height);
    do {
		//cv::Mat image_mat = cv::Mat(width, height, CV_8UC1/*CV_8UC3*/, gray_data_result_left);
        //cv::Mat image_mat = cv::Mat(height, width, CV_8UC1, gray_data_result_left);

		cv::imshow(window_name, output_image);
		int key = cv::waitKey(1) & 0xFF;
		if(27 == key) {
			break;	
		}	
	} while(true);

    return 0;


}  


std::shared_ptr<Patch_match> 
Patch_match::create(cl_context const& context, cl_device_id const& device_id,
						           cv::Vec2i const& image_dimensions) {
	std::shared_ptr<Patch_match> image_blender_ptr = std::make_shared<Patch_match>(context, device_id, image_dimensions);
	return image_blender_ptr;
}

Patch_match::Patch_match(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims)
  : GPUImageProcessorBinary(image_dims) {
    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
   // _init_helper_processors(context, device_id, image_dims);
}

void Patch_match::process(cl_command_queue const& command_queue, 
						 	  cl_mem const& in_image_buffer_1, cl_mem const& in_image_buffer_2, cl_mem const& in_image_buffer_3, cl_mem const& in_image_buffer_4, 
						      cl_mem& out_image_buffer /* , int width, int height */) {
    
    
    _set_arg_and_run_copy_3_channel_buffer_to_image_gray(command_queue, 
                                            in_image_buffer_1, cl_left_input_image_, width_, height_);

    _set_arg_and_run_copy_3_channel_buffer_to_image_gray(command_queue, 
                                            in_image_buffer_2, cl_right_input_image_, width_, height_);

    if(in_image_buffer_3 == 0 && in_image_buffer_4 == 0){
        std::cout << " EMPTY - NO INPUT GUESS \n";
        //clear plane images A and B
        _set_arg_and_run_random_initialization(command_queue, cl_plane_image_left_a_,   
                                           maximum_disparity_, minimum_disparity_, width_, height_);
        _set_arg_and_run_random_initialization(command_queue, cl_plane_image_left_b_,   
                                          maximum_disparity_, minimum_disparity_, width_, height_);
    
        _set_arg_and_run_random_initialization(command_queue, cl_plane_image_right_a_,   
                                           maximum_disparity_, minimum_disparity_, width_, height_);
        _set_arg_and_run_random_initialization(command_queue, cl_plane_image_right_b_,   
                                           maximum_disparity_, minimum_disparity_, width_, height_); 
    }else{
        /* convert buffer to image RGBA float , convert RGBA_image to plane image*/
        set_arg_and_run_disp_buffer_to_plane_image(command_queue, 
                                            in_image_buffer_3, cl_disp_initial_guess_image_3_, cl_plane_image_left_a_, cl_plane_image_left_b_/*cl_plane_initial_guess_image_ */, 
                                            maximum_disparity_, minimum_disparity_, width_, height_); 

        set_arg_and_run_disp_buffer_to_plane_image(command_queue, 
                                            in_image_buffer_4, cl_disp_initial_guess_image_4_, cl_plane_image_right_a_, cl_plane_image_right_b_/*cl_plane_initial_guess_image_ */, 
                                            maximum_disparity_, minimum_disparity_, width_, height_);  

    }
    
    _set_arg_and_run_bgr_3x8_buffer_image_copy(command_queue, 
                                            in_image_buffer_1, cl_left_input_rgb_, width_, height_);

    _set_arg_and_run_bgr_3x8_buffer_image_copy(command_queue, 
                                            in_image_buffer_2, cl_right_input_rgb_, width_, height_);                                        

    _set_arg_and_run_gradient_filter(command_queue, cl_left_input_image_, cl_right_input_image_,
                                                     cl_gradient_image_left_, cl_gradient_image_right_, width_, height_);

    _set_arg_and_run_combine_rgb_and_gradient(command_queue, cl_left_input_rgb_, cl_right_input_rgb_,
                                             cl_gradient_image_left_, cl_gradient_image_right_,
                                             cl_left_input_rgb_grad_, cl_right_input_rgb_grad_, width_, height_);

    
    for(int32_t curr_iteration = 0; curr_iteration < num_iterations_; ++curr_iteration) {
        _set_arg_and_run_propagation_iteration(command_queue,  
                                               cl_plane_image_left_a_, cl_plane_image_left_b_,
                                               cl_plane_image_right_a_, cl_plane_image_right_b_,
                                               cl_left_input_rgb_grad_, cl_right_input_rgb_grad_,
                                               cl_plane_image_last_left, cl_plane_image_last_right,
                                               cl_gradient_image_left_, cl_gradient_image_right_,
                                               curr_iteration, maximum_disparity_, search_window_half_size_, COMPUTE_LEFT_DISPARITY, temp_prop_, width_, height_);
    }
    

    if(temp_prop_ == WITH_TEMP_PROP){
        _set_arg_and_run_copy_plane_image_to_last_plane(command_queue,
                                                        cl_plane_image_left_b_, cl_plane_image_right_b_,
                                                        cl_plane_image_last_left, cl_plane_image_last_right, width_, height_);
    }

    //if(in_image_buffer_3 == 0 && in_image_buffer_4 == 0){
        _set_arg_and_run_outlier_detection(command_queue, 
									   cl_plane_image_left_b_ ,cl_plane_image_right_b_, 
									   cl_disparity_image_outlier_detection_, 
                                       cl_disparity_outlier_mask_1_,
									   minimum_disparity_, maximum_disparity_, width_, height_);

        _set_arg_and_run_convet_RGBA_to_R(command_queue,  
                                        cl_disparity_image_outlier_detection_,
                                        cl_disparity_image_left_,
                                        num_iterations_,
                                        maximum_disparity_, width_, height_);
    // }else{
    //     _set_arg_and_run_convet_RGBA_to_R(command_queue,  
    //                                     cl_plane_image_left_b_,
    //                                     cl_disparity_image_left_,
    //                                     num_iterations_,
    //                                     maximum_disparity_);

    // }
                                                               
    _set_arg_and_run_median_filter_float_disparity_3x3(command_queue,
                                                        cl_disparity_image_left_,
                                                        cl_left_input_rgb_grad_,
                                                        cl_disparity_outlier_mask_1_,
                                                        cl_disparity_image_left_filtered_, width_, height_);

    _set_arg_and_run_copy_image_to_1_channel_buffer_gray(command_queue, 
                                                     cl_disparity_image_left_filtered_,
                                                     out_image_buffer, num_iterations_, width_, height_);

                                                

}
void Patch_match::set_minimum_disparity(cl_context const& context, cl_device_id const& device_id, float in_minimum_disparity, int width, int height) {
    if (in_minimum_disparity != minimum_disparity_){
        minimum_disparity_ = in_minimum_disparity;
        reload_resources(context, device_id);
    }
}
void Patch_match::set_maximum_disparity(cl_context const& context, cl_device_id const& device_id, float in_maximum_disparity, int width, int height) {
    if(in_maximum_disparity != maximum_disparity_)
    {
        maximum_disparity_ = in_maximum_disparity;
        reload_resources(context, device_id);
    }
}

void Patch_match::set_search_window_half_size(int in_search_window_half_size) {
    search_window_half_size_ = in_search_window_half_size;
}

void Patch_match::set_num_iterations(int in_num_iterations) {
    num_iterations_ = in_num_iterations;
}

void Patch_match::set_temp_propagation(int in_temp_propagation) {
    temp_prop_ = in_temp_propagation;
}

void Patch_match::set_outlier_switch(int in_outlier_switch){
    outlier_switch = in_outlier_switch;
}

void Patch_match::reload_resources(cl_context const& context, cl_device_id const& device_id)
{
    _cleanup_kernels();
    _cleanup_memory_objects();

    _init_kernels(context, device_id);
    _init_memory_objects(context, device_id);
}

void Patch_match::_register_kernel(cl_context const& context, cl_device_id const& device_id,
                                                  std::string const& kernel_path, std::string const& kernel_function_name,
                                                  cl_kernel& in_out_compiled_kernel, std::string kernel_defines) {
    cl_program program_to_compile = 0;

    dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                  program_to_compile, in_out_compiled_kernel, kernel_defines);

    cl_programs_.push_back(program_to_compile);
}



void Patch_match::_init_random_initialization_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "random_initialization";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_random_initialization_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_temporal_propagation_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "temporal_propagation";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_temporal_propagation_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_spatial_propagation_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "spatial_propagation";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_spatial_propagation_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_view_propagation_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "view_propagation";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_view_propagation_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_convert_RGBA_to_R_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "convert_RGBA_to_R";

   _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_convert_RGBA_to_R_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_plane_refinement_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "plane_refinement";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_plane_refinement_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_gradient_filter_kernel(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "gradient_filter";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_gradient_filter, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_rgb_and_gradient_combiner_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "rgb_and_gradient_combiner";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_rgb_and_gradient_combiner, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_copy_plane_to_last_plane(cl_context const& context, cl_device_id const& device_id) {
    
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "copy_plane_image_to_last_plane";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_init_copy_plane_image_to_last_plane, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_optimize_outlier_detection_kernel(cl_context const& context, cl_device_id const& device_id) {

    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "outlier_detection";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_optimize_outlier_detection_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_optimize_fill_invalid_pixel_kernel(cl_context const& context, cl_device_id const& device_id) {

    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "fill_invalid_pixel";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_optimize_fill_invalid_pixel_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_disp_buffer_to_plane_image(cl_context const& context, cl_device_id const& device_id) {

    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name_1 = "convert_image_to_disp_image_unnormalized";

    _register_kernel(context, device_id, kernel_path, kernel_function_name_1, kernel_copy_1x8_buffer_to_unnormalized_disp_img, _kernel_defines_simple_patch_match());

    std::string const kernel_function_name_2 = "convert_disp_image_to_plane_image";

    _register_kernel(context, device_id, kernel_path, kernel_function_name_2, kernel_convert_disp_to_plane_img, _kernel_defines_simple_patch_match());

}


void Patch_match::_init_outlier_detection_after_converting_RGBA_in_R(cl_context const& context, cl_device_id const& device_id) {

    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "outlier_detection_after_converting_RGBA_in_R";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_outlier_detection_after_converting_RGBA_in_R, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_copy_grayscale_3x8_buffer_to_image_2D_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_gray_buffer_image_copy.cl";
    std::string const kernel_function_name = "copy_grayscale_3x8_buffer_to_image_2D";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_copy_grayscale_3x8_buffer_to_image_2D_);
}

void Patch_match::_init_copy_bgr_3x8_buffer_to_image_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_gray_buffer_image_copy.cl";
    std::string const kernel_function_name = "copy_3x8_buffer_to_image_2D";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_copy_bgr_3x8_buffer_to_image_);
}

void Patch_match::_init_copy_image_2D_to_buffer_1x8_buffer_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_gray_buffer_image_copy.cl";
    std::string const kernel_function_name = "copy_image_2D_to_1x8_buffer";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_copy_image_2D_to_1x8_buffer_);
}

void Patch_match::_init_median_filter_float_disparity_3x3_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "median_3x3";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_median_filter_float_disparity_3x3_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_region_voting_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "region_voting";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_region_voting_, _kernel_defines_simple_patch_match());
}

void Patch_match::_init_compute_limits_kernel(cl_context const& context, cl_device_id const& device_id) {
    std::string const kernel_path = "./kernels/image_processing/binary/stereo_matching/matching_algorithms/simple_patch_match.cl";
    std::string const kernel_function_name = "compute_limits";

    _register_kernel(context, device_id, kernel_path, kernel_function_name, kernel_compute_limits_, _kernel_defines_simple_patch_match());
}




void Patch_match::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
   
    _init_plane_refinement_kernel(context, device_id);
   
    _init_convert_RGBA_to_R_kernel(context, device_id);
    
    _init_spatial_propagation_kernel(context, device_id);

    _init_view_propagation_kernel(context, device_id);

    _init_gradient_filter_kernel(context, device_id);

    _init_rgb_and_gradient_combiner_kernel(context, device_id);

    _init_copy_plane_to_last_plane(context, device_id);

    _init_random_initialization_kernel(context, device_id);
   
    _init_copy_grayscale_3x8_buffer_to_image_2D_kernel(context, device_id);

    _init_copy_bgr_3x8_buffer_to_image_kernel(context, device_id);

    _init_copy_image_2D_to_buffer_1x8_buffer_kernel(context, device_id);

    _init_optimize_outlier_detection_kernel(context, device_id);
    _init_optimize_fill_invalid_pixel_kernel(context, device_id);
    _init_outlier_detection_after_converting_RGBA_in_R(context, device_id);

    _init_disp_buffer_to_plane_image(context, device_id);
    
    _init_median_filter_float_disparity_3x3_kernel(context, device_id);

    _init_region_voting_kernel(context, device_id);

	_init_compute_limits_kernel(context, device_id);

    _init_temporal_propagation_kernel(context, device_id);

}

void Patch_match::set_width(int width){
    width_ = width;

}
void Patch_match::set_height(int height){
    height_ = height;
}


void Patch_match::_set_arg_and_run_copy_3_channel_buffer_to_image_gray(cl_command_queue const& command_queue, 
                                                                       cl_mem const &buffer, cl_mem const &input_image, int width, int height){
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_copy_grayscale_3x8_buffer_to_image_2D_, 0, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_copy_grayscale_3x8_buffer_to_image_2D_, 1, sizeof(cl_mem), (void *)&input_image);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_copy_grayscale_3x8_buffer_to_image_2D_, 2, sizeof(int), (void *) &width);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_copy_grayscale_3x8_buffer_to_image_2D_, 3, sizeof(int), (void *) &height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_copy_grayscale_3x8_buffer_to_image_2D_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - rgb buffer to image");


}

void Patch_match::_set_arg_and_run_bgr_3x8_buffer_image_copy(cl_command_queue const& command_queue, 
                                                                       cl_mem const &buffer, cl_mem const &input_image , int width, int height){
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_copy_bgr_3x8_buffer_to_image_, 0, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_copy_bgr_3x8_buffer_to_image_, 1, sizeof(cl_mem), (void *)&input_image);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_copy_bgr_3x8_buffer_to_image_, 2, sizeof(int), (void *) &width);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_copy_bgr_3x8_buffer_to_image_, 3, sizeof(int), (void *) &height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_copy_bgr_3x8_buffer_to_image_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - rgb buffer to rgb image");


}


void Patch_match::_set_arg_and_run_copy_image_to_1_channel_buffer_gray(cl_command_queue const& command_queue, 
                                                                        cl_mem const &input_image, cl_mem const &buffer, int num_iterations , int width, int height){
    std::map<int, cl_int> kernel_arg_statuses;

    kernel_arg_statuses[0] = clSetKernelArg(kernel_copy_image_2D_to_1x8_buffer_, 0, sizeof(cl_mem), (void *)&input_image);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_copy_image_2D_to_1x8_buffer_, 1, sizeof(cl_mem), (void *)&buffer);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_copy_image_2D_to_1x8_buffer_, 2, sizeof(int), (void *) &width);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_copy_image_2D_to_1x8_buffer_, 3, sizeof(int), (void *) &height);
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_copy_image_2D_to_1x8_buffer_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - image to rgb buffer");

}


void Patch_match::_set_arg_and_run_convet_RGBA_to_R(cl_command_queue const& command_queue,  
                                           cl_mem &plane_image, cl_mem const &disparity_image, 
										   int num_iterations, float max_disparity , int width, int height){

    std::map<int, cl_int> kernel_arg_statuses;
  

    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_convert_RGBA_to_R_, 0, sizeof(cl_mem), (void *)&plane_image);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_convert_RGBA_to_R_, 1, sizeof(cl_mem), (void *)&disparity_image);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_init_convert_RGBA_to_R_, 2, sizeof(int), (void*)&width); 
    kernel_arg_statuses[3] = clSetKernelArg(kernel_init_convert_RGBA_to_R_, 3, sizeof(int), (void*)&height);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_init_convert_RGBA_to_R_, 4, sizeof(float), (void*)&max_disparity);
    
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_convert_RGBA_to_R_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel");


}

void Patch_match::_set_arg_and_run_random_initialization(cl_command_queue const& command_queue, 
                                                                        cl_mem& plane_image, //cl_mem const &plane_image_right, 
                                                                        float max_disparity, float min_disparity, int width, int height){



    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_random_initialization_, 0, sizeof(int), (void*) &width);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_random_initialization_, 1, sizeof(int), (void*) &height);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_init_random_initialization_, 2, sizeof(cl_mem), (void*) &plane_image);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_init_random_initialization_, 3, sizeof(float), (void*) &max_disparity);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_init_random_initialization_, 4, sizeof(float), (void*) &min_disparity);
    
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }
    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_random_initialization_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - random_init");

}

void Patch_match::_set_arg_and_run_median_filter_float_disparity_3x3(cl_command_queue const& command_queue, 
                                                                                    cl_mem const& in_disparity_image, //cl_mem const &plane_image_right,
                                                                                    cl_mem const& in_color_image,
                                                                                    cl_mem const& outlier_mask, 
                                                                                    cl_mem& out_disparity_image , int width, int height){

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    //local work group size
    int wgWidth = 16;
    int wgHeight = 16;
    int local_width = (global_work_size[0] + wgWidth - 1) / 16;
    int local_height = (global_work_size[1] + wgHeight - 1) / 16;
    size_t local_work_size[2] = {size_t(wgWidth),size_t(wgHeight)};

    global_work_size[0] = size_t(local_width * 16);
    global_work_size[1] = size_t(local_height * 16);

    std::map<int, cl_int> kernel_arg_statuses;
    //normal median filter
    // kernel_arg_statuses[0] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 0, sizeof(cl_mem), (void*) &in_disparity_image);
    // kernel_arg_statuses[1] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 1, sizeof(cl_mem), (void*) &in_color_image);
    // kernel_arg_statuses[2] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 2, sizeof(cl_mem), (void*) &outlier_mask);
    // kernel_arg_statuses[3] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 3, sizeof(cl_mem), (void*) &out_disparity_image);

    //bilateral median filter without local buffer
    // kernel_arg_statuses[0] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 0, sizeof(cl_mem), (void*) &in_disparity_image);
    // kernel_arg_statuses[1] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 1, sizeof(cl_mem), (void*) &out_disparity_image);
    // kernel_arg_statuses[2] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 2, sizeof(int), (void*) &width);
    // kernel_arg_statuses[3] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 3, sizeof(int), (void*) &height); 

    //bilateral median filter with local buffer
    kernel_arg_statuses[0] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 0, sizeof(cl_mem), (void*) &in_disparity_image);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 1, sizeof(cl_mem), (void*) &out_disparity_image);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 2, sizeof(int), (void*) &width);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 3, sizeof(int), (void*) &height);    
    kernel_arg_statuses[4] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 4, sizeof(int), (void*) &wgWidth);
    kernel_arg_statuses[5] = clSetKernelArg(kernel_median_filter_float_disparity_3x3_, 5, sizeof(int), (void*) &wgHeight);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_median_filter_float_disparity_3x3_, 2, NULL, global_work_size, /* NULL */local_work_size, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - median_filter");

}


void Patch_match::_set_arg_and_run_propagation_iteration(cl_command_queue const& command_queue, 
                                                                        cl_mem &plane_image_left_a, cl_mem &plane_image_left_b,
                                                                        cl_mem &plane_image_right_a, cl_mem &plane_image_right_b,
									  			                        cl_mem const &input_image_left, cl_mem const &input_image_right,
                                                                        cl_mem &cl_plane_image_last_left, cl_mem &cl_plane_image_last_right,
                                                                        cl_mem& cl_gradient_image_left, cl_mem& cl_gradient_image_right,
												                        int iteration_index, 
                                                                        float max_disparity, int radius, int compute_right, int switch_temp_prop , int width, int height){




    size_t global_work_size_propagation[2]  = { size_t(width), size_t(height)};

    size_t global_work_size_plane_refinement[2] = {size_t(width), size_t(height)};

    size_t global_work_size_red_black_propagation[2]  = { size_t(width), size_t(height/2)};

    int32_t propagation_index = 0;

    int const left = 0;
    int const right = 1;

    int const black = 0;
    int const red = 1;

        propagation_index = iteration_index;

        //SPATIAL PROPAGATION LEFT    BLACK
        //////////////////
        // current spatial propagation iteration left image
        clSetKernelArg(kernel_init_spatial_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);  //current
        clSetKernelArg(kernel_init_spatial_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);  //adjunct
        clSetKernelArg(kernel_init_spatial_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_a);
        clSetKernelArg(kernel_init_spatial_propagation_, 3, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_spatial_propagation_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        clSetKernelArg(kernel_init_spatial_propagation_, 5, sizeof(cl_mem), (void*) &plane_image_left_b);
        clSetKernelArg(kernel_init_spatial_propagation_, 6, sizeof(int), (void*)&width); 
        clSetKernelArg(kernel_init_spatial_propagation_, 7, sizeof(int), (void*)&height);
        clSetKernelArg(kernel_init_spatial_propagation_, 8, sizeof(int), (void*)&propagation_index);   
        clSetKernelArg(kernel_init_spatial_propagation_, 9, sizeof(int), (void*)&radius);
        clSetKernelArg(kernel_init_spatial_propagation_, 10, sizeof(int), (void*)&left);
        clSetKernelArg(kernel_init_spatial_propagation_, 11, sizeof(int), (void*)&black);  
        
        
        cl_int kernel_execution_status_propagation = clEnqueueNDRangeKernel(command_queue, kernel_init_spatial_propagation_, 2, NULL, 
                                                                            global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_propagation, "clEnqueueNDRangeKernel - pm propagation left");


        //PLANE REFINEMENT LEFT    BLACK
        /////////////////////////////
        // current plane refinement iteration left image
        clSetKernelArg(kernel_init_plane_refinement_, 0, sizeof(cl_mem), (void*) &input_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 1, sizeof(cl_mem), (void*) &input_image_right);
        clSetKernelArg(kernel_init_plane_refinement_, 2, sizeof(cl_mem), (void*) &plane_image_left_b);
        clSetKernelArg(kernel_init_plane_refinement_, 3, sizeof(cl_mem), (void*) &plane_image_left_a);
        clSetKernelArg(kernel_init_plane_refinement_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 5, sizeof(cl_mem), (void*) &cl_gradient_image_right);        
        clSetKernelArg(kernel_init_plane_refinement_, 6, sizeof(int), (void*) &width);
        clSetKernelArg(kernel_init_plane_refinement_, 7, sizeof(int), (void*) &height);
        clSetKernelArg(kernel_init_plane_refinement_, 8, sizeof(float), (void*) &max_disparity); 
        clSetKernelArg(kernel_init_plane_refinement_, 9, sizeof(int), (void*) &radius);
        clSetKernelArg(kernel_init_plane_refinement_, 10, sizeof(int), (void*) &left);    
        clSetKernelArg(kernel_init_plane_refinement_, 11, sizeof(int), (void*)&black); 
        clSetKernelArg(kernel_init_plane_refinement_, 12, sizeof(int), (void*)&propagation_index);

        //std::cout << "Executing refinement " << "\n";

        cl_int kernel_execution_status_plane_refinement = clEnqueueNDRangeKernel(command_queue, kernel_init_plane_refinement_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_plane_refinement, "clEnqueueNDRangeKernel - pm plane refinement left");

        
        

        //SPATIAL PROPAGATION RIGHT     BLACK
        // current spatial propagation iteration right image
        clSetKernelArg(kernel_init_spatial_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);  //current
        clSetKernelArg(kernel_init_spatial_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);  //adjunct
        clSetKernelArg(kernel_init_spatial_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_right_a);
        clSetKernelArg(kernel_init_spatial_propagation_, 3, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_spatial_propagation_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        clSetKernelArg(kernel_init_spatial_propagation_, 5, sizeof(cl_mem), (void*) &plane_image_right_b);
        clSetKernelArg(kernel_init_spatial_propagation_, 6, sizeof(int), (void*)&width); 
        clSetKernelArg(kernel_init_spatial_propagation_, 7, sizeof(int), (void*)&height);
        clSetKernelArg(kernel_init_spatial_propagation_, 8, sizeof(int), (void*)&propagation_index);
        clSetKernelArg(kernel_init_spatial_propagation_, 9, sizeof(int), (void*)&radius);
        clSetKernelArg(kernel_init_spatial_propagation_, 10, sizeof(int), (void*)&right);    
        clSetKernelArg(kernel_init_spatial_propagation_, 11, sizeof(int), (void*)&black);  
        
        kernel_execution_status_propagation = clEnqueueNDRangeKernel(command_queue, kernel_init_spatial_propagation_, 2, NULL, 
                                                                            global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_propagation, "clEnqueueNDRangeKernel - pm propagation right");


        
        //PLANE REFINEMENT RIGHT    BLACK
        // current plane refinement iteration right image
        clSetKernelArg(kernel_init_plane_refinement_, 0, sizeof(cl_mem), (void*) &input_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 1, sizeof(cl_mem), (void*) &input_image_right);
        clSetKernelArg(kernel_init_plane_refinement_, 2, sizeof(cl_mem), (void*) &plane_image_right_b);
        clSetKernelArg(kernel_init_plane_refinement_, 3, sizeof(cl_mem), (void*) &plane_image_right_a);
        clSetKernelArg(kernel_init_plane_refinement_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 5, sizeof(cl_mem), (void*) &cl_gradient_image_right); 
        clSetKernelArg(kernel_init_plane_refinement_, 6, sizeof(int), (void*) &width);
        clSetKernelArg(kernel_init_plane_refinement_, 7, sizeof(int), (void*) &height);
        clSetKernelArg(kernel_init_plane_refinement_, 8, sizeof(float), (void*) &max_disparity); 
        clSetKernelArg(kernel_init_plane_refinement_, 9, sizeof(int), (void*) &radius);
        clSetKernelArg(kernel_init_plane_refinement_, 10, sizeof(int), (void*) &right);    
        clSetKernelArg(kernel_init_plane_refinement_, 11, sizeof(int), (void*)&black); 
        clSetKernelArg(kernel_init_plane_refinement_, 12, sizeof(int), (void*)&propagation_index);

        kernel_execution_status_plane_refinement = clEnqueueNDRangeKernel(command_queue, kernel_init_plane_refinement_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_plane_refinement, "clEnqueueNDRangeKernel - pm plane refinement right");


        /*
        if (switch_temp_prop == WITH_TEMP_PROP){
            clSetKernelArg(kernel_temporal_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);
            clSetKernelArg(kernel_temporal_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);
            clSetKernelArg(kernel_temporal_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_a);
            clSetKernelArg(kernel_temporal_propagation_, 3, sizeof(cl_mem), (void*) &plane_image_right_a);
            clSetKernelArg(kernel_temporal_propagation_, 4, sizeof(cl_mem), (void*) &cl_plane_image_last_left);
            clSetKernelArg(kernel_temporal_propagation_, 5, sizeof(cl_mem), (void*) &cl_plane_image_last_right);
            clSetKernelArg(kernel_temporal_propagation_, 6, sizeof(cl_mem), (void*) &cl_gradient_image_left);
            clSetKernelArg(kernel_temporal_propagation_, 7, sizeof(cl_mem), (void*) &cl_gradient_image_right);
            clSetKernelArg(kernel_temporal_propagation_, 8, sizeof(cl_mem), (void*) &plane_image_left_b);
            clSetKernelArg(kernel_temporal_propagation_, 9, sizeof(cl_mem), (void*) &plane_image_right_b);                
            clSetKernelArg(kernel_temporal_propagation_, 10, sizeof(int), (void*) &width);
            clSetKernelArg(kernel_temporal_propagation_, 11, sizeof(int), (void*) &height);
            clSetKernelArg(kernel_temporal_propagation_, 12, sizeof(int), (void*) &radius);
            clSetKernelArg(kernel_temporal_propagation_, 13, sizeof(int), (void*)&black);  

            cl_int kernel_execution_status_temporal_propagation = clEnqueueNDRangeKernel(command_queue, kernel_temporal_propagation_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

            dsm::check_cl_error(kernel_execution_status_temporal_propagation, "clEnqueueNDRangeKernel - temporal propagation");
            clFlush(command_queue);

            std::swap(plane_image_right_b, plane_image_right_a);
            std::swap(plane_image_left_b, plane_image_left_a);

        // }else{
        //     std::swap(plane_image_right_b, plane_image_right_a);
        //     std::swap(plane_image_left_b, plane_image_left_a);
        }

        // std::swap(plane_image_right_b, plane_image_right_a);
        // std::swap(plane_image_left_b, plane_image_left_a);
       
        */
    //REEEEEEEEEEEEEEEEEEED

        // //SPATIAL PROPAGATION LEFT               RED
        // //////////////////
        // //current spatial propagation iteration left image


        clSetKernelArg(kernel_init_spatial_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);  //current
        clSetKernelArg(kernel_init_spatial_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);  //adjunct
        clSetKernelArg(kernel_init_spatial_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_a);
        clSetKernelArg(kernel_init_spatial_propagation_, 3, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_spatial_propagation_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        clSetKernelArg(kernel_init_spatial_propagation_, 5, sizeof(cl_mem), (void*) &plane_image_left_b);
        clSetKernelArg(kernel_init_spatial_propagation_, 6, sizeof(int), (void*)&width); 
        clSetKernelArg(kernel_init_spatial_propagation_, 7, sizeof(int), (void*)&height);
        clSetKernelArg(kernel_init_spatial_propagation_, 8, sizeof(int), (void*)&propagation_index);   
        clSetKernelArg(kernel_init_spatial_propagation_, 9, sizeof(int), (void*)&radius);
        clSetKernelArg(kernel_init_spatial_propagation_, 10, sizeof(int), (void*)&left);
        clSetKernelArg(kernel_init_spatial_propagation_, 11, sizeof(int), (void*)&red);  
        

        kernel_execution_status_propagation = clEnqueueNDRangeKernel(command_queue, kernel_init_spatial_propagation_, 2, NULL, 
                                                                            global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_propagation, "clEnqueueNDRangeKernel - pm propagation left");



  
        //PLANE REFINEMENT LEFT                     RED
        /////////////////////////////
        // current plane refinement iteration left image
        clSetKernelArg(kernel_init_plane_refinement_, 0, sizeof(cl_mem), (void*) &input_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 1, sizeof(cl_mem), (void*) &input_image_right);
        clSetKernelArg(kernel_init_plane_refinement_, 2, sizeof(cl_mem), (void*) &plane_image_left_b);
        clSetKernelArg(kernel_init_plane_refinement_, 3, sizeof(cl_mem), (void*) &plane_image_left_a);
        clSetKernelArg(kernel_init_plane_refinement_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 5, sizeof(cl_mem), (void*) &cl_gradient_image_right);        
        clSetKernelArg(kernel_init_plane_refinement_, 6, sizeof(int), (void*) &width);
        clSetKernelArg(kernel_init_plane_refinement_, 7, sizeof(int), (void*) &height);
        clSetKernelArg(kernel_init_plane_refinement_, 8, sizeof(float), (void*) &max_disparity); 
        clSetKernelArg(kernel_init_plane_refinement_, 9, sizeof(int), (void*) &radius);
        clSetKernelArg(kernel_init_plane_refinement_, 10, sizeof(int), (void*) &left);    
        clSetKernelArg(kernel_init_plane_refinement_, 11, sizeof(int), (void*)&red); 
        clSetKernelArg(kernel_init_plane_refinement_, 12, sizeof(int), (void*)&propagation_index);

        //std::cout << "Executing refinement " << "\n";

        kernel_execution_status_plane_refinement = clEnqueueNDRangeKernel(command_queue, kernel_init_plane_refinement_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_plane_refinement, "clEnqueueNDRangeKernel - pm plane refinement left");



        //SPATIAL PROPAGATION RIGHT                    RED
        // current spatial propagation iteration right image
        clSetKernelArg(kernel_init_spatial_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);  //current
        clSetKernelArg(kernel_init_spatial_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);  //adjunct
        clSetKernelArg(kernel_init_spatial_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_right_a);
        clSetKernelArg(kernel_init_spatial_propagation_, 3, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_spatial_propagation_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        clSetKernelArg(kernel_init_spatial_propagation_, 5, sizeof(cl_mem), (void*) &plane_image_right_b);
        clSetKernelArg(kernel_init_spatial_propagation_, 6, sizeof(int), (void*)&width); 
        clSetKernelArg(kernel_init_spatial_propagation_, 7, sizeof(int), (void*)&height);
        clSetKernelArg(kernel_init_spatial_propagation_, 8, sizeof(int), (void*)&propagation_index);
        clSetKernelArg(kernel_init_spatial_propagation_, 9, sizeof(int), (void*)&radius);
        clSetKernelArg(kernel_init_spatial_propagation_, 10, sizeof(int), (void*)&right);    
        clSetKernelArg(kernel_init_spatial_propagation_, 11, sizeof(int), (void*)&red);  

        kernel_execution_status_propagation = clEnqueueNDRangeKernel(command_queue, kernel_init_spatial_propagation_, 2, NULL, 
                                                                            global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_propagation, "clEnqueueNDRangeKernel - pm propagation right");


        //PLANE REFINEMENT RIGHT                    RED
        // current plane refinement iteration right image
        clSetKernelArg(kernel_init_plane_refinement_, 0, sizeof(cl_mem), (void*) &input_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 1, sizeof(cl_mem), (void*) &input_image_right);
        clSetKernelArg(kernel_init_plane_refinement_, 2, sizeof(cl_mem), (void*) &plane_image_right_b);
        clSetKernelArg(kernel_init_plane_refinement_, 3, sizeof(cl_mem), (void*) &plane_image_right_a);
        clSetKernelArg(kernel_init_plane_refinement_, 4, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        clSetKernelArg(kernel_init_plane_refinement_, 5, sizeof(cl_mem), (void*) &cl_gradient_image_right); 
        clSetKernelArg(kernel_init_plane_refinement_, 6, sizeof(int), (void*) &width);
        clSetKernelArg(kernel_init_plane_refinement_, 7, sizeof(int), (void*) &height);
        clSetKernelArg(kernel_init_plane_refinement_, 8, sizeof(float), (void*) &max_disparity); 
        clSetKernelArg(kernel_init_plane_refinement_, 9, sizeof(int), (void*) &radius);
        clSetKernelArg(kernel_init_plane_refinement_, 10, sizeof(int), (void*) &right);    
        clSetKernelArg(kernel_init_plane_refinement_, 11, sizeof(int), (void*)&red); 
        clSetKernelArg(kernel_init_plane_refinement_, 12, sizeof(int), (void*)&propagation_index);

        kernel_execution_status_plane_refinement = clEnqueueNDRangeKernel(command_queue, kernel_init_plane_refinement_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

        dsm::check_cl_error(kernel_execution_status_plane_refinement, "clEnqueueNDRangeKernel - pm plane refinement right");


        //std::swap(plane_image_right_b, plane_image_right_a);
        //std::swap(plane_image_left_b, plane_image_left_a);
    
        
        if (switch_temp_prop == WITH_TEMP_PROP){
            clSetKernelArg(kernel_temporal_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);
            clSetKernelArg(kernel_temporal_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);
            clSetKernelArg(kernel_temporal_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_a);
            clSetKernelArg(kernel_temporal_propagation_, 3, sizeof(cl_mem), (void*) &plane_image_right_a);
            clSetKernelArg(kernel_temporal_propagation_, 4, sizeof(cl_mem), (void*) &cl_plane_image_last_left);
            clSetKernelArg(kernel_temporal_propagation_, 5, sizeof(cl_mem), (void*) &cl_plane_image_last_right);
            clSetKernelArg(kernel_temporal_propagation_, 6, sizeof(cl_mem), (void*) &cl_gradient_image_left);
            clSetKernelArg(kernel_temporal_propagation_, 7, sizeof(cl_mem), (void*) &cl_gradient_image_right);
            clSetKernelArg(kernel_temporal_propagation_, 8, sizeof(cl_mem), (void*) &plane_image_left_b);
            clSetKernelArg(kernel_temporal_propagation_, 9, sizeof(cl_mem), (void*) &plane_image_right_b);                
            clSetKernelArg(kernel_temporal_propagation_, 10, sizeof(int), (void*) &width);
            clSetKernelArg(kernel_temporal_propagation_, 11, sizeof(int), (void*) &height);
            clSetKernelArg(kernel_temporal_propagation_, 12, sizeof(int), (void*) &radius);
            clSetKernelArg(kernel_temporal_propagation_, 13, sizeof(int), (void*)&red);  

            cl_int kernel_execution_status_temporal_propagation = clEnqueueNDRangeKernel(command_queue, kernel_temporal_propagation_, 2, NULL,
                                                                                 global_work_size_red_black_propagation, NULL, 0, NULL, NULL);

            dsm::check_cl_error(kernel_execution_status_temporal_propagation, "clEnqueueNDRangeKernel - temporal propagation");


        }else{
            std::swap(plane_image_right_b, plane_image_right_a);
            std::swap(plane_image_left_b, plane_image_left_a);
        }

        
        


        // #if 0
        //   std::swap(plane_image_right_b, plane_image_right_a);
        //   std::swap(plane_image_left_b, plane_image_left_a);
        // #else

        

        
          //VIEW PROPAGATION LEFT RIGHT
          /////////////////////////////
        //    clSetKernelArg(kernel_init_view_propagation_, 0, sizeof(cl_mem), (void*) &plane_image_left_b);
        //    clSetKernelArg(kernel_init_view_propagation_, 1, sizeof(cl_mem), (void*) &plane_image_right_b);
        //    clSetKernelArg(kernel_init_view_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_a);
        //    clSetKernelArg(kernel_init_view_propagation_, 3, sizeof(cl_mem), (void*) &plane_image_right_a);
        //    clSetKernelArg(kernel_init_view_propagation_, 4, sizeof(cl_mem), (void*) &input_image_left);
        //    clSetKernelArg(kernel_init_view_propagation_, 5, sizeof(cl_mem), (void*) &input_image_right);
        //    clSetKernelArg(kernel_init_view_propagation_, 6, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        //    clSetKernelArg(kernel_init_view_propagation_, 7, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        //    clSetKernelArg(kernel_init_view_propagation_, 8, sizeof(int), (void*) &width);
        //    clSetKernelArg(kernel_init_view_propagation_, 9, sizeof(int), (void*) &height);
        //    clSetKernelArg(kernel_init_view_propagation_, 10, sizeof(float), (void*) &max_disparity); 
        //    clSetKernelArg(kernel_init_view_propagation_, 11, sizeof(int), (void*) &radius);

        //    cl_int kernel_execution_status_view_propagation = clEnqueueNDRangeKernel(command_queue, kernel_init_view_propagation_, 2, NULL,
        //                                                                             global_work_size_plane_refinement, NULL, 0, NULL, NULL);

        //    dsm::check_cl_error(kernel_execution_status_view_propagation, "clEnqueueNDRangeKernel - pm view propagation left");
        //    clFlush(command_queue);
        
        // #endif


       std::swap(plane_image_left_a, plane_image_left_b);
       std::swap(plane_image_right_a, plane_image_right_b);       
        //TEMPORAL PROPAGATION LEFT RIGHT
        /////////////////////////////////////////////
        // if (switch_temp_prop == WITH_TEMP_PROP){
        //     clSetKernelArg(kernel_temporal_propagation_, 0, sizeof(cl_mem), (void*) &input_image_left);
        //     clSetKernelArg(kernel_temporal_propagation_, 1, sizeof(cl_mem), (void*) &input_image_right);
        //     clSetKernelArg(kernel_temporal_propagation_, 2, sizeof(cl_mem), (void*) &plane_image_left_b);
        //     clSetKernelArg(kernel_temporal_propagation_, 3, sizeof(cl_mem), (void*) &plane_image_right_b);
        //     clSetKernelArg(kernel_temporal_propagation_, 4, sizeof(cl_mem), (void*) &cl_plane_image_last_left);
        //     clSetKernelArg(kernel_temporal_propagation_, 5, sizeof(cl_mem), (void*) &cl_plane_image_last_right);
        //     clSetKernelArg(kernel_temporal_propagation_, 6, sizeof(cl_mem), (void*) &cl_gradient_image_left);
        //     clSetKernelArg(kernel_temporal_propagation_, 7, sizeof(cl_mem), (void*) &cl_gradient_image_right);
        //     clSetKernelArg(kernel_temporal_propagation_, 8, sizeof(cl_mem), (void*) &plane_image_left_a);
        //     clSetKernelArg(kernel_temporal_propagation_, 9, sizeof(cl_mem), (void*) &plane_image_right_a);                
        //     clSetKernelArg(kernel_temporal_propagation_, 10, sizeof(int), (void*) &width);
        //     clSetKernelArg(kernel_temporal_propagation_, 11, sizeof(int), (void*) &height);
        //     clSetKernelArg(kernel_temporal_propagation_, 12, sizeof(int), (void*) &radius);

        //     cl_int kernel_execution_status_temporal_propagation = clEnqueueNDRangeKernel(command_queue, kernel_temporal_propagation_, 2, NULL,
        //                                                                          global_work_size_propagation, NULL, 0, NULL, NULL);

        //     dsm::check_cl_error(kernel_execution_status_temporal_propagation, "clEnqueueNDRangeKernel - temporal propagation");
        //     clFlush(command_queue);
        // }else{
        //     std::swap(plane_image_right_b, plane_image_right_a);
        //     std::swap(plane_image_left_b, plane_image_left_a);
        // }




        ////// 4. dummy 
        //std::swap(plane_image_right_b, plane_image_right_a);
        //std::swap(plane_image_left_b, plane_image_left_a);
    //std::cout << "x" << "\n";
    
}

void Patch_match::_set_arg_and_run_gradient_filter(cl_command_queue const& command_queue,
														cl_mem &input_image_left, cl_mem &input_image_right,
														cl_mem &gradient_image_left, cl_mem &gradient_image_right , int width, int height) {

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_gradient_filter, 0, sizeof(cl_mem), (void *) &input_image_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_gradient_filter, 1, sizeof(cl_mem), (void *) &gradient_image_left);
    // kernel_arg_statuses[2] = clSetKernelArg(kernel_init_gradient_filter, 2, sizeof(int), (void *) &width);
    // kernel_arg_statuses[3] = clSetKernelArg(kernel_init_gradient_filter, 3, sizeof(int), (void *) &height);        

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_gradient_filter, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - gradient_filter_left");


    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_gradient_filter, 0, sizeof(cl_mem), (void *) &input_image_right);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_gradient_filter, 1, sizeof(cl_mem), (void *) &gradient_image_right);

    kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_gradient_filter, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - gradient_filter_right");


}

void Patch_match::_set_arg_and_run_combine_rgb_and_gradient(cl_command_queue const& command_queue,
                                                                           cl_mem& in_rgb_left, cl_mem& in_rgb_right,
                                                                           cl_mem& in_gradient_image_left, cl_mem& in_gradient_image_right,
                                                                           cl_mem& out_rgb_grad_left, cl_mem& out_rgb_grad_right , int width, int height) {
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 0, sizeof(cl_mem), (void *) &in_rgb_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 1, sizeof(cl_mem), (void *) &in_rgb_right);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 2, sizeof(cl_mem), (void *) &in_gradient_image_left);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 3, sizeof(cl_mem), (void *) &in_gradient_image_right);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 4, sizeof(cl_mem), (void *) &out_rgb_grad_left);
    kernel_arg_statuses[5] = clSetKernelArg(kernel_init_rgb_and_gradient_combiner, 5, sizeof(cl_mem), (void *) &out_rgb_grad_right);
    // kernel_arg_statuses[2] = clSetKernelArg(kernel_init_gradient_filter, 2, sizeof(int), (void *) &width);
    // kernel_arg_statuses[3] = clSetKernelArg(kernel_init_gradient_filter, 3, sizeof(int), (void *) &height);        


    
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_rgb_and_gradient_combiner, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - rgb and gradient combiner");

   
}

void Patch_match::_set_arg_and_run_copy_plane_image_to_last_plane(cl_command_queue const& command_queue,
														cl_mem &final_plane_left, cl_mem &final_plane_right,
														cl_mem &last_plane_left, cl_mem &last_plane_right
                                                        , int width, int height) {

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_init_copy_plane_image_to_last_plane, 0, sizeof(cl_mem), (void *) &final_plane_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_init_copy_plane_image_to_last_plane, 1, sizeof(cl_mem), (void *) &final_plane_right);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_init_copy_plane_image_to_last_plane, 2, sizeof(cl_mem), (void *) &last_plane_left);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_init_copy_plane_image_to_last_plane, 3, sizeof(cl_mem), (void *) &last_plane_right);        

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_init_copy_plane_image_to_last_plane, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - copy plane image");



}

void Patch_match::_set_arg_and_run_outlier_detection_for_Convert_RGBA_in_R(cl_command_queue const& command_queue, 
											cl_mem const& disparity_image_left , cl_mem const& disparity_image_right, 
											cl_mem& disparity_image_out,
                                            cl_mem& cl_disparity_outlier_mask, 
											float min_disparity, float max_disparity , int width, int height){

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 0, sizeof(cl_mem), (void *)&disparity_image_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 1, sizeof(cl_mem), (void *)&disparity_image_right);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 2, sizeof(cl_mem), (void *)&disparity_image_out);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 3, sizeof(cl_mem), (void *)&cl_disparity_outlier_mask);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 4, sizeof(float), (void *)&min_disparity);
    kernel_arg_statuses[5] = clSetKernelArg(kernel_outlier_detection_after_converting_RGBA_in_R, 5, sizeof(float), (void *)&max_disparity);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_outlier_detection_after_converting_RGBA_in_R, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - outlier detection");


}


void Patch_match::_set_arg_and_run_outlier_detection(cl_command_queue const& command_queue, 
											cl_mem const& disparity_image_left , cl_mem const& disparity_image_right, 
											cl_mem& disparity_image_out,
                                            cl_mem& cl_disparity_outlier_mask, 
											float min_disparity, float max_disparity , int width, int height){

    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_optimize_outlier_detection_, 0, sizeof(cl_mem), (void *)&disparity_image_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_optimize_outlier_detection_, 1, sizeof(cl_mem), (void *)&disparity_image_right);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_optimize_outlier_detection_, 2, sizeof(cl_mem), (void *)&disparity_image_out);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_optimize_outlier_detection_, 3, sizeof(cl_mem), (void *)&cl_disparity_outlier_mask);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_optimize_outlier_detection_, 4, sizeof(float), (void *)&min_disparity);
    kernel_arg_statuses[5] = clSetKernelArg(kernel_optimize_outlier_detection_, 5, sizeof(float), (void *)&max_disparity);
    kernel_arg_statuses[6] = clSetKernelArg(kernel_optimize_outlier_detection_, 6, sizeof(int), (void *)&width);
    kernel_arg_statuses[7] = clSetKernelArg(kernel_optimize_outlier_detection_, 7, sizeof(int), (void *)&height);
    
    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_optimize_outlier_detection_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - outlier detection");



    //************************ FILL INVALID PIXEL *******************************
    
    kernel_arg_statuses[0] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 0, sizeof(cl_mem), (void *)&disparity_image_left);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 1, sizeof(cl_mem), (void *)&disparity_image_right);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 2, sizeof(cl_mem), (void *)&disparity_image_out);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 3, sizeof(cl_mem), (void *)&cl_disparity_outlier_mask);
    kernel_arg_statuses[4] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 4, sizeof(int), (void *)&width);
    kernel_arg_statuses[5] = clSetKernelArg(kernel_optimize_fill_invalid_pixel_, 5, sizeof(int), (void *)&height);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument INVALID PIXEL #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }

    //size_t global_work_size[2] = {size_t(width), size_t(height)};
    kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_optimize_fill_invalid_pixel_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - outlier detection - invalid pixel filling");
    

}

void Patch_match::_set_arg_and_run_compute_limits(cl_command_queue const& command_queue, 
                                    cl_mem disparity_image, cl_mem limits , int width, int height)
                                    
{


    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_compute_limits_, 0, sizeof(cl_mem), (void*) &disparity_image);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_compute_limits_, 1, sizeof(cl_mem), (void*) &limits);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_compute_limits_, 2, sizeof(int), (void*) &width);
    kernel_arg_statuses[3] = clSetKernelArg(kernel_compute_limits_, 3, sizeof(int), (void*) &height);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }
    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_compute_limits_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - compute limits");


}

void Patch_match::_set_arg_and_run_region_voting(cl_command_queue const& command_queue, cl_mem compute_disparity_image, 
                                    	cl_mem disparity_image_1, cl_mem outlier_mask_1,
										cl_mem disparity_image_2, cl_mem outlier_mask_2,
                                        cl_mem limits_left, cl_mem limits_right, 
										int min_disparity, int max_disparity , int width, int height)
{

int horizontal = 1;
int vertical = 0;
int num_region_voting = 1;
for(int iteration = 0; iteration < num_region_voting; ++iteration) {

    std::map<int, cl_int> kernel_arg_statuses;

    if(iteration == 0) {
        kernel_arg_statuses[0] = clSetKernelArg(kernel_region_voting_, 0, sizeof(cl_mem), (void*) &compute_disparity_image);
        kernel_arg_statuses[1] = clSetKernelArg(kernel_region_voting_, 1, sizeof(cl_mem), (void*) &outlier_mask_1);
        kernel_arg_statuses[2] = clSetKernelArg(kernel_region_voting_, 2, sizeof(cl_mem), (void*) &limits_left); 
        kernel_arg_statuses[3] = clSetKernelArg(kernel_region_voting_, 3, sizeof(cl_mem), (void*) &disparity_image_2); 
        kernel_arg_statuses[4] = clSetKernelArg(kernel_region_voting_, 4, sizeof(cl_mem), (void*) &outlier_mask_2);        
        kernel_arg_statuses[7] = clSetKernelArg(kernel_region_voting_, 7, sizeof(int), (void*) &horizontal);        
    }
    else if(iteration % 2 == 1) {
        kernel_arg_statuses[0] = clSetKernelArg(kernel_region_voting_, 0, sizeof(cl_mem), (void*) &disparity_image_2);
        kernel_arg_statuses[1] = clSetKernelArg(kernel_region_voting_, 1, sizeof(cl_mem), (void*) &outlier_mask_2);
        kernel_arg_statuses[2] = clSetKernelArg(kernel_region_voting_, 2, sizeof(cl_mem), (void*) &limits_left); 
        kernel_arg_statuses[3] = clSetKernelArg(kernel_region_voting_, 3, sizeof(cl_mem), (void*) &disparity_image_1); 
        kernel_arg_statuses[4] = clSetKernelArg(kernel_region_voting_, 4, sizeof(cl_mem), (void*) &outlier_mask_1);        
        kernel_arg_statuses[7] = clSetKernelArg(kernel_region_voting_, 7, sizeof(int), (void*) &vertical);        
    }
    else {
        kernel_arg_statuses[0] = clSetKernelArg(kernel_region_voting_, 0, sizeof(cl_mem), (void*) &disparity_image_1);
        kernel_arg_statuses[1] = clSetKernelArg(kernel_region_voting_, 1, sizeof(cl_mem), (void*) &outlier_mask_1);
        kernel_arg_statuses[2] = clSetKernelArg(kernel_region_voting_, 2, sizeof(cl_mem), (void*) &limits_left); 
        kernel_arg_statuses[3] = clSetKernelArg(kernel_region_voting_, 3, sizeof(cl_mem), (void*) &disparity_image_2); 
        kernel_arg_statuses[4] = clSetKernelArg(kernel_region_voting_, 4, sizeof(cl_mem), (void*) &outlier_mask_2);        
        kernel_arg_statuses[7] = clSetKernelArg(kernel_region_voting_, 7, sizeof(int), (void*) &horizontal); 
    }

        kernel_arg_statuses[5] = clSetKernelArg(kernel_region_voting_, 5, sizeof(int), (void*) &min_disparity);
        kernel_arg_statuses[6] = clSetKernelArg(kernel_region_voting_, 6, sizeof(int), (void*) &max_disparity);
    

        for(auto const& status_pair : kernel_arg_statuses) {
            if(CL_SUCCESS != status_pair.second) {
                std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                            + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
                DSM_LOG_ERROR(error_message);
            }
        }
        size_t global_work_size[2] = {size_t(width), size_t(height)};
        cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_region_voting_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
        dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - compute limits");
  
}
/*
clFlush(command_queue);
    std::map<int, cl_int> kernel_arg_statuses;
    kernel_arg_statuses[0] = clSetKernelArg(kernel_region_voting_, 0, sizeof(cl_mem), (void*) &compute_disparity_image);
    kernel_arg_statuses[1] = clSetKernelArg(kernel_region_voting_, 1, sizeof(cl_mem), (void*) &outlier_mask_1);
    kernel_arg_statuses[2] = clSetKernelArg(kernel_region_voting_, 2, sizeof(cl_mem), (void*) &limits_left); 
    kernel_arg_statuses[3] = clSetKernelArg(kernel_region_voting_, 3, sizeof(cl_mem), (void*) &disparity_image_2); 
    kernel_arg_statuses[4] = clSetKernelArg(kernel_region_voting_, 4, sizeof(cl_mem), (void*) &outlier_mask_2);           
    kernel_arg_statuses[5] = clSetKernelArg(kernel_region_voting_, 5, sizeof(int), (void*) &min_disparity);
    kernel_arg_statuses[6] = clSetKernelArg(kernel_region_voting_, 6, sizeof(int), (void*) &max_disparity);
    kernel_arg_statuses[7] = clSetKernelArg(kernel_region_voting_, 7, sizeof(int), (void*) &horizontal);    

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }
    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_region_voting_, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - compute limits");
    clFlush(command_queue);*/

}

void Patch_match::set_arg_and_run_disp_buffer_to_plane_image(cl_command_queue const& command_queue, 
                                            		cl_mem const &in_image_buffer, cl_mem& in_out_disp_image,cl_mem& out_plane_image_3, cl_mem& out_plane_image_4, 
                                                    int min_disp, int max_disp, int width, int height){

    std::map<int, cl_int> kernel_arg_statuses_1;
    kernel_arg_statuses_1[0] = clSetKernelArg(kernel_copy_1x8_buffer_to_unnormalized_disp_img, 0, sizeof(cl_mem), (void *)&in_image_buffer);
    kernel_arg_statuses_1[1] = clSetKernelArg(kernel_copy_1x8_buffer_to_unnormalized_disp_img, 1, sizeof(cl_mem), (void *)&in_out_disp_image);
    kernel_arg_statuses_1[2] = clSetKernelArg(kernel_copy_1x8_buffer_to_unnormalized_disp_img, 2, sizeof(int), (void *)&width);
    kernel_arg_statuses_1[3] = clSetKernelArg(kernel_copy_1x8_buffer_to_unnormalized_disp_img, 3, sizeof(int), (void *)&height);
    
    for(auto const& status_pair : kernel_arg_statuses_1) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }
    size_t global_work_size[2] = {size_t(width), size_t(height)};
    cl_int kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_copy_1x8_buffer_to_unnormalized_disp_img, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - copy buffer to unnormalized disparity image");

    /* DISPARITY IMAGE TO PLANE IMAGE*/
    std::map<int, cl_int> kernel_arg_statuses_2;
    kernel_arg_statuses_2[0] = clSetKernelArg(kernel_convert_disp_to_plane_img, 0, sizeof(cl_mem), (void *)&in_out_disp_image);
    kernel_arg_statuses_2[1] = clSetKernelArg(kernel_convert_disp_to_plane_img, 1, sizeof(cl_mem), (void *)&out_plane_image_3);
    kernel_arg_statuses_2[2] = clSetKernelArg(kernel_convert_disp_to_plane_img, 2, sizeof(cl_mem), (void *)&out_plane_image_4);
    kernel_arg_statuses_2[3] = clSetKernelArg(kernel_convert_disp_to_plane_img, 3, sizeof(int), (void *)&min_disp);
    kernel_arg_statuses_2[4] = clSetKernelArg(kernel_convert_disp_to_plane_img, 4, sizeof(int), (void *)&max_disp);    
    for(auto const& status_pair : kernel_arg_statuses_2) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first)
                                        + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }
    kernel_execution_status = clEnqueueNDRangeKernel(command_queue, kernel_convert_disp_to_plane_img, 2, NULL, global_work_size, NULL, 0, NULL, 0);
    dsm::check_cl_error(kernel_execution_status, "clEnqueueNDRangeKernel - convert disparity image to plane image");


}


std::string Patch_match::_kernel_defines_simple_patch_match()
{
    int num_disp = std::abs(int(maximum_disparity_ - minimum_disparity_));
    std::string defines = "#define NUM_DISPARITIES " + std::to_string(num_disp) + "\n";
    return defines;
}

void Patch_match::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {

    //image discriptor
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width_;
    desc.image_height = height_;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    //different image formats
    cl_image_format format_R_UNORM_INT8;
    format_R_UNORM_INT8.image_channel_order = CL_R;
    format_R_UNORM_INT8.image_channel_data_type = CL_UNORM_INT8;

    cl_image_format format_RGBA_UNORM_INT8;
    format_RGBA_UNORM_INT8.image_channel_order = CL_RGBA;
    format_RGBA_UNORM_INT8.image_channel_data_type = CL_UNORM_INT8;


    cl_image_format format_R_SNORM_INT8;
    format_R_SNORM_INT8.image_channel_order = CL_R;
    format_R_SNORM_INT8.image_channel_data_type = CL_SNORM_INT8;

    cl_image_format format_R_FLOAT;
    format_R_FLOAT.image_channel_order = CL_R;
    format_R_FLOAT.image_channel_data_type = CL_FLOAT;

    cl_image_format format_RGBA_32F;
    format_RGBA_32F.image_channel_order = CL_RGBA;
    format_RGBA_32F.image_channel_data_type = CL_FLOAT;

    cl_image_format format_R_32F;
    format_R_32F.image_channel_order = CL_R;
    format_R_32F.image_channel_data_type = CL_FLOAT;

    cl_left_input_image_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_UNORM_INT8, &desc,NULL, NULL);

    cl_right_input_image_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_UNORM_INT8, &desc,NULL, NULL);

    cl_disp_initial_guess_image_3_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);
    cl_disp_initial_guess_image_4_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);
    cl_plane_initial_guess_image_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_left_input_rgb_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                          &format_RGBA_UNORM_INT8, &desc, NULL, NULL);

    cl_right_input_rgb_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                          &format_RGBA_UNORM_INT8, &desc, NULL, NULL);

    cl_left_input_rgb_grad_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                          &format_RGBA_UNORM_INT8, &desc, NULL, NULL);

    cl_right_input_rgb_grad_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                          &format_RGBA_UNORM_INT8, &desc, NULL, NULL);                                       
    
    
    cl_plane_image_left_a_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_left_b_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_right_a_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);

    cl_plane_image_right_b_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);
    
    
   
    /*
    cl_plane_image_left_a_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);

    cl_plane_image_left_b_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);

    cl_plane_image_right_a_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);

    cl_plane_image_right_b_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_32F, &desc,NULL, NULL);
    */
    

    cl_disparity_image_left_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_FLOAT, &desc,NULL, NULL);

    cl_disparity_image_left_filtered_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                                      &format_R_FLOAT, &desc,NULL, NULL);

    cl_disparity_image_right_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_R_FLOAT, &desc,NULL, NULL);

    cl_disparity_image_outlier_detection_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                                &format_R_FLOAT, &desc,NULL, NULL);  

    cl_disparity_outlier_mask_1_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_R_UNORM_INT8, &desc,NULL, NULL);   
    
    cl_disparity_outlier_mask_2_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_R_FLOAT, &desc,NULL, NULL);   

    cl_disparity_region_voting_1_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_R_FLOAT, &desc,NULL, NULL); 
    cl_disparity_region_voting_2_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_R_FLOAT, &desc,NULL, NULL);  
    cl_limits_left_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_RGBA_32F, &desc,NULL, NULL);  
    cl_limits_right_ = clCreateImage(context, CL_MEM_READ_WRITE, 
                                    &format_RGBA_32F, &desc,NULL, NULL);  

    cl_gradient_image_left_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                            &format_R_SNORM_INT8, &desc, NULL, NULL);       
    cl_gradient_image_right_ = clCreateImage(context, CL_MEM_READ_WRITE,
                                            &format_R_SNORM_INT8, &desc, NULL, NULL);                                                                           

    cl_plane_image_last_left = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);
    cl_plane_image_last_right = clCreateImage(context, CL_MEM_READ_WRITE, 
                                &format_RGBA_32F, &desc,NULL, NULL);                                
                                  
}

void Patch_match::_cleanup_kernels() {
    clReleaseKernel(kernel_copy_grayscale_3x8_buffer_to_image_2D_);
    clReleaseKernel(kernel_copy_image_2D_to_1x8_buffer_ = 0);
    clReleaseKernel(kernel_temporal_propagation_ = 0);
    clReleaseKernel(kernel_init_random_initialization_);
    clReleaseKernel(kernel_init_spatial_propagation_ );
    clReleaseKernel(kernel_init_view_propagation_);
    clReleaseKernel(kernel_init_plane_refinement_);
    clReleaseKernel(kernel_init_view_propagation);
    clReleaseKernel(kernel_init_copy_plane_image_to_last_plane);
    clReleaseKernel(kernel_init_gradient_filter);
    clReleaseKernel(kernel_region_voting_);
    clReleaseKernel(kernel_compute_limits_);
    clReleaseKernel(kernel_init_convert_RGBA_to_R_);
    clReleaseKernel(kernel_optimize_outlier_detection_);
    clReleaseKernel(kernel_median_filter_float_disparity_3x3_);
    clReleaseKernel(kernel_copy_bgr_3x8_buffer_to_image_);
    clReleaseKernel(kernel_optimize_fill_invalid_pixel_);
    clReleaseKernel(kernel_outlier_detection_after_converting_RGBA_in_R);

/*     kernel_outlier_detection_after_converting_RGBA_in_R = 0;
	kernel_copy_bgr_3x8_buffer_to_image_ = 0;
	kernel_optimize_fill_invalid_pixel_ = 0;
    kernel_copy_grayscale_3x8_buffer_to_image_2D_ = 0;
    kernel_copy_image_2D_to_1x8_buffer_ = 0;
    kernel_temporal_propagation_ = 0;
    kernel_init_random_initialization_ = 0;
    kernel_init_spatial_propagation_ = 0;
    kernel_init_view_propagation_ = 0;
    kernel_init_plane_refinement_ = 0;
    kernel_init_view_propagation = 0;
    kernel_init_gradient_filter = 0;
    kernel_init_copy_plane_image_to_last_plane = 0;
    kernel_region_voting_ = 0;
    kernel_compute_limits_ = 0;
    kernel_init_convert_RGBA_to_R_ = 0;
    kernel_optimize_outlier_detection_ = 0;
    kernel_median_filter_float_disparity_3x3_ = 0; */
}

void Patch_match::_cleanup_memory_objects() {
    cl_int  status;
    status = clReleaseMemObject(cl_left_input_image_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_right_input_image_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_left_input_rgb_grad_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_right_input_rgb_grad_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_disp_initial_guess_image_3_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disp_initial_guess_image_4_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_plane_initial_guess_image_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_left_input_rgb_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_right_input_rgb_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_plane_image_left_a_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_plane_image_left_b_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_plane_image_right_a_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_plane_image_right_b_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_disparity_image_left_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_image_left_filtered_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_image_right_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_gradient_image_left_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_gradient_image_right_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_plane_image_last_left); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_plane_image_last_right); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_disparity_image_outlier_detection_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_region_voting_1_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_region_voting_2_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_outlier_mask_1_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_disparity_outlier_mask_2_); dsm::check_cl_error(status, "clReleaseMemObject");

    status = clReleaseMemObject(cl_limits_left_); dsm::check_cl_error(status, "clReleaseMemObject");
    status = clReleaseMemObject(cl_limits_right_); dsm::check_cl_error(status, "clReleaseMemObject");


/*     cl_left_input_image_ = 0;
    cl_right_input_image_ = 0;

    cl_disp_initial_guess_image_3_ = 0;
    cl_disp_initial_guess_image_4_ = 0;
    cl_plane_initial_guess_image_ = 0;

    cl_left_input_rgb_grad_ = 0;
    cl_right_input_rgb_grad_ = 0;

    cl_plane_image_left_a_ = 0;
    cl_plane_image_left_b_ = 0;
    cl_plane_image_right_a_ = 0;
    cl_plane_image_right_b_ = 0;

    cl_disparity_image_left_ = 0;
    cl_disparity_image_left_filtered_ = 0;
    cl_disparity_image_right_ = 0;

    cl_gradient_image_left_ = 0;
	cl_gradient_image_right_ = 0;

    cl_plane_image_last_left = 0;
    cl_plane_image_last_right = 0;

    cl_disparity_image_outlier_detection_ = 0;
    cl_disparity_region_voting_1_ = 0;
    cl_disparity_region_voting_2_ = 0;
    cl_disparity_outlier_mask_1_ = 0;
    cl_disparity_outlier_mask_2_ = 0;
    cl_limits_left_ = 0;
    cl_limits_right_ = 0; */
}
//}
