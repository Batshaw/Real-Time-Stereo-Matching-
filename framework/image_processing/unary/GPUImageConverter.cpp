#include <image_processing/unary/GPUImageConverter.h>

#include <core/utils.h>

#include <map>
#include <string>

namespace dsm {


// factory method for creating image converters
std::shared_ptr<GPUImageConverter> 
GPUImageConverter::create(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dimensions, ConversionMode const& conversion_mode) {
	std::shared_ptr<GPUImageConverter> color_converter_ptr = std::make_shared<GPUImageConverter>(context, device_id, image_dimensions, conversion_mode);
	return color_converter_ptr;
}

//non-static member functions
////////////

// constructor
GPUImageConverter::GPUImageConverter(cl_context const& context, cl_device_id const& device_id, cv::Vec2i const& image_dims, ConversionMode const& conversion_mode) 
    : GPUImageProcessorUnary(image_dims),
      conversion_mode_(conversion_mode) {
        _init_kernels(context, device_id);
        _init_memory_objects(context, device_id);
    }

// destructor
GPUImageConverter::~GPUImageConverter() {
    _cleanup_kernels();
}

void 
GPUImageConverter::set_mode(ConversionMode const& in_conversion_mode) {
    conversion_mode_ = in_conversion_mode;
};

void GPUImageConverter::process(cl_command_queue const& command_queue, cl_mem const& in_image_buffer, cl_mem& out_image_buffer) {
    
    // reference to kernel used used for our color conversion;
    auto const& map_iterator = cl_kernels_per_mode_.find(conversion_mode_);
    if(cl_kernels_per_mode_.end() == map_iterator) {
        std::cout << "Error: ConversionMode was not defined.\n";
        throw std::exception();
    }
    cl_kernel const& active_kernel = map_iterator->second;

    std::map<int, cl_int> kernel_arg_statuses; 
    kernel_arg_statuses[0] = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), (void *)&in_image_buffer);
    kernel_arg_statuses[1] = clSetKernelArg(active_kernel, 1, sizeof(cl_mem), (void *)&out_image_buffer);
    kernel_arg_statuses[2] = clSetKernelArg(active_kernel, 2, sizeof(int), (void *) &image_dims_[0]);
    kernel_arg_statuses[3] = clSetKernelArg(active_kernel, 3, sizeof(int), (void *) &image_dims_[1]);

    for(auto const& status_pair : kernel_arg_statuses) {
        if(CL_SUCCESS != status_pair.second) {
            std::string error_message = "Could not set kernel argument #" + std::to_string(status_pair.first) 
                      + ": " + std::to_string(status_pair.second) + ", " + dsm::get_cl_error_string(status_pair.second);
            DSM_LOG_ERROR(error_message);
        }
    }


size_t global_work_size[2] = {size_t(image_dims_[0]), size_t(image_dims_[1])};
#if ENABLE_KERNEL_PROFILING
    cl_event image_conversion_timer_event;
    clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &image_conversion_timer_event);

    register_kernel_execution_time(command_queue, image_conversion_timer_event,  get_filename_from_path(__FILE__) + ":: image_conversion_kernel");
#else
    clEnqueueNDRangeKernel(command_queue, active_kernel, 2, NULL, global_work_size, NULL, 0, NULL, 0);
#endif

}


void GPUImageConverter::_register_kernel(cl_context const& context, cl_device_id const& device_id,
                                        ConversionMode const& mode, std::string const& kernel_path, std::string const& kernel_function_name) {
    cl_program program_to_compile = 0;
    cl_kernel kernel_to_compile = 0;

    dsm::compile_kernel_from_file(context, device_id, kernel_path, kernel_function_name,
                                  program_to_compile, kernel_to_compile);

    // register compiled kernel in defined enum
    cl_kernels_per_mode_[mode] = kernel_to_compile;
    // assign a the name to kernel for profiling prints
    cl_kernel_names_per_mode_[mode] = kernel_function_name;

    cl_programs_.push_back(program_to_compile);
}


void GPUImageConverter::_init_bgr_3x8_to_grayscale_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_GRAYSCALE_3x8;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_grayscale_3x8_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_grayscale_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_bgr_3x8_to_binary_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {    
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_BINARY_3x8;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_binary_3x8_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_binary_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_bgr_3x8_to_rgb_3x8_kernel(cl_context const& context, cl_device_id const& device_id) {
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_RGB_3x8;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_rgb_3x8_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_rgb_3x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_bgr_3x8_to_grayscale_1x8_kernel(cl_context const& context, cl_device_id const& device_id) {    
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_GRAYSCALE_1x8;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_grayscale_1x8_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_grayscale_1x8";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_bgr_3x8_to_lab_3x16f_kernel(cl_context const& context, cl_device_id const& device_id) {
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_LAB_3x16F;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_lab_3x16f_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_lab_3x16f";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_bgr_3x8_to_lab_3x32f_kernel(cl_context const& context, cl_device_id const& device_id) {
    ConversionMode const mode = ConversionMode::BGR_3x8_TO_LAB_3x32F;
    std::string const kernel_path = "./kernels/image_processing/unary/conversion/bgr_3x8_to_lab_3x32f_conversion.cl";
    std::string const kernel_function_name = "convert_image_bgr_3x8_to_lab_3x32f";

    _register_kernel(context, device_id, mode, kernel_path, kernel_function_name);
}

void GPUImageConverter::_init_kernels(cl_context const& context, cl_device_id const& device_id) {
    //3x8 -> 3x8 modes
    _init_bgr_3x8_to_grayscale_3x8_kernel(context, device_id);
    _init_bgr_3x8_to_binary_3x8_kernel(context, device_id);    
    _init_bgr_3x8_to_rgb_3x8_kernel(context, device_id);

    //3x8 -> 1x8 modes
    _init_bgr_3x8_to_grayscale_1x8_kernel(context, device_id);

    //3x8 -> 3x16f modes
    //_init_bgr_3x8_to_lab_3x16f_kernel(context, device_id);
    
    //3x8 -> 3x32f modes
    _init_bgr_3x8_to_lab_3x32f_kernel(context, device_id);

}

void GPUImageConverter::_init_memory_objects(cl_context const& context, cl_device_id const& device_id) {
    //no intermediate passes are required for the color converter
}

void GPUImageConverter::_cleanup_kernels() {
    for(auto& map_kernel_iterator : cl_kernels_per_mode_) {

        if(NULL != map_kernel_iterator.second) {
            clReleaseKernel(map_kernel_iterator.second);
            map_kernel_iterator.second = NULL;
        }
    }

    //programs are released in destructor of GPUBaseImageProcessor
}

void GPUImageConverter::_cleanup_memory_objects() {
    //no intermediate passes are required for the color converter
}


}