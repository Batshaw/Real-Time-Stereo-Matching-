#include <image_processing/base/GPUBaseImageProcessor.h>

#include <core/utils.h>

namespace dsm {

#ifdef ENABLE_KERNEL_PROFILING
std::multimap<std::string, double> GPUBaseImageProcessor::kernel_execution_times_;
#endif

GPUBaseImageProcessor::GPUBaseImageProcessor(cv::Vec2i const& image_dims) : image_dims_(image_dims) {
}

GPUBaseImageProcessor::~GPUBaseImageProcessor() {
	_cleanup_programs();
};

void GPUBaseImageProcessor::reload_kernels(cl_context const& context, cl_device_id const& device_id) {
    _init_kernels(context, device_id);
    DSM_LOG_INFO("Reloaded Shaders.");
}


void GPUBaseImageProcessor::_cleanup_programs() {
	for(auto& program : cl_programs_) {
        if(NULL != program) {
          clReleaseProgram(program);
          program = NULL;
        } else {
            std::cout << "GPUImageColorConverter: Could not release program" << std::endl;
        }
    }
}


#ifdef ENABLE_KERNEL_PROFILING
void 
GPUBaseImageProcessor::register_kernel_execution_time(cl_command_queue const& command_queue, cl_event const& timer_event, std::string const& label) {
    clWaitForEvents(1, &timer_event);
    clFinish(command_queue);
    cl_ulong time_start; cl_ulong time_end;
    clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double passed_nano_seconds = time_end-time_start;

    std::pair<std::string, double> label_time_pair = std::make_pair(label, passed_nano_seconds);

    kernel_execution_times_.insert(label_time_pair);
}

void 
GPUBaseImageProcessor::print_and_clear_kernel_execution_times() {

    double total_kernel_execution_time = 0.0;

    for(auto const& kernel_execution_time_entry : kernel_execution_times_) {
        total_kernel_execution_time += kernel_execution_time_entry.second / 1000000.0;
    }


	std::cout << std::endl;
	std::cout << "======================="  << std::endl;
	std::cout << "KERNEL EXECUTION TIMES: " << std::endl;
	std::cout << "======================="  << std::endl;
	for(auto const& kernel_execution_time_entry : kernel_execution_times_) {
		std::cout << kernel_execution_time_entry.first << ": "
				  << kernel_execution_time_entry.second / 1000000.0 << " ms" << std::endl;
	}
    std::cout << "-----------------------" << std::endl;
    std::cout << "total time: " << total_kernel_execution_time << " ms" << std::endl;

	kernel_execution_times_.clear();
}
#endif //ENABLE_KERNEL_PROFILING

} //namespace dsm