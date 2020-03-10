#ifndef DSM_GPU_BASE_IMAGE_PROCESSOR_H
#define DSM_GPU_BASE_IMAGE_PROCESSOR_H

#include <CL/cl.h>
#include <opencv2/opencv.hpp> //types

#include <map>
#include <memory> //smart pointers

/* GPUBaseImageProcessor is the base class for *any* image processing operation in this framework
   it just defines some interfaces which need to be implemented in the base classes

  There are two classes which inherit directly from this one: GPUImageProcessorUnary and GPUImageProcessorBinary,
  which operate on 1 or 2 input images, respectively and both produce exactly one output image
*/


namespace dsm {
class GPUBaseImageProcessor {
public:
	//actual functionality of the image processor, override in subclass
	//virtual void process(cl_command_queue const& command_queue, cl_mem const& in_image_buffer, cl_mem& out_image_buffer) const = 0;

	// derived classes such as GPUImageProcessorUnary an GPUImageProcessorBinary
	// define an interface called "process"

	virtual ~GPUBaseImageProcessor();

	void reload_kernels(cl_context const& context, cl_device_id const& device_id);

	static void print_and_clear_kernel_execution_times();

protected:
	virtual void _init_kernels(cl_context const& context, cl_device_id const& device_id) = 0;
	virtual void _init_memory_objects(cl_context const& context, cl_device_id const& device_id) = 0;

	virtual void _cleanup_kernels() = 0;
	virtual void _cleanup_memory_objects() = 0;
protected:
	GPUBaseImageProcessor(cv::Vec2i const& image_dims);

	void _cleanup_programs();

#ifdef ENABLE_KERNEL_PROFILING
	static void register_kernel_execution_time(cl_command_queue const& command_queue, cl_event const& timer_event, std::string const& label);
#endif

	cv::Vec2i image_dims_ = cv::Vec2i{-1, -1};
	std::vector<cl_program> cl_programs_;


#ifdef ENABLE_KERNEL_PROFILING
	// we want to clear this in a const function, because it is only a auxiliary object
	static std::multimap<std::string, double> kernel_execution_times_;
#endif

};

} //namespace dsm
#endif //DSM_GPU_BASE_IMAGE_PROCESSPOR_H