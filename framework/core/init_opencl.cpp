#include "init_opencl.h"
#include <opencv2/core/ocl.hpp>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <fstream>

namespace dsm {

void initialize_cl_environment(cl_context& out_context, cl_device_id& out_device_id, cl_command_queue& out_command_queue, bool attached_to_opencv) {
	std::cout << "Initializing CL environment for 1 Device and 1 command queue\n";

	cl_uint num_platforms = 0;
	cl_platform_id platform = NULL;
	cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
	if (CL_SUCCESS != status ) {
		std::cout << "Error: Getting platforms!" << std::endl;
		return;
	}

	/*Choose the first available platform. */
	if (num_platforms > 0)
	{
		cl_platform_id* platforms = 
                     (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	/*Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint num_devices = 0;
	cl_device_id* device_ids = nullptr;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	//no GPU available.
	if (num_devices == 0) {
		std::cout << "No GPU device available." << std::endl;

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);

		if(num_devices == 0) {
		  std::cout << "Also no CPU device available. Aborting" << std::endl;
		  exit(-1);
		} else {

		  device_ids = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
                  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices, device_ids, NULL);
		
		  std::cout << "Using CPU Device.\n";
		}

	}
	else {
		device_ids = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device_ids, NULL);
	}

	out_device_id = device_ids[0];

	//Create context
	out_context = clCreateContext(NULL, 1, &out_device_id, NULL, NULL, NULL);


	cl_queue_properties queue_properties[3] = {0, 0, 0};


#if ENABLE_KERNEL_PROFILING
		queue_properties[0] = CL_QUEUE_PROPERTIES;
		queue_properties[1] = CL_QUEUE_PROFILING_ENABLE;
#endif

	//Creating command queue associate with the context
	out_command_queue = clCreateCommandQueueWithProperties(out_context, out_device_id, queue_properties, NULL);

	if (attached_to_opencv){
        PlatformInfo platform_info;
        platform_info.QueryInfo(platform);
        // attach OpenCL context to OpenCV
        cv::ocl::attachContext(platform_info.Name(), platform, out_context, out_device_id);
	}

	return;
}


} //end dsm
