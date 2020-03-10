#ifndef DSM_INIT_OPENCL_H
#define DSM_INIT_OPENCL_H

//DSM_INIT_OPENCL_H

#include <fstream>
#include <CL/cl.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>

namespace dsm {

    //https://github.com/opencv/opencv/blob/master/samples/opencl/opencl-opencv-interop.cpp
    class PlatformInfo
    {
    public:
        PlatformInfo()
        {}

        ~PlatformInfo()
        {}

        cl_int QueryInfo(cl_platform_id id)
        {
            query_param(id, CL_PLATFORM_PROFILE, m_profile);
            query_param(id, CL_PLATFORM_VERSION, m_version);
            query_param(id, CL_PLATFORM_NAME, m_name);
            query_param(id, CL_PLATFORM_VENDOR, m_vendor);
            query_param(id, CL_PLATFORM_EXTENSIONS, m_extensions);
            return CL_SUCCESS;
        }

        std::string Profile()    { return m_profile; }
        std::string Version()    { return m_version; }
        std::string Name()       { return m_name; }
        std::string Vendor()     { return m_vendor; }
        std::string Extensions() { return m_extensions; }

    private:
        cl_int query_param(cl_platform_id id, cl_platform_info param, std::string& paramStr)
        {
            cl_int res;

            size_t psize;
            cv::AutoBuffer<char> buf;

            res = clGetPlatformInfo(id, param, 0, 0, &psize);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetPlatformInfo failed"));

            buf.resize(psize);
            res = clGetPlatformInfo(id, param, psize, buf, 0);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetPlatformInfo failed"));

            // just in case, ensure trailing zero for ASCIIZ string
            buf[psize] = 0;

            paramStr = buf;

            return CL_SUCCESS;
        }

    private:
        std::string m_profile;
        std::string m_version;
        std::string m_name;
        std::string m_vendor;
        std::string m_extensions;
    };

    class DeviceInfo
    {
    public:
        DeviceInfo()
        {}

        ~DeviceInfo()
        {}

        cl_int QueryInfo(cl_device_id id)
        {
            query_param(id, CL_DEVICE_TYPE, m_type);
            query_param(id, CL_DEVICE_VENDOR_ID, m_vendor_id);
            query_param(id, CL_DEVICE_MAX_COMPUTE_UNITS, m_max_compute_units);
            query_param(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, m_max_work_item_dimensions);
            query_param(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, m_max_work_item_sizes);
            query_param(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, m_max_work_group_size);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, m_preferred_vector_width_char);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, m_preferred_vector_width_short);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, m_preferred_vector_width_int);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, m_preferred_vector_width_long);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, m_preferred_vector_width_float);
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, m_preferred_vector_width_double);
#if defined(CL_VERSION_1_1)
            query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, m_preferred_vector_width_half);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, m_native_vector_width_char);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, m_native_vector_width_short);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, m_native_vector_width_int);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, m_native_vector_width_long);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, m_native_vector_width_float);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, m_native_vector_width_double);
            query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, m_native_vector_width_half);
#endif
            query_param(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, m_max_clock_frequency);
            query_param(id, CL_DEVICE_ADDRESS_BITS, m_address_bits);
            query_param(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, m_max_mem_alloc_size);
            query_param(id, CL_DEVICE_IMAGE_SUPPORT, m_image_support);
            query_param(id, CL_DEVICE_MAX_READ_IMAGE_ARGS, m_max_read_image_args);
            query_param(id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, m_max_write_image_args);
#if defined(CL_VERSION_2_0)
            query_param(id, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, m_max_read_write_image_args);
#endif
            query_param(id, CL_DEVICE_IMAGE2D_MAX_WIDTH, m_image2d_max_width);
            query_param(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, m_image2d_max_height);
            query_param(id, CL_DEVICE_IMAGE3D_MAX_WIDTH, m_image3d_max_width);
            query_param(id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, m_image3d_max_height);
            query_param(id, CL_DEVICE_IMAGE3D_MAX_DEPTH, m_image3d_max_depth);
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, m_image_max_buffer_size);
            query_param(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, m_image_max_array_size);
#endif
            query_param(id, CL_DEVICE_MAX_SAMPLERS, m_max_samplers);
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, m_image_pitch_alignment);
            query_param(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, m_image_base_address_alignment);
#endif
#if defined(CL_VERSION_2_0)
            query_param(id, CL_DEVICE_MAX_PIPE_ARGS, m_max_pipe_args);
            query_param(id, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, m_pipe_max_active_reservations);
            query_param(id, CL_DEVICE_PIPE_MAX_PACKET_SIZE, m_pipe_max_packet_size);
#endif
            query_param(id, CL_DEVICE_MAX_PARAMETER_SIZE, m_max_parameter_size);
            query_param(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, m_mem_base_addr_align);
            query_param(id, CL_DEVICE_SINGLE_FP_CONFIG, m_single_fp_config);
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_DOUBLE_FP_CONFIG, m_double_fp_config);
#endif
            query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, m_global_mem_cache_type);
            query_param(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, m_global_mem_cacheline_size);
            query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, m_global_mem_cache_size);
            query_param(id, CL_DEVICE_GLOBAL_MEM_SIZE, m_global_mem_size);
            query_param(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, m_max_constant_buffer_size);
            query_param(id, CL_DEVICE_MAX_CONSTANT_ARGS, m_max_constant_args);
#if defined(CL_VERSION_2_0)
            query_param(id, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, m_max_global_variable_size);
            query_param(id, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, m_global_variable_preferred_total_size);
#endif
            query_param(id, CL_DEVICE_LOCAL_MEM_TYPE, m_local_mem_type);
            query_param(id, CL_DEVICE_LOCAL_MEM_SIZE, m_local_mem_size);
            query_param(id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, m_error_correction_support);
#if defined(CL_VERSION_1_1)
            query_param(id, CL_DEVICE_HOST_UNIFIED_MEMORY, m_host_unified_memory);
#endif
            query_param(id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, m_profiling_timer_resolution);
            query_param(id, CL_DEVICE_ENDIAN_LITTLE, m_endian_little);
            query_param(id, CL_DEVICE_AVAILABLE, m_available);
            query_param(id, CL_DEVICE_COMPILER_AVAILABLE, m_compiler_available);
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_LINKER_AVAILABLE, m_linker_available);
#endif
            query_param(id, CL_DEVICE_EXECUTION_CAPABILITIES, m_execution_capabilities);
            query_param(id, CL_DEVICE_QUEUE_PROPERTIES, m_queue_properties);
#if defined(CL_VERSION_2_0)
            query_param(id, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, m_queue_on_host_properties);
            query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, m_queue_on_device_properties);
            query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, m_queue_on_device_preferred_size);
            query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, m_queue_on_device_max_size);
            query_param(id, CL_DEVICE_MAX_ON_DEVICE_QUEUES, m_max_on_device_queues);
            query_param(id, CL_DEVICE_MAX_ON_DEVICE_EVENTS, m_max_on_device_events);
#endif
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_BUILT_IN_KERNELS, m_built_in_kernels);
#endif
            query_param(id, CL_DEVICE_PLATFORM, m_platform);
            query_param(id, CL_DEVICE_NAME, m_name);
            query_param(id, CL_DEVICE_VENDOR, m_vendor);
            query_param(id, CL_DRIVER_VERSION, m_driver_version);
            query_param(id, CL_DEVICE_PROFILE, m_profile);
            query_param(id, CL_DEVICE_VERSION, m_version);
#if defined(CL_VERSION_1_1)
            query_param(id, CL_DEVICE_OPENCL_C_VERSION, m_opencl_c_version);
#endif
            query_param(id, CL_DEVICE_EXTENSIONS, m_extensions);
#if defined(CL_VERSION_1_2)
            query_param(id, CL_DEVICE_PRINTF_BUFFER_SIZE, m_printf_buffer_size);
            query_param(id, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, m_preferred_interop_user_sync);
            query_param(id, CL_DEVICE_PARENT_DEVICE, m_parent_device);
            query_param(id, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, m_partition_max_sub_devices);
            query_param(id, CL_DEVICE_PARTITION_PROPERTIES, m_partition_properties);
            query_param(id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, m_partition_affinity_domain);
            query_param(id, CL_DEVICE_PARTITION_TYPE, m_partition_type);
            query_param(id, CL_DEVICE_REFERENCE_COUNT, m_reference_count);
#endif
            return CL_SUCCESS;
        }

        std::string Name() { return m_name; }

    private:
        template<typename T>
        cl_int query_param(cl_device_id id, cl_device_info param, T& value)
        {
            cl_int res;
            size_t size = 0;

            res = clGetDeviceInfo(id, param, 0, 0, &size);
            if (CL_SUCCESS != res && size != 0)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            if (0 == size)
                return CL_SUCCESS;

            if (sizeof(T) != size)
                throw std::runtime_error(std::string("clGetDeviceInfo: param size mismatch"));

            res = clGetDeviceInfo(id, param, size, &value, 0);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            return CL_SUCCESS;
        }

        template<typename T>
        cl_int query_param(cl_device_id id, cl_device_info param, std::vector<T>& value)
        {
            cl_int res;
            size_t size;

            res = clGetDeviceInfo(id, param, 0, 0, &size);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            if (0 == size)
                return CL_SUCCESS;

            value.resize(size / sizeof(T));

            res = clGetDeviceInfo(id, param, size, &value[0], 0);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            return CL_SUCCESS;
        }

        cl_int query_param(cl_device_id id, cl_device_info param, std::string& value)
        {
            cl_int res;
            size_t size;

            res = clGetDeviceInfo(id, param, 0, 0, &size);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            value.resize(size + 1);

            res = clGetDeviceInfo(id, param, size, &value[0], 0);
            if (CL_SUCCESS != res)
                throw std::runtime_error(std::string("clGetDeviceInfo failed"));

            // just in case, ensure trailing zero for ASCIIZ string
            value[size] = 0;

            return CL_SUCCESS;
        }

    private:
        cl_device_type                            m_type;
        cl_uint                                   m_vendor_id;
        cl_uint                                   m_max_compute_units;
        cl_uint                                   m_max_work_item_dimensions;
        std::vector<size_t>                       m_max_work_item_sizes;
        size_t                                    m_max_work_group_size;
        cl_uint                                   m_preferred_vector_width_char;
        cl_uint                                   m_preferred_vector_width_short;
        cl_uint                                   m_preferred_vector_width_int;
        cl_uint                                   m_preferred_vector_width_long;
        cl_uint                                   m_preferred_vector_width_float;
        cl_uint                                   m_preferred_vector_width_double;
#if defined(CL_VERSION_1_1)
        cl_uint                                   m_preferred_vector_width_half;
        cl_uint                                   m_native_vector_width_char;
        cl_uint                                   m_native_vector_width_short;
        cl_uint                                   m_native_vector_width_int;
        cl_uint                                   m_native_vector_width_long;
        cl_uint                                   m_native_vector_width_float;
        cl_uint                                   m_native_vector_width_double;
        cl_uint                                   m_native_vector_width_half;
#endif
        cl_uint                                   m_max_clock_frequency;
        cl_uint                                   m_address_bits;
        cl_ulong                                  m_max_mem_alloc_size;
        cl_bool                                   m_image_support;
        cl_uint                                   m_max_read_image_args;
        cl_uint                                   m_max_write_image_args;
#if defined(CL_VERSION_2_0)
        cl_uint                                   m_max_read_write_image_args;
#endif
        size_t                                    m_image2d_max_width;
        size_t                                    m_image2d_max_height;
        size_t                                    m_image3d_max_width;
        size_t                                    m_image3d_max_height;
        size_t                                    m_image3d_max_depth;
#if defined(CL_VERSION_1_2)
        size_t                                    m_image_max_buffer_size;
        size_t                                    m_image_max_array_size;
#endif
        cl_uint                                   m_max_samplers;
#if defined(CL_VERSION_1_2)
        cl_uint                                   m_image_pitch_alignment;
        cl_uint                                   m_image_base_address_alignment;
#endif
#if defined(CL_VERSION_2_0)
        cl_uint                                   m_max_pipe_args;
        cl_uint                                   m_pipe_max_active_reservations;
        cl_uint                                   m_pipe_max_packet_size;
#endif
        size_t                                    m_max_parameter_size;
        cl_uint                                   m_mem_base_addr_align;
        cl_device_fp_config                       m_single_fp_config;
#if defined(CL_VERSION_1_2)
        cl_device_fp_config                       m_double_fp_config;
#endif
        cl_device_mem_cache_type                  m_global_mem_cache_type;
        cl_uint                                   m_global_mem_cacheline_size;
        cl_ulong                                  m_global_mem_cache_size;
        cl_ulong                                  m_global_mem_size;
        cl_ulong                                  m_max_constant_buffer_size;
        cl_uint                                   m_max_constant_args;
#if defined(CL_VERSION_2_0)
        size_t                                    m_max_global_variable_size;
        size_t                                    m_global_variable_preferred_total_size;
#endif
        cl_device_local_mem_type                  m_local_mem_type;
        cl_ulong                                  m_local_mem_size;
        cl_bool                                   m_error_correction_support;
#if defined(CL_VERSION_1_1)
        cl_bool                                   m_host_unified_memory;
#endif
        size_t                                    m_profiling_timer_resolution;
        cl_bool                                   m_endian_little;
        cl_bool                                   m_available;
        cl_bool                                   m_compiler_available;
#if defined(CL_VERSION_1_2)
        cl_bool                                   m_linker_available;
#endif
        cl_device_exec_capabilities               m_execution_capabilities;
        cl_command_queue_properties               m_queue_properties;
#if defined(CL_VERSION_2_0)
        cl_command_queue_properties               m_queue_on_host_properties;
        cl_command_queue_properties               m_queue_on_device_properties;
        cl_uint                                   m_queue_on_device_preferred_size;
        cl_uint                                   m_queue_on_device_max_size;
        cl_uint                                   m_max_on_device_queues;
        cl_uint                                   m_max_on_device_events;
#endif
#if defined(CL_VERSION_1_2)
        std::string                               m_built_in_kernels;
#endif
        cl_platform_id                            m_platform;
        std::string                               m_name;
        std::string                               m_vendor;
        std::string                               m_driver_version;
        std::string                               m_profile;
        std::string                               m_version;
#if defined(CL_VERSION_1_1)
        std::string                               m_opencl_c_version;
#endif
        std::string                               m_extensions;
#if defined(CL_VERSION_1_2)
        size_t                                    m_printf_buffer_size;
        cl_bool                                   m_preferred_interop_user_sync;
        cl_device_id                              m_parent_device;
        cl_uint                                   m_partition_max_sub_devices;
        std::vector<cl_device_partition_property> m_partition_properties;
        cl_device_affinity_domain                 m_partition_affinity_domain;
        std::vector<cl_device_partition_property> m_partition_type;
        cl_uint                                   m_reference_count;
#endif
    };


    void initialize_cl_environment(cl_context& out_context, cl_device_id& out_device_id, cl_command_queue& out_command_queue, bool attached_to_opencv = false);
} //end dsm

#endif //DSM_INIT_OPENCL_H