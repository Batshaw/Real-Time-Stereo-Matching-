#include <core/utils.h>

#include <iostream>
#include <set>
#include <vector>

namespace dsm {

//loads kernel from source file filename and stores content in string s.
int load_kernel_from_file(const char* filename, std::string& s) {
    size_t size = 0;
    char*  str = 0;
    std::ifstream in_file(filename, (std::ios::in | std::ios::binary));

    if (in_file.is_open())
    {
        size_t fileSize = 0;
        in_file.seekg(0, std::fstream::end);
        size = fileSize = (size_t)in_file.tellg();
        in_file.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            in_file.close();
            return 0;
        }

        in_file.read(str, fileSize);
        in_file.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error: failed to open file: " << filename << std::endl;
    return 1;
}


int check_program_build_status(cl_program const& program, cl_device_id const& device_id) {
    char *log = nullptr;
    size_t size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);

    log = (char *)malloc(size+1);
    if (log) {
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
        size, log, NULL);

        log[size] = '\0';
        printf("%s", log);
        free(log);
        return 1;
    }

    return 0;
}

int compile_kernel_from_file(cl_context const& context, cl_device_id const& device_id,
							 std::string const& file_path, std::string const& kernel_function,
							 cl_program& out_program, cl_kernel& out_kernel, std::string const& kernel_defines) {

    std::string kernel_source_code = "";
    cl_int status = dsm::load_kernel_from_file(file_path.c_str(), kernel_source_code);
    if(status != CL_SUCCESS){
        std::cout << "Failed to parse cl source file: " <<  file_path.c_str() << std::endl;
    }

    kernel_source_code = kernel_defines + "\n" + kernel_source_code;

    size_t source_size = kernel_source_code.size();

    char const* c_style_source = kernel_source_code.c_str();

    //current_program;// = cl_programs_per_mode_[ColorConversionMode::BGR_3x8_TO_GRAYSCALE_3x8];

	out_program = clCreateProgramWithSource(context, 1, &c_style_source, &source_size, NULL);

    status = clBuildProgram(out_program, 1, &device_id, NULL, NULL, NULL);
    
    if(CL_SUCCESS != status) {
        check_program_build_status(out_program, device_id);
    }

    out_kernel = clCreateKernel(out_program, kernel_function.c_str(), &status);
    dsm::check_cl_error(status, "clCreateKernel");

	return 0;
}

// Function to check and handle OpenCL errors
void check_cl_error(cl_int err, const std::string &name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << get_cl_error_string(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::string get_cl_error_string(cl_int cl_error_code) {
    switch(cl_error_code) {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}



std::string get_filename_from_path(std::string const& path) {
  std::vector<std::string> result;

  std::set<char> delimiters{'\\', '/'};

  char const* pch = path.c_str();
  char const* start = pch;
  for(; *pch; ++pch)
  {
    if (delimiters.find(*pch) != delimiters.end())
    {
      if (start != pch)
      {
        std::string str(start, pch);
        result.push_back(str);
      }
      else
      {
        result.push_back("");
      }
      start = pch + 1;
    }
  }
  result.push_back(start);

  return result.back();
}


void write_ply_file(std::vector<xyz_rgb_point> const& xyz_rgb_points_vec, std::string const& output_filename)
{
  std::ofstream out_ply_file;
  out_ply_file.open(output_filename, std::ofstream::out);

  int num_valid_points = 0;

  for(auto const& p : xyz_rgb_points_vec){
    if( (p.x != 0.0f) || (p.y != 0.0f) || (p.z != 0.0f) ) {
      ++num_valid_points;
    }
  }

  std::string header_part_1 = "";
  header_part_1 += "ply\n";
  header_part_1 += "format ascii 1.0\n";

  std::string header_part_2 = "element vertex " + std::to_string(num_valid_points) + "\n";
  header_part_2 += "property float x\n";
  header_part_2 += "property float y\n";
  header_part_2 += "property float z\n";
  header_part_2 += "property uchar red\n";
  header_part_2 += "property uchar green\n";
  header_part_2 += "property uchar blue\n";
  header_part_2 += "end_header\n";

  //write file headers
  out_ply_file << header_part_1 << header_part_2;

  for(auto const& p : xyz_rgb_points_vec){
    if(p.x != 0.0f || p.y != 0.0f || p.z != 0.0f) {
      out_ply_file << p.x <<" " << p.y << " " << p.z << " " << int(p.r) << " " << int(p.g) << " " << int(p.b) <<"\n";
    }
  }

  out_ply_file.close();
}



} //namespace dsm
