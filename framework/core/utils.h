#ifndef DSM_UTILS_H
#define DSM_UTILS_H

#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>



#define ENABLE_BLUE_CONSOLE_COLOR "\033[1;44m"
#define ENABLE_RED_CONSOLE_COLOR "\033[1;31m"
#define ENABLE_YELLOW_CONSOLE_COLOR "\033[1;33m"
#define DISABLE_CONSOLE_COLOR "\033[0m\n"

#define DSM_LOG_INFO(message) std::cout << "\n" << ENABLE_BLUE_CONSOLE_COLOR << "[INFO] Line "<< __LINE__ <<" in file " << \
											 __FILE__ << ": "<<  __PRETTY_FUNCTION__ << "\n\n" \
											 << "message: " << message << "\n" << \
											 DISABLE_CONSOLE_COLOR

#define DSM_LOG_WARNING(message) std::cout << "\n" <<  ENABLE_YELLOW_CONSOLE_COLOR << "[WARNING] Line "<< __LINE__ <<" in file " << \
											 __FILE__ << ": "<<  __PRETTY_FUNCTION__ << "\n\n" \
											 << "message: " << message << "\n" << \
											 DISABLE_CONSOLE_COLOR

#define DSM_LOG_ERROR(message) std::cout << "\n" << ENABLE_RED_CONSOLE_COLOR << "[ERROR] Line "<< __LINE__ <<" in file " << \
											 __FILE__ << ": "<<  __PRETTY_FUNCTION__ << "\n\n" \
											 << "message: " << message << "\n" << \
											 DISABLE_CONSOLE_COLOR


struct xyz_rgb_point
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;
};

namespace dsm {
  //loads kernel from source file filename and stores content in string s.
  int load_kernel_from_file(const char* file_path, std::string& s);
  int check_program_build_status(cl_program const& program, cl_device_id const& device_id);

  int compile_kernel_from_file(cl_context const& context, cl_device_id const& device_id,
  							   std::string const& file_path, std::string const& kernel_function,
  							   cl_program& out_program, cl_kernel& out_kernel, std::string const& kernel_defines="");

  void check_cl_error(cl_int err, const std::string &name);

  std::string get_cl_error_string(cl_int error);

  std::string get_filename_from_path(std::string const& path);


  void write_ply_file(std::vector<xyz_rgb_point> const& xyz_rgb_points_vec, std::string const& output_filename);


}

#endif //DSM_UTILS_H
