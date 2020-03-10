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


int main(int argc, char** argv) {


  std::cout << "XXXXXXXXXXXXXXXXXX\n";

  if(argc < 7){
    std::cout << "USAGE: " << argv[0] << " <disparity_map_filename> <color_map_filename> <output_filename> <FOV_in_degree> <cameras_baseline_in_meters> <disparity_scaling> \n";
    return 1;
  }

  std::string const disparity_map_filename = argv[1];

  std::string color_map_filename = "";

  color_map_filename = argv[2];
  

  std::cout << "COLOR MAP FILENAME: " << color_map_filename << "\n";

  int argument_offset = 1;

  //load image


  int forced_num_channels = 4;
  int bgr_xyz_channels = 3;

  // 1. load disparity_map
  //d0d0d0 1   d1d1d1 1   d2d2d2 1   ...

  //rd0 1    1   r2g2b2 1   ...
  unsigned char *color_map = nullptr;
  cv::Mat color_image;
  cv::Mat disparity_image;

  disparity_image = cv::imread(disparity_map_filename.c_str(), cv::IMREAD_GRAYSCALE);

  // 2. load color map

  //color_map = stbi_load(color_map_filename.c_str(), &width, &height, &actual_num_channels, forced_num_channels);
  

  color_image = cv::imread(color_map_filename.c_str(), cv::IMREAD_COLOR);
  int32_t width = color_image.cols;
  int32_t height = color_image.rows;

  // 3. recieve input from USAGE
  float const field_of_view_angle_deg = atof(argv[3+argument_offset]);
  float const pi =  3.14159265359f;
  // compute the focal length (f) in pixels
  float const half_field_of_view_angle_rad = 0.5 *((field_of_view_angle_deg * pi) / 180.0);
  float const half_width = width / 2;
  float const focal_length_in_pixels = half_width / std::tan(half_field_of_view_angle_rad);
  float const baseline_in_meters = atof(argv[4+argument_offset]);
  float const disparity_scaling = atof(argv[5+argument_offset]);

  // 4. Initial vectors for 2 output files (xyz point and xyz point with rgb)
  int const max_num_3D_points = width * height * 3;
  std::vector<float> points_vec(max_num_3D_points * 3 ); 
  std::vector<unsigned char> color_vec(max_num_3D_points * 3 );
  cv::Vec3f const camera_position = {0.0f, 0.0f, 0.0f};

  cl_context       context = 0;
  cl_device_id     device_id = 0;
  cl_command_queue command_queue = 0;
  cl_program       program = 0; 
  cl_kernel        kernel = 0;
  
  cl_mem           cl_disparity_input_buffer = 0;
  cl_mem           cl_output_buffer = 0;
  cl_mem           cl_color_input_buffer = 0;
  cl_mem           cl_color_output_buffer = 0;


  dsm::initialize_cl_environment(context, device_id, command_queue);  

  cl_disparity_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                width * height * sizeof(char), (void *)disparity_image.data, NULL); // 1 grayscale channel 
  cl_color_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            width * height  * bgr_xyz_channels * sizeof(char), (void *)color_image.data, NULL); // 3 bgr channels

  cl_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                      width * height * bgr_xyz_channels* sizeof(float), NULL, NULL);
  cl_color_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                      width * height * bgr_xyz_channels* sizeof(char), NULL, NULL);
  


  std::string const kernel_file_path = "./kernels/image_processing/binary/3d_reconstruction/reconstruct_3D_points_GPU.cl";
  std::string const& main_kernel_name = "reconstruct_3D_points_GPU";
  std::string source_string{""};
  dsm::load_kernel_from_file(kernel_file_path.c_str(), source_string);
  char const* source = source_string.c_str();
  size_t source_buffer_size = source_string.size();
  program = clCreateProgramWithSource(context, 1, &source, &source_buffer_size, NULL);
  cl_int status_code = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  if (CL_SUCCESS != status_code) {
    char *log;
    size_t size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    log = (char *)malloc(size+1);
    if (log) {

      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
      size, log, NULL);
      log[size] = '\0';
      printf("%s", log);
      free(log);
    }
        exit(-1);
  }
 
  kernel = clCreateKernel(program, main_kernel_name.c_str(), NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_disparity_input_buffer); 
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_output_buffer);  
  clSetKernelArg(kernel, 2, sizeof(int), (void*) &width); 
  clSetKernelArg(kernel, 3, sizeof(int), (void*) &height);
  clSetKernelArg(kernel, 4, sizeof(float), (void*) &baseline_in_meters);
  clSetKernelArg(kernel, 5, sizeof(float), (void*) &focal_length_in_pixels);
  clSetKernelArg(kernel, 6, sizeof(float), (void*) &disparity_scaling);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&cl_color_input_buffer); 
  clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&cl_color_output_buffer); 

  size_t global_work_size[2] = {size_t(width), size_t(height)};

  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                           global_work_size, NULL, 0, NULL, NULL);


  std::size_t num_byte_to_read_result = width * height * 3 * sizeof(float);
  std::size_t num_byte_to_read_result_color = width * height * 3 * sizeof(char);

  status_code = clEnqueueReadBuffer(command_queue, cl_output_buffer, CL_TRUE, 0, 
                                    num_byte_to_read_result, points_vec.data(), 0, NULL, NULL);

  status_code = clEnqueueReadBuffer(command_queue, cl_color_output_buffer, CL_TRUE, 0, 
                                    num_byte_to_read_result_color, color_vec.data(), 0, NULL, NULL);

  std::vector<xyz_rgb_point> xyz_rgb_points_for_ply_writer(width*height * 3);

  for(int point_index = 0; point_index < points_vec.size() / 3; ++point_index) { 
      size_t point_base_offset = 3 * point_index; 
        
            /*std::cout << points_vec[point_base_offset + 0] << " " 
                     << points_vec[point_base_offset + 1] << " "
                     << points_vec[point_base_offset + 2] << std::endl;
            */
      xyz_rgb_point& p = xyz_rgb_points_for_ply_writer[point_index];

      p.x = points_vec[point_base_offset + 0];
      p.y = points_vec[point_base_offset + 1];
      p.z = points_vec[point_base_offset + 2];

      p.r = color_vec[point_base_offset + 0];
      p.g = color_vec[point_base_offset + 1];
      p.b = color_vec[point_base_offset + 2];

  }


    std::string const output_filename = argv[2+argument_offset];
  
    dsm::write_ply_file(xyz_rgb_points_for_ply_writer, output_filename);
    


  if(CL_SUCCESS != status_code) {
    std::cout << "Encountered a cl-Error in the execution of submitted commandQueue operations." << "\n";
    std::cout << "Error Code: " << status_code << "\n";
  }

    return 0;
}
