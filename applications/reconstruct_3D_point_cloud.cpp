
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/highgui.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>  //stbi_load

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

struct xyz_point
{
  float x = FLT_MAX;
  float y = FLT_MAX;
  float z = FLT_MAX;
};

struct xyz_rgb_point
{
  float x = FLT_MAX;
  float y = FLT_MAX;
  float z = FLT_MAX;
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;
};

void write_xyz_file(std::vector<xyz_point> const& xyz_points_vec, std::string const& output_filename){
  std::ofstream out_xyz_file;
  out_xyz_file.open(output_filename, std::ofstream::out);
  for(auto const& p : xyz_points_vec){
    if(p.x != FLT_MAX) {
      out_xyz_file << p.x <<" " << p.y << " " << p.z <<"\n";
    }
  }

  out_xyz_file.close();
}

void write_ply_file(std::vector<xyz_rgb_point> const& xyz_rgb_points_vec, std::string const& output_filename){
  std::ofstream out_ply_file;
  out_ply_file.open(output_filename, std::ofstream::out);

  int num_valid_points = 0;
  for(auto const& p : xyz_rgb_points_vec){
    if(p.x != FLT_MAX) {
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
    if(p.x != FLT_MAX) {
      out_ply_file << p.x <<" " << p.y << " " << p.z << " " << int(p.r) << " " << int(p.g) << " " << int(p.b) <<"\n";
    }
  }

  out_ply_file.close();
}

int main(int argc, char** argv) {

  if(argc < 7){
    std::cout << "USAGE: " << argv[0] << " <disparity_map_filename> <color_map_filename> <output_filename> <FOV_in_degree> <cameras_baseline_in_meters> <disparity_scaling> \n";
    return 0;
  }

  bool write_out_color = (argc == 7);

  std::string const disparity_map_filename = argv[1];

  std::string color_map_filename = "";
  if(write_out_color) {
    color_map_filename = argv[2];
  }

  std::cout << "COLOR MAP FILENAME: " << color_map_filename << "\n";

  int argument_offset = write_out_color ? 1 : 0;

  //load image
  int width, height, actual_num_channels;

  int forced_num_channels = 4;

  // 1. load disparity_map
  //d0d0d0 1   d1d1d1 1   d2d2d2 1   ...
  unsigned char *disparity_map = stbi_load(disparity_map_filename.c_str(), &width, &height, &actual_num_channels, forced_num_channels);

  //r0g0b0 1   r1g1b1 1   r2g2b2 1   ...
  unsigned char *color_map = nullptr;

  // 2. load color map
  //int width_col, height_col, num_channels_col;
  if(write_out_color) {
    color_map = stbi_load(color_map_filename.c_str(), &width, &height, &actual_num_channels, forced_num_channels);
  }

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
  int const max_num_3D_points = width * height;
  std::vector<xyz_point> xyz_points_vec(max_num_3D_points);
  std::vector<xyz_rgb_point> xyz_rgb_points_vec(max_num_3D_points);
  cv::Vec3f const camera_position = {0.0f, 0.0f, 0.0f};


  // 5. copmute 3Ddepth (Z) and new 3D position for each pixels
  for(int row_indx = 0; row_indx < height; ++row_indx){
    for(int col_indx = 0; col_indx < width; ++col_indx){ //Interprets a floating point value in a byte string pointed to by str.
      int pixel_1D_index = row_indx * width + col_indx;   // 2D -> 1D
      int pixel_1D_position_with_channel_offset = pixel_1D_index * forced_num_channels;
      unsigned char current_pixel_diparity = disparity_map[pixel_1D_position_with_channel_offset];
      
      // ignore the pixels with disparity=0 because could not coumpute Z
      if(0 == current_pixel_diparity) {
        continue;
      }

      //3D depth
      float z_coord = (focal_length_in_pixels * baseline_in_meters) / (current_pixel_diparity * disparity_scaling);

      // compute the ray back-projected from pixel to the camera (like ray-tracing)
      cv::Vec3f const current_image_plane_point = {col_indx - width*0.5f, row_indx - height * 0.5f, focal_length_in_pixels};
      cv::Vec3f const current_ray_direction = current_image_plane_point - camera_position;
      cv::Vec3f const normalized_ray_direction = cv::normalize(current_ray_direction);

      // camera_position + t * normalized_ray_direction = xyz_posion
      float t = z_coord / normalized_ray_direction[2]; //cam_posision is assumed to be in the origine 
      cv::Vec3f const new_3D_position = t * normalized_ray_direction;
      
      if(!write_out_color) {
        xyz_point new_xyz_point {new_3D_position[0], new_3D_position[1], new_3D_position[2]};
        xyz_points_vec[pixel_1D_index] = new_xyz_point;
      } else {
        unsigned char current_r = color_map[pixel_1D_position_with_channel_offset + 0];
        unsigned char current_g = color_map[pixel_1D_position_with_channel_offset + 1];
        unsigned char current_b = color_map[pixel_1D_position_with_channel_offset + 2];
        xyz_rgb_point new_xyz_rgb_point {new_3D_position[0], new_3D_position[1], new_3D_position[2],
                                        current_r, current_g, current_b};
        xyz_rgb_points_vec[pixel_1D_index] = new_xyz_rgb_point;
      }
    }
  }

  std::string const output_filename = argv[2+argument_offset];
  if(!write_out_color) {
    write_xyz_file(xyz_points_vec, output_filename);
  } else {
    write_ply_file(xyz_rgb_points_vec, output_filename);
  }
  return 0;
}
