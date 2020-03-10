/*** forward declarations of device functions ***/
void _convert_pixel_bgr_3x8_to_lab_3x32f(int2 pixel_2d_index,  __global unsigned char* input_image, __global float* output_image, int image_width, int image_height);
void _convert_rgb_to_lab(float* in_rgb_color, float* out_lab_color);

/*** main kernels ***/
__kernel void convert_image_bgr_3x8_to_lab_3x32f(__global unsigned char* input_image,
							  					 __global float* output_image,
							  					 int image_width, int image_height) {
	

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			_convert_pixel_bgr_3x8_to_lab_3x32f(pixel_2d_index, input_image, output_image, image_width, image_height);
		}
	}

	
}


/*** definition of device functions ***/
void _convert_pixel_bgr_3x8_to_lab_3x32f(int2 pixel_2d_index, 
								 	     __global unsigned char* input_image,
								 	     __global float* output_image,
								 	     int image_width, int image_height) {

	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
	int pixel_1d_offset = 3 * pixel_1d_index;

	float rgb_pixel_color[3] = {input_image[pixel_1d_offset + 2],
		   			            input_image[pixel_1d_offset + 1],
					            input_image[pixel_1d_offset + 0]};

	float lab_pixel_color[3] = {0.0f, 0.0f, 0.0f};

	_convert_rgb_to_lab(rgb_pixel_color, lab_pixel_color);

	//explicit channel assignment
	output_image[pixel_1d_offset    ] = lab_pixel_color[0];
	output_image[pixel_1d_offset + 1] = lab_pixel_color[1];
	output_image[pixel_1d_offset + 2] = lab_pixel_color[2];
	
	
}

void _convert_rgb_to_lab(float* in_rgb_color, float* out_lab_color) {
  float3 div_255 = {255.0f, 255.0f, 255.0f};

  float3 normalized_rgb_color = {in_rgb_color[0],
  							 	 in_rgb_color[1],
  							 	 in_rgb_color[2],
  							 	 };

  normalized_rgb_color /= div_255;


  float3 add_0p55 = {0.055f, 0.055f, 0.055f};
  float3 div_1p055 = {1.055f, 1.055f, 1.055f};
  float3 pow_2p4 = {2.4f, 2.4f, 2.4f};

  float3 powered_normalized_rgb_color = pow(((normalized_rgb_color + add_0p55) / div_1p055), pow_2p4);

  float3 div_12p92 = {12.92f, 12.92f, 12.92f};
  float3 normalized_rgb_by_12p92 = normalized_rgb_color / div_12p92;


  float3 tmp_rgb;
  tmp_rgb.x = (normalized_rgb_color.x > 0.04045f) ? powered_normalized_rgb_color.x : normalized_rgb_by_12p92.x;
  tmp_rgb.y = (normalized_rgb_color.y > 0.04045f) ? powered_normalized_rgb_color.y : normalized_rgb_by_12p92.y;
  tmp_rgb.z = (normalized_rgb_color.z > 0.04045f) ? powered_normalized_rgb_color.z : normalized_rgb_by_12p92.z;

  float3 dot_tmp_x = {0.4124f, 0.3576f, 0.1805f};
  float3 dot_tmp_y = {0.2126f, 0.7152f, 0.0722f};
  float3 dot_tmp_z = {0.1805f, 0.0193f, 0.9505f};

  float3 div_tmp = {0.95047f, 1.00000f, 1.08883f};

  float3 xyz_tmp;  
  xyz_tmp.x = dot(dot_tmp_x, tmp_rgb);
  xyz_tmp.y = dot(dot_tmp_y, tmp_rgb);
  xyz_tmp.z = dot(dot_tmp_z, tmp_rgb);

  xyz_tmp /= div_tmp;

  float3 xyz;
  
  float3 one_third_power = {0.33333333f, 0.33333333f, 0.33333333f}; 

  float3 powered_xyz_tmp = pow(xyz_tmp, one_third_power);

  float3 mul_7p787 = {7.787f, 7.787f, 7.787f};
  float3 add_16_by_116 = {0.13793103f, 0.13793103f, 0.13793103f};

  float3 rhs_result = mul_7p787 * xyz_tmp + add_16_by_116;

  xyz.x = (xyz_tmp.x > 0.008856f) ? powered_xyz_tmp.x : rhs_result.x;
  xyz.y = (xyz_tmp.y > 0.008856f) ? powered_xyz_tmp.y : rhs_result.y;
  xyz.z = (xyz_tmp.z > 0.008856f) ? powered_xyz_tmp.z : rhs_result.z;

  out_lab_color[0] = xyz.x;
  out_lab_color[1] = xyz.y;
  out_lab_color[2] = xyz.z; 


// OLD CODE WITHOUT USING VECTOR OPERATIONS
/*
  float3 tmp_rgb;
  tmp_rgb.x = (normalized_rgb_color.x > 0.04045) ? pow((normalized_rgb_color.x + 0.055) / 1.055, 2.4) : normalized_rgb_color.x / 12.92;
  tmp_rgb.y = (normalized_rgb_color.y > 0.04045) ? pow((normalized_rgb_color.y + 0.055) / 1.055, 2.4) : normalized_rgb_color.y / 12.92;
  tmp_rgb.z = (normalized_rgb_color.z > 0.04045) ? pow((normalized_rgb_color.z + 0.055) / 1.055, 2.4) : normalized_rgb_color.z / 12.92;

  float3 xyz_tmp;
  xyz_tmp.x = (tmp_rgb.x * 0.4124 + tmp_rgb.y * 0.3576 + tmp_rgb.z * 0.1805) / 0.95047;
  xyz_tmp.y = (tmp_rgb.x * 0.2126 + tmp_rgb.y * 0.7152 + tmp_rgb.z * 0.0722) / 1.00000;
  xyz_tmp.z = (tmp_rgb.x * 0.0193 + tmp_rgb.y * 0.1192 + tmp_rgb.z * 0.9505) / 1.08883;

  float3 xyz;
  float one_third = 1.0/3.0;
  xyz.x = (xyz_tmp.x > 0.008856) ? pow(xyz_tmp.x, one_third) : (7.787 * xyz_tmp.x) + 16.0/116.0;
  xyz.y = (xyz_tmp.y > 0.008856) ? pow(xyz_tmp.y, one_third) : (7.787 * xyz_tmp.y) + 16.0/116.0;
  xyz.z = (xyz_tmp.z > 0.008856) ? pow(xyz_tmp.z, one_third) : (7.787 * xyz_tmp.z) + 16.0/116.0;

  out_lab_color[0] = xyz.x;
  out_lab_color[1] = xyz.y;
  out_lab_color[2] = xyz.z; 
*/
}