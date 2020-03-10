
__constant int num_channels = 3;

/*** forward declarations of device functions ***/
void _convert_bgr_3x8_to_binary_3x8(int2 pixel_2d_index,  __global unsigned char* input_image, __global unsigned char* output_image, int image_width, int image_height);

/*** main kernels ***/

// input  : buffer bgr_3x8_input_image 3x8 bit unsigned char in bgr space
// output : buffer grayscale_1x8_output_image 1x8 bit unsigned char containing luminance greyscale
__kernel void convert_image_bgr_3x8_to_binary_3x8(__global unsigned char* bgr_3x8_input_image,
							  					  __global unsigned char* binary_3x8_output_image, 
							  			          int image_width, int image_height) {

	int2 const thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 const thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			_convert_bgr_3x8_to_binary_3x8(pixel_2d_index, bgr_3x8_input_image, binary_3x8_output_image, 
										   image_width, image_height);
		}
	}

}


__constant float3 average_grayscale_conversion_weights = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};

/*** definition of device functions ***/

__constant int window_half_size = 60;
void _convert_bgr_3x8_to_binary_3x8(int2 pixel_2d_index, 
									__global unsigned char* bgr_3x8_input_image,
									__global unsigned char* binary_3x8_output_image,
									int image_width, int image_height) {

	int const pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
	int const pixel_1d_offset = 3 * pixel_1d_index;

	// swap red and blue channels @ index 2 and 0 in order to convert BGR->RgB
	float3 pixel_color = {bgr_3x8_input_image[pixel_1d_offset + 2],
		   			      bgr_3x8_input_image[pixel_1d_offset + 1],
					      bgr_3x8_input_image[pixel_1d_offset + 0]};

	// use dot product for average = 0.3333333*c0 + 0.3333333*c1 * 0.3333333*c2, because it is built-in
	int average_grayscale_current_pixel = dot(average_grayscale_conversion_weights, pixel_color);

	int num_neighbors_values_above_center = 0;
	int num_neighbors_considered = 0;
	for(int neigh_y_index = pixel_2d_index.y - window_half_size; neigh_y_index <= pixel_2d_index.y + window_half_size; ++neigh_y_index) {
		for(int neigh_x_index = pixel_2d_index.x - window_half_size; neigh_x_index <= pixel_2d_index.x + window_half_size; ++neigh_x_index) {
			if(neigh_x_index < 0 || neigh_y_index < 0 || neigh_x_index >= image_width || neigh_y_index >= image_height) {
				continue; //we do not consider pixels out of the image region for this example
			}
			if(pixel_2d_index.x == neigh_x_index && pixel_2d_index.y == neigh_y_index) {
				continue; //do not consider the center pixel itself, because we compare against him
			}

			int const neighbor_pixel_1d_index = neigh_x_index + neigh_y_index * image_width;
		    int const neighbor_pixel_1d_offset = 3 * neighbor_pixel_1d_index;

			float3 neighbor_pixel_color = {bgr_3x8_input_image[neighbor_pixel_1d_offset + 2],
				   			      		   bgr_3x8_input_image[neighbor_pixel_1d_offset + 1],
							      		   bgr_3x8_input_image[neighbor_pixel_1d_offset + 0]};

			// use dot product for average = 0.3333333*c0 + 0.3333333*c1 * 0.3333333*c2, because it is built-in
			int average_grayscale_neighbor_pixel = dot(average_grayscale_conversion_weights, neighbor_pixel_color);

			if(average_grayscale_neighbor_pixel > average_grayscale_current_pixel) {
				++num_neighbors_values_above_center;
			}
			++num_neighbors_considered;
		}
	}

	if(2*num_neighbors_values_above_center > num_neighbors_considered) {
		for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
			binary_3x8_output_image[pixel_1d_offset + channel_idx] = 0;
		}
	} else {
		for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
			binary_3x8_output_image[pixel_1d_offset + channel_idx] = 255;
		}		
	}

}
