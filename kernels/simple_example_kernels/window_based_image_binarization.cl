__kernel void binarize_image (__global unsigned char* input_image,
                              __global unsigned char* output_image,
							  int image_width, int image_height,
							  int window_half_size) {

	int2 const pixel_2d_index = {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width; // "flatten" index 2D -> 1D
	int num_channels = 3; // for element offset calculation
	int pixel_1d_offset = num_channels * pixel_1d_index; // actual position of 1D pixel



	//simply convert rgb pixels to grayscale pixels using average of r,g and b values
	uint current_pixel_gray_value = (  input_image[pixel_1d_offset + 0] 
                                     + input_image[pixel_1d_offset + 1]
                                     + input_image[pixel_1d_offset + 2])/3;



	uint num_higher_value_neighbor_pixels = 0;
	uint total_num_neighbor_pixels = 0;

	for(int neigh_y_idx = pixel_2d_index.y - window_half_size; neigh_y_idx <= pixel_2d_index.y + window_half_size; ++neigh_y_idx) {
		for(int neigh_x_idx = pixel_2d_index.x - window_half_size; neigh_x_idx <= pixel_2d_index.x + window_half_size; ++neigh_x_idx) {
			if(neigh_x_idx < 0 || neigh_y_idx < 0 || 
			   neigh_x_idx >= image_width || neigh_y_idx >= image_height) {
				/* skip this iteration if this pixel is not inside of the image */	
			   continue; 
			}

			if(pixel_2d_index.x == neigh_x_idx &&
			   pixel_2d_index.y == neigh_y_idx) {
				/* we skip this iteration, if we are at the center pixel 
			     (because this is not a neighbor but the pixel with the value current_pixel_gray_value)
				*/
			   continue; 
			}

			int neighbor_pixel_1d_index = neigh_x_idx + neigh_y_idx * image_width; // "flatten" neighbor index 2D -> 1D
			
			int neighbor_pixel_1d_offset = num_channels * neighbor_pixel_1d_index; // actual position of 1D neighbor pixel

			uint current_neighbor_pixel_value = (  input_image[neighbor_pixel_1d_offset + 0] 
		                                         + input_image[neighbor_pixel_1d_offset + 1]
		                                         + input_image[neighbor_pixel_1d_offset + 2])/3;

		    if(current_neighbor_pixel_value > current_pixel_gray_value) {
		    	++num_higher_value_neighbor_pixels;
		    }
		    ++total_num_neighbor_pixels;
		}
	}



	if( 2 * num_higher_value_neighbor_pixels > total_num_neighbor_pixels ) {

		for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
			output_image[pixel_1d_offset + channel_idx] = 0;
		}
	} else {
		for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
			output_image[pixel_1d_offset + channel_idx] = 255;
		}	
	}
	

}