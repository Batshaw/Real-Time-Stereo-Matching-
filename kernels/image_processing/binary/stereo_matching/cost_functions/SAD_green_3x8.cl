#ifndef COST_SAD_GREEN_3X8_CL
#define COST_SAD_GREEN_3X8_CL


float _compute_SAD_cost_green_3x8(
							      __global unsigned char* bgr_3x8_reference_image,
							      __global unsigned char* bgr_3x8_search_image,
							      int2 reference_pixel_2d_index,
							      int2 search_pixel_2d_index,
							      int window_half_width,
							      int image_width){

		float current_absolute_diff = 0;
		int counter = 0;
		int window_width = 2 * window_half_width + 1;
		for(int window_index_Y = - window_half_width; window_index_Y <= window_half_width; ++window_index_Y) {
			for(int window_index_X = - window_half_width; window_index_X <= window_half_width; ++window_index_X) {

				int window_pixel_1d_offset = (window_index_X + window_half_width) 
									   	   + (window_index_Y + window_half_width) * window_width;

				//float ref_value = ref_pixel_window_values[window_pixel_1d_offset];

				int2 current_search_pixel_index = {search_pixel_2d_index.x + window_index_X, search_pixel_2d_index.y + window_index_Y};  
				int current_1d_search_pixel_idx = (current_search_pixel_index.x + current_search_pixel_index.y * image_width);
				
				int2 current_reference_pixel_index = {reference_pixel_2d_index.x + window_index_X, reference_pixel_2d_index.y + window_index_Y};  
				int current_1d_reference_pixel_idx = (current_reference_pixel_index.x + current_reference_pixel_index.y * image_width);
				unsigned char ref_value = bgr_3x8_reference_image[ 3 * current_1d_reference_pixel_idx + 1]; //green channel
				unsigned char search_value = bgr_3x8_search_image[ 3 * current_1d_search_pixel_idx + 1]; // green channel
				current_absolute_diff += abs(ref_value - search_value);
				++counter;
			}
		}

		current_absolute_diff /= counter;
		return current_absolute_diff;		
}

#endif