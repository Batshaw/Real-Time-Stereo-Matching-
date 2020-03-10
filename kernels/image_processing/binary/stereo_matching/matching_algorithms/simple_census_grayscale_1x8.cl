__constant int num_channels = 3;
//__constant int MAX_DISPARITY = 75;


float _compute_census_cost_grayscale_1x8(__global unsigned char* grayscale_1x8_reference_image,
									     __global unsigned char* grayscale_1x8_search_image,
									     int2 reference_pixel_2d_index,int2 search_pixel_2d_index,
									     int window_half_width,int image_width){


		float current_absolute_diff = 0;
		int counter = 0;
		int window_width = 2 * window_half_width + 1;

		int2 center_reference_pixel_index = {reference_pixel_2d_index.x, reference_pixel_2d_index.y};  
		int center_1d_reference_pixel_idx = (center_reference_pixel_index.x + center_reference_pixel_index.y * image_width);
		unsigned char center_ref_value = grayscale_1x8_reference_image[center_1d_reference_pixel_idx]; //no channel offsets, tightly packed

		int2 center_search_pixel_index = {search_pixel_2d_index.x, search_pixel_2d_index.y};  
		int center_1d_search_pixel_idx = (center_search_pixel_index.x + center_search_pixel_index.y * image_width);

		unsigned char center_search_value = grayscale_1x8_search_image[center_1d_search_pixel_idx]; // no channel offsets, tightly packed

		for(int window_index_Y = - window_half_width; window_index_Y <= window_half_width; ++window_index_Y) {

			int current_reference_pixel_row_offset = (reference_pixel_2d_index.y + window_index_Y) * image_width;
			int current_search_pixel_row_offset    = (search_pixel_2d_index.y + window_index_Y) * image_width;

			for(int window_index_X = - window_half_width; window_index_X <= window_half_width; ++window_index_X) {

				int window_pixel_1d_offset = (window_index_X + window_half_width) 
									   	   + (window_index_Y + window_half_width) * window_width;

				//float ref_value = ref_pixel_window_values[window_pixel_1d_offset];

				int2 current_search_pixel_index = {search_pixel_2d_index.x + window_index_X, search_pixel_2d_index.y + window_index_Y};  
				int current_1d_search_pixel_idx = (current_search_pixel_index.x + current_search_pixel_index.y * image_width);
				
				int2 current_reference_pixel_index = {reference_pixel_2d_index.x + window_index_X, reference_pixel_2d_index.y + window_index_Y};  
				int current_1d_reference_pixel_idx = (current_reference_pixel_index.x + current_reference_pixel_index.y * image_width);
				unsigned char ref_value = grayscale_1x8_reference_image[current_1d_reference_pixel_idx]; //no channel offsets, tightly packed
				unsigned char search_value = grayscale_1x8_search_image[current_1d_search_pixel_idx]; // no channel offsets, tightly packed


	           if(((int)ref_value - (int)center_ref_value) * ((int)search_value - (int)center_search_value) < 0) {
	               current_absolute_diff += 1;
	           }

				//current_absolute_diff += abs(ref_value - search_value);
				++counter;
			}
		}


		//current_absolute_diff /= counter;
		return current_absolute_diff;		
}


/*** main kernel ***/

// input 1 : grayscale_1x8_reference_image 1x8 bit unsigned char tightly packed 1 value per pixel in luminance 
// input 2 : grayscale_1x8_reference_image 1x8 bit unsigned char tightly packed 1 value per pixel in luminance 
// output :  disparity_1x8_image containing 1 unsigned char disparity information [0..255]
__kernel void compute_disparity_simple_grayscale_1x8(__global unsigned char* grayscale_1x8_reference_image, // r32f_in_image
								  					 __global unsigned char* grayscale_1x8_search_image,	 // r32f_in_image
								 					 __global unsigned char* disparity_1x8_image,		 // r32f_in_image
								  					 int image_width,
								  					 int image_height,
								  					 int window_half_width,
								  					 int MIN_DISPARITY,
								  					 int MAX_DISPARITY) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
	int2 reference_pixel_idx;




	// in case we do not have so many physical workers, let each one handle several pixels
	for(reference_pixel_idx.y = thread_2d_index.y; reference_pixel_idx.y < image_height - window_half_width; reference_pixel_idx.y += thread_2d_sizes.y) {
		for(reference_pixel_idx.x = thread_2d_index.x; reference_pixel_idx.x < image_width  - window_half_width; reference_pixel_idx.x += thread_2d_sizes.x) {

			if(reference_pixel_idx.x > window_half_width && reference_pixel_idx.x > window_half_width  ){

				int reference_pixel_1d_index = reference_pixel_idx.x + reference_pixel_idx.y * image_width;
				
				float best_absolute_diff = FLT_MAX;
				int best_search_pixel_idx = 0;


				int start_search_pixel_idx = max(window_half_width, reference_pixel_idx.x + MIN_DISPARITY - MAX_DISPARITY);
				int end_search_pixel_idx   = min(image_width - window_half_width, reference_pixel_idx.x + MAX_DISPARITY);
				int2 search_pixel_2d_index;
								
				for(int search_pixel_index = start_search_pixel_idx; search_pixel_index < end_search_pixel_idx; search_pixel_index++){
    
					search_pixel_2d_index.x  = search_pixel_index;
					search_pixel_2d_index.y = reference_pixel_idx.y;
					float current_absolute_diff =  _compute_census_cost_grayscale_1x8(
																		              grayscale_1x8_reference_image,
							  					  						              grayscale_1x8_search_image,
							  					  						              reference_pixel_idx,
							  					  						              search_pixel_2d_index,
							  					  						              window_half_width,
							  					  						              image_width); 
							  					  						        

					if (best_absolute_diff > current_absolute_diff) {
						best_absolute_diff = current_absolute_diff;
						best_search_pixel_idx = search_pixel_index;
					}
					
				}

				int out_disparity = abs(reference_pixel_idx.x - best_search_pixel_idx);
				
				disparity_1x8_image[reference_pixel_1d_index] = out_disparity;
					
			}
		}

	}
}

