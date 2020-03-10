__constant int num_channels = 3;



inline float _compute_SAD_cost_lab_3x32f(  __global float* lab_3x32f_reference_image,
							  			   __global float* lab_3x32f_search_image,
							  			   int2 reference_pixel_2d_index,
							  			   int2 search_pixel_2d_index,
							  			   int window_half_width,
							  			   int image_width);



inline float _compute_SAD_cost_lab_3x32f(  __global float* lab_3x32f_reference_image,
							  			   __global float* lab_3x32f_search_image,
							  			   int2 reference_pixel_2d_index,
							  			   int2 search_pixel_2d_index,
							  			   int window_half_width,
							  			   int image_width) {

		float current_absolute_diff = 0;
		int counter = 0;
		int window_width = 2 * window_half_width + 1;
		for(int window_index_Y = - window_half_width; window_index_Y <= window_half_width; ++window_index_Y) {

			int current_reference_pixel_row_offset = (reference_pixel_2d_index.y + window_index_Y) * image_width;
			int current_search_pixel_row_offset    = (search_pixel_2d_index.y + window_index_Y) * image_width;

			for(int window_index_X = - window_half_width; window_index_X <= window_half_width; ++window_index_X) {


				int current_1d_reference_pixel_idx = reference_pixel_2d_index.x + window_index_X + current_reference_pixel_row_offset;


				int current_1d_search_pixel_idx = search_pixel_2d_index.x + window_index_X + current_search_pixel_row_offset;				

				int current_1d_reference_pixel_offset = 3 * current_1d_reference_pixel_idx;
				int current_1d_search_pixel_offset = 3 * current_1d_search_pixel_idx;

				float3 ref_lab_value = {lab_3x32f_reference_image[current_1d_reference_pixel_offset    ],
										lab_3x32f_reference_image[current_1d_reference_pixel_offset + 1],
										lab_3x32f_reference_image[current_1d_reference_pixel_offset + 2]
										};
				float3 search_lab_value = {lab_3x32f_search_image[current_1d_search_pixel_offset    ],
										   lab_3x32f_search_image[current_1d_search_pixel_offset + 1],
										   lab_3x32f_search_image[current_1d_search_pixel_offset + 2]
										  };
				
				//float3 diff_lab_value = ref_lab_value - search_lab_value;



				current_absolute_diff += fast_distance(ref_lab_value, search_lab_value);//fast_length(diff_lab_value);
				++counter;
			}
		}

		//current_absolute_diff /= counter;
		return current_absolute_diff;	
}