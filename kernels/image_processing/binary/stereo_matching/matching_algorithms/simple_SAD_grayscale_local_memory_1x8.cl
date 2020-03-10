//#include <kernels/image_processing/binary/stereo_matching/cost_functions/SAD_grayscale_local_mem_1x8.cl>




float _compute_SAD_cost_grayscale_1x8_local_mem( 
									    __local unsigned char ref_image_cache[72][72],
										__local unsigned char search_image_cache[72][147],
										__global unsigned char* grayscale_1x8_reference_image,
							  			__global unsigned char* grayscale_1x8_search_image,
							  			int2 reference_pixel_2d_index,
							  			int2 search_pixel_2d_index,
							  			int window_half_width,
							  			int image_width);




float _compute_SAD_cost_grayscale_1x8_local_mem(
									     __local unsigned char ref_image_cache[72][72],
										 __local unsigned char search_image_cache[72][147],
									    __global unsigned char* grayscale_1x8_reference_image,
									    __global unsigned char* grayscale_1x8_search_image,
									    int2 reference_pixel_2d_index, int2 search_pixel_2d_index,
									    int window_half_width,int image_width){

		float current_absolute_diff = 0;
		int counter = 0;

		int disparity_to_test = abs(search_pixel_2d_index.x - reference_pixel_2d_index.x);

		int2 base_offsets = {get_local_id(0) + window_half_width, get_local_id(1) + window_half_width };

		for(int window_index_Y = - window_half_width; window_index_Y <= window_half_width; ++window_index_Y) {
			for(int window_index_X = - window_half_width; window_index_X <= window_half_width; ++window_index_X) {



				int2 ref_image_pos = { base_offsets.x + window_index_X, base_offsets.y + window_index_Y};

				unsigned char ref_value = ref_image_cache[ref_image_pos.y][ref_image_pos.x];
				unsigned char search_value = search_image_cache[ref_image_pos.y][ref_image_pos.x + disparity_to_test];
				current_absolute_diff += abs(ref_value - search_value);
				++counter;
			}
		}

		current_absolute_diff /= counter;
		return current_absolute_diff;		
}






__kernel void compute_disparity_grayscale_local_memory_1x8(__global unsigned char* grayscale_1x8_reference_image, // r32f_in_image
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

	int2 local_thread_index = {get_local_id(0), get_local_id(1)};
	int2 local_sizes = {get_local_size(0), get_local_size(1)};

	__local unsigned char ref_image_cache[72][72];//[5184];
	__local unsigned char search_image_cache[72][147];

	int halo_padding_1d = 2 * window_half_width;

	int effective_shared_memory_block_width_1d = halo_padding_1d + local_sizes.x;
	//__local unsigned char search_image_cache[8];

	// in case we do not have so many physical workers, let each one handle several pixels

	//for()

	uint iteration_count = 0;
	for(reference_pixel_idx.y = thread_2d_index.y + window_half_width; reference_pixel_idx.y < image_height - window_half_width; reference_pixel_idx.y += thread_2d_sizes.y) {
		for(reference_pixel_idx.x = thread_2d_index.x + window_half_width ; reference_pixel_idx.x < image_width - ( window_half_width); reference_pixel_idx.x += thread_2d_sizes.x) {

		//reference_pixel_idx.y = thread_2d_index.y + window_half_width;
		//reference_pixel_idx.x = thread_2d_index.x + window_half_width;

			/*
			for(int local_idx_y = local_thread_index.y; local_idx_y < effective_shared_memory_block_width_1d; local_idx_y += local_sizes.y) {
				for(int local_idx_x = local_thread_index.x; local_idx_x < effective_shared_memory_block_width_1d; local_idx_x += local_sizes.x) {

				    ref_image_cache[local_idx_x + local_idx_y * effective_shared_memory_block_width_1d] 
				    	//= local_idx_x;
				        = grayscale_1x8_reference_image[ ( (local_idx_x - window_half_width) + reference_pixel_idx.x ) +
				   									    ( (local_idx_y - window_half_width) + reference_pixel_idx.y ) * image_width];
				}				
			}*/
			/*
			for(int local_idx_y = local_thread_index.y; local_idx_y < local_sizes.y; local_idx_y += local_sizes.y) {
				for(int local_idx_x = local_thread_index.x; local_idx_x < local_sizes.x; local_idx_x += local_sizes.x) {

				    ref_image_cache[(local_idx_x + window_half_width) + (local_idx_y + window_half_width) * effective_shared_memory_block_width_1d] 
				    	//= local_idx_x;
				        = grayscale_1x8_reference_image[  (reference_pixel_idx.x ) + (reference_pixel_idx.y) * image_width];
				}				
			}*/
			int thread_iteration_x = 0;
			int thread_iteration_y = 0;
			for(int local_idx_y = local_thread_index.y - window_half_width; local_idx_y < local_sizes.y + window_half_width; local_idx_y += local_sizes.y) {
				for(int local_idx_x = local_thread_index.x - window_half_width; local_idx_x < local_sizes.x + window_half_width; local_idx_x += local_sizes.x) {

				    ref_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width]
				    	//= local_idx_x;
				        = grayscale_1x8_reference_image[  (thread_iteration_x * local_sizes.x + reference_pixel_idx.x - window_half_width) 
				        								+ (thread_iteration_y * local_sizes.y + reference_pixel_idx.y - window_half_width) * image_width];
				
				        ++thread_iteration_x;
				}
				++thread_iteration_y;
				thread_iteration_x = 0;
			}


			thread_iteration_x = 0;
			thread_iteration_y = 0;

		
			for(int local_idx_y = local_thread_index.y - window_half_width; local_idx_y < local_sizes.y + window_half_width; local_idx_y += local_sizes.y) {
				for(int local_idx_x = local_thread_index.x - window_half_width; local_idx_x < (local_sizes.x + 75 + window_half_width); local_idx_x += local_sizes.x) {

				    //search_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width]

				    //int potential_x_pos =  ;

				    //potential_x_pos = min(potential_x_pos, potential_x_pos);
				    search_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width]	
				    	//= local_idx_x;
						//if()
				     	=  grayscale_1x8_search_image[  (thread_iteration_x * local_sizes.x + reference_pixel_idx.x - window_half_width)
				        							  + (thread_iteration_y * local_sizes.y + reference_pixel_idx.y - window_half_width) * image_width];
				
				    ++thread_iteration_x;
				}
				++thread_iteration_y;
				thread_iteration_x = 0;
			}

			
			barrier(CLK_LOCAL_MEM_FENCE);



			//if(reference_pixel_idx.x > window_half_width && reference_pixel_idx.x > window_half_width ){

				int reference_pixel_1d_index = reference_pixel_idx.x + reference_pixel_idx.y * image_width;
				


				float best_absolute_diff = FLT_MAX;
				int best_search_pixel_idx = 0;


				int start_search_pixel_idx = max(window_half_width, reference_pixel_idx.x + MIN_DISPARITY - MAX_DISPARITY);
				int end_search_pixel_idx   = min(image_width - window_half_width, reference_pixel_idx.x + MAX_DISPARITY);
				int2 search_pixel_2d_index;
					
							
				for(int search_pixel_index = start_search_pixel_idx; search_pixel_index < end_search_pixel_idx; search_pixel_index++){
    
					search_pixel_2d_index.x  = search_pixel_index;
					search_pixel_2d_index.y = reference_pixel_idx.y;
					float current_absolute_diff =  _compute_SAD_cost_grayscale_1x8_local_mem(
																		    ref_image_cache,
																			search_image_cache,
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

				int out_disparity =  abs(reference_pixel_idx.x - best_search_pixel_idx);
				
				disparity_1x8_image[reference_pixel_1d_index] 
				=  out_disparity;
				//= abs(ref_image_cache[local_thread_index.y + window_half_width][local_thread_index.x + window_half_width]
				//	- search_image_cache[local_thread_index.y + window_half_width][local_thread_index.x + window_half_width] );
				//= abs( grayscale_1x8_search_image[reference_pixel_idx.x + reference_pixel_idx.y * image_width]
				///	 - grayscale_1x8_reference_image[reference_pixel_idx.x + reference_pixel_idx.y * image_width]);

			thread_iteration_x = 0;
			thread_iteration_y = 0;

			/*
			int group_idx_x = get_group_id(0);
			int group_idx_y = get_group_id(1);

			if( group_idx_x == 1 && group_idx_y == 2 ) {
				for(int local_idx_y = local_thread_index.y - window_half_width; local_idx_y < local_sizes.y + window_half_width; local_idx_y += local_sizes.y) {
					for(int local_idx_x = local_thread_index.x - window_half_width; local_idx_x < (local_sizes.x + 75 + window_half_width); local_idx_x += local_sizes.x) {

					    //search_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width]

					    //int potential_x_pos =  ;
						disparity_1x8_image[(thread_iteration_x * local_sizes.x + reference_pixel_idx.x - window_half_width)
					        				+ (thread_iteration_y * local_sizes.y + reference_pixel_idx.y - window_half_width) * image_width]
						= search_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width];
					    //potential_x_pos = min(potential_x_pos, potential_x_pos);
					    //search_image_cache[local_idx_y + window_half_width][local_idx_x + window_half_width]	
					    	//= local_idx_x;
							//if()
					     	//=  grayscale_1x8_search_image[  (thread_iteration_x * local_sizes.x + reference_pixel_idx.x - window_half_width)
					        //							  + (thread_iteration_y * local_sizes.y + reference_pixel_idx.y - window_half_width) * image_width];
					
					    ++thread_iteration_x;
					}
					++thread_iteration_y;
					thread_iteration_x = 0;
				}
			}
			*/


			barrier(CLK_LOCAL_MEM_FENCE);

			++iteration_count;
		}
	}
}