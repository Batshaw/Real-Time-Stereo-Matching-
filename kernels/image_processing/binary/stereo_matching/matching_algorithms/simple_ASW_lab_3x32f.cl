// DUMMY KERNEL, COPIES THE FUNCTIONALITY OF SAD LAB_3x32 FOR NOW


#include <kernels/image_processing/binary/stereo_matching/cost_functions/ASW_lab_3x32f.cl>

/*** main kernel ***/

// input 1 : buffer lab_32f_reference_image 3x32F containing 3 floats per pixel in lab space
// input 2 : buffer lab_32f_search_image 3x32F containing 3 floats per pixel in lab space
// output  : buffer disparity_1x8_image containing 1 unsigned char disparity information [0..255]
__kernel void compute_disparity_simple_lab_3x32f(__global float* lab_3x32f_reference_image, // r32f_in_image
								  				 __global float* lab_3x32f_search_image,	 // r32f_in_image
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
		int reference_pixel_row_offset = reference_pixel_idx.y * image_width;

		for(reference_pixel_idx.x = thread_2d_index.x; reference_pixel_idx.x < image_width  - window_half_width; reference_pixel_idx.x += thread_2d_sizes.x) {

			if(reference_pixel_idx.x > window_half_width && reference_pixel_idx.y > window_half_width  ) {
			int reference_pixel_1d_index = reference_pixel_idx.x + reference_pixel_row_offset; 
			float best_absolute_diff = FLT_MAX; // max float
			int best_search_pixel_idx = 0;


			int start_search_pixel_idx = max(window_half_width, reference_pixel_idx.x + MIN_DISPARITY - MAX_DISPARITY);
			int end_search_pixel_idx   = min(image_width - window_half_width, reference_pixel_idx.x + MAX_DISPARITY);
			int2 search_pixel_2d_index;
							
			for(int search_pixel_index = start_search_pixel_idx; search_pixel_index < end_search_pixel_idx; search_pixel_index++){

				search_pixel_2d_index.x  = search_pixel_index;
				search_pixel_2d_index.y = reference_pixel_idx.y;
				float current_absolute_diff =  _compute_SAD_cost_lab_3x32f(
																	    	lab_3x32f_reference_image,
						  					  						    	lab_3x32f_search_image,
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
			disparity_1x8_image[reference_pixel_1d_index] = out_disparity;
			
			}		
		}
	}


}
