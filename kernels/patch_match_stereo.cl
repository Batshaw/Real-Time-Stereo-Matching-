// forward declarations begin

int get_index_2d_to_1d(int x, int y, int width);

int get_index_2d_to_1d(int x, int y, int width) {
	return x + y * width;
}

__constant int NUM_CHANNELS = 3;

__kernel void convertImageUchar3ToGrayscaleFloatAndNormalize (__global unsigned char* colored_input_image,
							  		  		  				  __global float* grayscale_normalized_output_image,
									  		  				  int image_width, 
									  		  				  int image_height) {


	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	grayscale_normalized_output_image[thread_2d_index.x + thread_2d_index.y * image_width] = 0.8;

	
	int2 pixel_idx;

	// CIE Y based weighted color channel conversion
	//https://core.ac.uk/download/pdf/14697119.pdf
	float3 grayscale_conversion_weights = {0.212671, 0.71516,  0.072169};
	// in case we do not have so many physical workers, let each one handle several pixels
	for(pixel_idx.y = thread_2d_index.y; pixel_idx.y < image_height; pixel_idx.y += thread_2d_sizes.y) {
		for(pixel_idx.x = thread_2d_index.x; pixel_idx.x < image_width; pixel_idx.x += thread_2d_sizes.x) {
			
			//convert 2d pixel idx to 1d pixel index
			int pixel_1d_index = get_index_2d_to_1d(pixel_idx.x, pixel_idx.y, image_width);
			
			int pixel_1d_buffer_offset = NUM_CHANNELS * pixel_1d_index;
			
			//implicit unsigned char to float cast
			float3 pixel_color_rgb = {(colored_input_image[pixel_1d_buffer_offset + 2]),
							   		  (colored_input_image[pixel_1d_buffer_offset + 1]),
							   		  (colored_input_image[pixel_1d_buffer_offset + 0])};


			// use built in dot product for vectorized w_r*r + w_g*g + w_b*b
			float grayscale_value = dot(grayscale_conversion_weights, pixel_color_rgb) / 255.0;

			//no channels anymore! we store only one float per pixel in the gs image
			grayscale_normalized_output_image[pixel_1d_index] = grayscale_value;
		}
	}
}

__kernel void convertImageFloatToUChar3 (__global float* r32f_in_image,
							  	 		 __global unsigned char* rgb8_out_image,
								 		 int image_width,
								 		 int image_height) {


	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_idx;


	// in case we do not have so many physical workers, let each one handle several pixels
	for(pixel_idx.y = thread_2d_index.y; pixel_idx.y < image_height; pixel_idx.y += thread_2d_sizes.y) {
		for(pixel_idx.x = thread_2d_index.x; pixel_idx.x < image_width; pixel_idx.x += thread_2d_sizes.x) {
			
			//convert 2d pixel idx to 1d pixel index
			int pixel_1d_index = get_index_2d_to_1d(pixel_idx.x, pixel_idx.y, image_width);
			

			
			//implicit unsigned char to float cast
			uchar char_converted = 4 * (r32f_in_image[pixel_1d_index]);// * 255.0;

			int pixel_1d_buffer_offset = NUM_CHANNELS * pixel_1d_index;

			//no channels anymore! we store only one float per pixel in the gs image
			rgb8_out_image[pixel_1d_buffer_offset + 0] = char_converted;
			rgb8_out_image[pixel_1d_buffer_offset + 1] = char_converted;
			rgb8_out_image[pixel_1d_buffer_offset + 2] = char_converted;
		}
	}
}


float4 convert_rgb_to_lab(uchar4 rgb_color) {
  float4 normalized_color = {rgb_color.z / 255.0,
  							 rgb_color.y / 255.0,
  							 rgb_color.x / 255.0,
  							 1.0};

  float4 tmp_rgb;
  tmp_rgb.x = (normalized_color.x > 0.04045) ? pow((normalized_color.x + 0.055) / 1.055, 2.4) : normalized_color.x / 12.92;
  tmp_rgb.y = (normalized_color.y > 0.04045) ? pow((normalized_color.y + 0.055) / 1.055, 2.4) : normalized_color.y / 12.92;
  tmp_rgb.z = (normalized_color.z > 0.04045) ? pow((normalized_color.z + 0.055) / 1.055, 2.4) : normalized_color.z / 12.92;

  float4 xyz_tmp;
  xyz_tmp.x = (tmp_rgb.x * 0.4124 + tmp_rgb.y * 0.3576 + tmp_rgb.z * 0.1805) / 0.95047;
  xyz_tmp.y = (tmp_rgb.x * 0.2126 + tmp_rgb.y * 0.7152 + tmp_rgb.z * 0.0722) / 1.00000;
  xyz_tmp.z = (tmp_rgb.x * 0.0193 + tmp_rgb.y * 0.1192 + tmp_rgb.z * 0.9505) / 1.08883;

  float4 xyz;
  float one_third = 1.0/3.0;
  xyz.x = (xyz_tmp.x > 0.008856) ? pow(xyz_tmp.x, one_third) : (7.787 * xyz_tmp.x) + 16.0/116.0;
  xyz.y = (xyz_tmp.y > 0.008856) ? pow(xyz_tmp.y, one_third) : (7.787 * xyz_tmp.y) + 16.0/116.0;
  xyz.z = (xyz_tmp.z > 0.008856) ? pow(xyz_tmp.z, one_third) : (7.787 * xyz_tmp.z) + 16.0/116.0;

  xyz.w = 1.0;

  float4 lab;
  lab.w = 1.0;

  lab.x = 116 * xyz.y - 16;
  lab.y = 500 * (xyz.x - xyz.y);
  lab.z = 200 * (xyz.y - xyz.z);
  
  return lab;
}
// 3 channel rgb to 3 channel lab conversion
__kernel void convert_rgb_images_to_lab(__global unsigned char* in_rgb_image_one,
										__global unsigned char* in_rgb_image_two,
										__global float* out_lab_image_one,
										__global float* out_lab_image_two,
										int image_width,
										int image_height
		) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_idx;


	// in case we do not have so many physical workers, let each one handle several pixels
	for(pixel_idx.y = thread_2d_index.y; pixel_idx.y < image_height; pixel_idx.y += thread_2d_sizes.y) {
		for(pixel_idx.x = thread_2d_index.x; pixel_idx.x < image_width; pixel_idx.x += thread_2d_sizes.x) {
			
			//convert 2d pixel idx to 1d pixel index
			int pixel_1d_index = get_index_2d_to_1d(pixel_idx.x, pixel_idx.y, image_width);
			int pixel_1d_channel_offset = 3 * pixel_1d_index;

			uchar4 rgb_value_one = {in_rgb_image_one[pixel_1d_channel_offset],
									in_rgb_image_one[pixel_1d_channel_offset + 1],
									in_rgb_image_one[pixel_1d_channel_offset + 2],
									1.0};

			float4 lab_value_one = convert_rgb_to_lab(rgb_value_one);

			out_lab_image_one[pixel_1d_channel_offset    ] = lab_value_one.x;
			out_lab_image_one[pixel_1d_channel_offset + 1] = lab_value_one.y;
			out_lab_image_one[pixel_1d_channel_offset + 2] = lab_value_one.z;

			uchar4 rgb_value_two = {in_rgb_image_two[pixel_1d_channel_offset],
									in_rgb_image_two[pixel_1d_channel_offset + 1],
									in_rgb_image_two[pixel_1d_channel_offset + 2],
									1.0};

			float4 lab_value_two = convert_rgb_to_lab(rgb_value_two);

			out_lab_image_two[pixel_1d_channel_offset    ] = lab_value_two.x;
			out_lab_image_two[pixel_1d_channel_offset + 1] = lab_value_two.y;
			out_lab_image_two[pixel_1d_channel_offset + 2] = lab_value_two.z;

		}
	}

}			



//compute a w(p,q)-term
__constant float gamma_color = 7.0; // according to paper
//__constant float gamma_position = 12; // according to paper

float4 get_color_from_image(int2 pos_2d, __global float* color_image, int image_width) {
	int pos_1d = get_index_2d_to_1d(pos_2d.x, pos_2d.y, image_width);
	int color_offset_1d = 3 * pos_1d;
	float4 color = {  color_image[color_offset_1d  /*+ 0*/],
					  color_image[color_offset_1d   + 1],
					  color_image[color_offset_1d   + 2],
					  1.0
					  };
	return color;
}

float compute_w_of_p_q(int2 p_pos_2d, int2 q_pos_2d, __global float* color_image, int image_width, int window_half_width) {
	float4 color_p = get_color_from_image(p_pos_2d, color_image, image_width);
	float4 color_q = get_color_from_image(q_pos_2d, color_image, image_width);
	float gamma_position = 0.8 * window_half_width;
	float scaled_color_delta   = distance(color_p, color_q) / gamma_color;

	float2 p_pos_2d_float = {p_pos_2d.x, p_pos_2d.y};
	float2 q_pos_2d_float = {q_pos_2d.x, q_pos_2d.y};
	float scaled_position_delta = distance(p_pos_2d_float, q_pos_2d_float) / gamma_position;

	return exp(-(scaled_color_delta + scaled_position_delta));
}

//helper function for ASW disparity 
void accumulate_ASW_costs(float* accumulated_weighted_color_differences,
						  float* accumulated_weight,
						  int window_half_width,
						  int2 reference_center_pixel_idx_2d,
						  int2 search_center_pixel_idx_2d,
						  int image_width,
						  __global float* lab_color_reference_image,
						  __global float* lab_color_search_image,
						  __global unsigned char* rgb_color_reference_image,
						  __global unsigned char* rgb_color_search_image) {

	for(int window_index_Y = - window_half_width; window_index_Y < window_half_width; ++window_index_Y) {
		for(int window_index_X = - window_half_width; window_index_X < window_half_width; ++window_index_X) {

			int2 search_neighbor_pixel_idx_2d = {search_center_pixel_idx_2d.x + window_index_X, search_center_pixel_idx_2d.y + window_index_Y};  
			int search_neighbor_pixel_idx_1d = get_index_2d_to_1d(search_neighbor_pixel_idx_2d.x, search_neighbor_pixel_idx_2d.y, image_width);
			

			int2 reference_neighbor_pixel_idx_2d = {reference_center_pixel_idx_2d.x + window_index_X, reference_center_pixel_idx_2d.y + window_index_Y};
		
			int reference_neighbor_pixel_idx_1d = get_index_2d_to_1d(reference_neighbor_pixel_idx_2d.x, reference_neighbor_pixel_idx_2d.y, image_width);
			

			float w_p_current_q = compute_w_of_p_q(reference_center_pixel_idx_2d, 
												   reference_neighbor_pixel_idx_2d,
												   lab_color_reference_image, image_width, window_half_width);

			float w_p_bar_q_bar = compute_w_of_p_q(search_center_pixel_idx_2d, 
											   	   search_neighbor_pixel_idx_2d,
											       lab_color_search_image, image_width, window_half_width);

			float current_weight = w_p_current_q * w_p_bar_q_bar;
			*accumulated_weight += current_weight;

			//float ref_grayscale_value = reference_pixel_window_sampled_grayscale_values[window_pixel_1d_offset];
			//color intensity diff (currently used: grayscale; paper uses diff over color channels)

			uchar3 rgb_reference_neighbor_pixel = {rgb_color_reference_image[reference_neighbor_pixel_idx_1d*3 + 0],
												   rgb_color_reference_image[reference_neighbor_pixel_idx_1d*3 + 1],
												   rgb_color_reference_image[reference_neighbor_pixel_idx_1d*3 + 2] };

			uchar3 rgb_search_neighbor_pixel = {rgb_color_search_image[search_neighbor_pixel_idx_1d*3 + 0],
												rgb_color_search_image[search_neighbor_pixel_idx_1d*3 + 1],
												rgb_color_search_image[search_neighbor_pixel_idx_1d*3 + 2] };

			int e_intermediate =    abs(rgb_reference_neighbor_pixel.x - rgb_search_neighbor_pixel.x)
								  + abs(rgb_reference_neighbor_pixel.y - rgb_search_neighbor_pixel.y)
								  + abs(rgb_reference_neighbor_pixel.z - rgb_search_neighbor_pixel.z);

			int e_q_qd = min(e_intermediate, 255);

			
			//float e_q_qd = 1.0 * fabs(ref_grayscale_value - grayscale_search_image[search_neighbor_pixel_idx_1d]);
			*accumulated_weighted_color_differences += current_weight * e_q_qd;
		}
	}
 }

 #define MAX_DISPARITY 60

__kernel void computeASWbasedDisparityMap(__global unsigned char* rgb_color_reference_image,  //rgb8f_in_image
										  __global unsigned char* rgb_color_search_image,  //rgb8f_in_image
										  __global float* lab_color_reference_image,	 // lab8f_in_image 
								  		  __global float* lab_color_search_image,		 // lab8f_in_image 
								          __global float* disparity_out_image,		 // r32f_in_image
								  		  int image_width,
								  		  int image_height,
								  		  int window_half_width) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
	int2 reference_center_pixel_idx_2d;

	if(thread_2d_index.x > window_half_width && thread_2d_index.x < image_width - window_half_width &&
	   thread_2d_index.y > window_half_width && thread_2d_index.y < image_height - window_half_width) {
		// in case we do not have so many physical workers, let each one handle several pixels
		for(reference_center_pixel_idx_2d.y = thread_2d_index.y; reference_center_pixel_idx_2d.y < image_height - window_half_width; reference_center_pixel_idx_2d.y += thread_2d_sizes.y) {
			for(reference_center_pixel_idx_2d.x = thread_2d_index.x; reference_center_pixel_idx_2d.x < image_width  - window_half_width; reference_center_pixel_idx_2d.x += thread_2d_sizes.x) {
				
				int reference_center_pixel_idx_1d = get_index_2d_to_1d(reference_center_pixel_idx_2d.x, reference_center_pixel_idx_2d.y, image_width);
	
				float best_absolute_diff = 999999.9; //???
				int best_search_pixel_x_idx = 0;


				int start_search_pixel_x_idx = max(window_half_width, reference_center_pixel_idx_2d.x - MAX_DISPARITY);
				int end_search_pixel_x_idx   = min(image_width - window_half_width, reference_center_pixel_idx_2d.x);
				bool use_ASW = true;
				
				for(int search_pixel_x_index = start_search_pixel_x_idx; search_pixel_x_index < end_search_pixel_x_idx; ++search_pixel_x_index){								
					int2 search_center_pixel_idx_2d = {search_pixel_x_index, reference_center_pixel_idx_2d.y};
					int search_center_pixel_idx_1d = get_index_2d_to_1d(search_center_pixel_idx_2d.x, search_center_pixel_idx_2d.y, image_width);

					
					float accumulated_weighted_color_differences = 0.0; //sums up upper part of E_p_pd
					float accumulated_weight = 0.0; //sums up lower part of E_p_pd

 					accumulate_ASW_costs(&accumulated_weighted_color_differences,
 										 &accumulated_weight,  window_half_width,
 										 reference_center_pixel_idx_2d,
 										 search_center_pixel_idx_2d, image_width,
 										 lab_color_reference_image, lab_color_search_image,
 										 rgb_color_reference_image, rgb_color_search_image);

					// actual E_p_pd
					float current_absolute_diff = accumulated_weighted_color_differences / accumulated_weight;
					if (best_absolute_diff > current_absolute_diff) {
						best_absolute_diff = current_absolute_diff;
						best_search_pixel_x_idx = search_pixel_x_index;
						
					}
				}
				disparity_out_image[reference_center_pixel_idx_1d] =  abs(reference_center_pixel_idx_2d.x - best_search_pixel_x_idx);

			}
		}
	}
}


//#define WINDOW_HALF_WIDTH  3
#define WINDOW_WIDTH 2 * WINDOW_HALF_WIDTH + 1


float computeSADmatchingCosts(/*float* ref_pixel_window_values, */
							  __global float* grayscale_reference_image,
							  __global float* grayscale_search_image,
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
				int current_1d_search_pixel_idx = get_index_2d_to_1d(current_search_pixel_index.x, current_search_pixel_index.y, image_width);
				
				int2 current_reference_pixel_index = {reference_pixel_2d_index.x + window_index_X, reference_pixel_2d_index.y + window_index_Y};  
				int current_1d_reference_pixel_idx = get_index_2d_to_1d(current_reference_pixel_index.x, current_reference_pixel_index.y, image_width);
				float ref_value = grayscale_reference_image[current_1d_reference_pixel_idx];
				float search_value = grayscale_search_image[current_1d_search_pixel_idx];
				current_absolute_diff += fabs(ref_value - search_value);
				++counter;
			}
		}

		current_absolute_diff /= counter;
		return current_absolute_diff;
}

__kernel void computeDisparityMap(__global float* grayscale_reference_image, // r32f_in_image
								  __global float* grayscale_search_image,	 // r32f_in_image
								  __global float* disparity_out_image,		 // r32f_in_image
								  int image_width,
								  int image_height,
								  int window_half_width) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
	int2 reference_pixel_idx;

	if(thread_2d_index.x > window_half_width && thread_2d_index.y > window_half_width /*&&
	   thread_2d_index.x < image_width - window_half_width && thread_2d_index.y < image_height - window_half_width*/ ){
		// in case we do not have so many physical workers, let each one handle several pixels
		for(reference_pixel_idx.y = thread_2d_index.y; reference_pixel_idx.y < image_height - window_half_width; reference_pixel_idx.y += thread_2d_sizes.y) {
			for(reference_pixel_idx.x = thread_2d_index.x; reference_pixel_idx.x < image_width  - window_half_width; reference_pixel_idx.x += thread_2d_sizes.x) {
				
				int reference_pixel_1d_index = get_index_2d_to_1d(reference_pixel_idx.x, reference_pixel_idx.y, image_width);
				
				float best_absolute_diff = 9999.9; //???
				int best_search_pixel_idx = 0;

				/*float ref_pixel_window_values[WINDOW_WIDTH * WINDOW_WIDTH];
				for(int window_index_Y = - window_half_width; window_index_Y < window_half_width; window_index_Y++) {
					for(int window_index_X = - window_half_width; window_index_X < window_half_width; window_index_X++) {
						int2 current_ref_pixel_index = {reference_pixel_idx.x + window_index_X, reference_pixel_idx.y + window_index_Y};
						int current_1d_ref_pixel_idx = get_index_2d_to_1d(current_ref_pixel_index.x, current_ref_pixel_index.y, image_width);
						
						int window_pixel_1d_offset = (window_index_X + window_half_width) 
												   + (window_index_Y + window_half_width) * WINDOW_WIDTH;

						ref_pixel_window_values[window_pixel_1d_offset] = grayscale_reference_image[current_1d_ref_pixel_idx];
					}
				}*/

				int start_search_pixel_idx = max(window_half_width, reference_pixel_idx.x - MAX_DISPARITY);
				int end_search_pixel_idx   = min(image_width - window_half_width, reference_pixel_idx.x);
				int2 search_pixel_2d_index;
				for(int search_pixel_index = start_search_pixel_idx; search_pixel_index < end_search_pixel_idx; search_pixel_index++){
    
					search_pixel_2d_index.x  = search_pixel_index;
					search_pixel_2d_index.y = reference_pixel_idx.y;
					float current_absolute_diff = computeSADmatchingCosts(/*&ref_pixel_window_values, */
																		   grayscale_reference_image,
							  					  						   grayscale_search_image,
							  					  						   reference_pixel_idx,
							  					  						   search_pixel_2d_index,
							  					  						   window_half_width,
							  					  						   image_width);

					if (best_absolute_diff > current_absolute_diff) {
						best_absolute_diff = current_absolute_diff;
						best_search_pixel_idx = search_pixel_index;
					}
					
				}
				//if(433 != best_search_pixel_idx){
					disparity_out_image[reference_pixel_1d_index] = abs(reference_pixel_idx.x - best_search_pixel_idx);
				/*}else{
					disparity_out_image[reference_pixel_1d_index] = 255;
				}*/
			}
		}
	}
}

__kernel void simplePatchMatchPropagation(__global float* grayscale_reference_image,	     // r32f_in_image 
								  	  	  __global float* grayscale_search_image,		     // r32f_in_image 
								  	  	  __global float* initial_random_guess_image,        // r32f_in_image
								      	  __global float* disparity_out_image,		         // r32f_out_image
								  	  	  int image_width,
								  	  	  int image_height,
								  	  	  int window_half_width,
								  	  	  int propagation_direction) {

	int thread_1d_index = get_global_id(0); //one thread per input image COLUMN OR per image ROW dependent on the itteration
	int num_threads = get_global_size(0);
	int2 reference_center_pixel_idx_2d;
	int start_column_ind; // = window_half_width; //leftwards propagaion
	int start_row_index; // = thread_1d_index;
	int col_increment; // = 1;
	int row_increment; // = num_threads;
	int end_column_ind = image_width - window_half_width;
	int end_row_ind = image_height - window_half_width;

	if(propagation_direction == 0 || propagation_direction == 2){ //leftwards OR rightwards
		start_column_ind = window_half_width + 1 ; 
	    start_row_index = thread_1d_index;
	    col_increment = 1;
	    row_increment = num_threads;
	}else if(propagation_direction == 1 || propagation_direction == 3){ //downwards OR  upwards propagaion
		start_column_ind = thread_1d_index;
		start_row_index = window_half_width + 1;
		col_increment = num_threads;
		row_increment = 1;
	}

	//const int window_width = 2 * window_half_width + 1;

	for(int column_ind_counter = start_column_ind; column_ind_counter < end_column_ind; column_ind_counter += col_increment) {
		for(int row_ind_counter = start_row_index; row_ind_counter < end_row_ind; row_ind_counter += row_increment){
				
			if(propagation_direction == 0){ //leftwards propagation
				reference_center_pixel_idx_2d.x = column_ind_counter;
				reference_center_pixel_idx_2d.y = row_ind_counter;
			}else if(propagation_direction == 1){ //downwards propagaion
				reference_center_pixel_idx_2d.x = column_ind_counter;
				reference_center_pixel_idx_2d.y = row_ind_counter;
			}else if(propagation_direction == 2){ //rightwards propagation
				reference_center_pixel_idx_2d.x = end_column_ind - column_ind_counter;
				reference_center_pixel_idx_2d.y = end_row_ind - row_ind_counter;
			}else if(propagation_direction == 3){ //upwards propagation
				reference_center_pixel_idx_2d.x = end_column_ind - column_ind_counter;
				reference_center_pixel_idx_2d.y = end_row_ind - row_ind_counter;
			}


			if(    (reference_center_pixel_idx_2d.y - window_half_width < 0)
			    || (reference_center_pixel_idx_2d.y + window_half_width >= image_height)
			    || (reference_center_pixel_idx_2d.x - window_half_width < 0)
			    || (reference_center_pixel_idx_2d.x + window_half_width >= image_width)
			  ) {
				continue;
			}

			//coppy grayscale ref_img patch values that will be used multiple times
			/*float ref_pixel_window_values[WINDOW_WIDTH * WINDOW_WIDTH];
			for(int window_index_Y = - window_half_width; window_index_Y < window_half_width; window_index_Y++) {
				for(int window_index_X = - window_half_width; window_index_X < window_half_width; window_index_X++) {
					int2 current_ref_pixel_index = {reference_center_pixel_idx_2d.x + window_index_X, reference_center_pixel_idx_2d.y + window_index_Y};
					int current_1d_ref_pixel_idx = get_index_2d_to_1d(current_ref_pixel_index.x, current_ref_pixel_index.y, image_width);
					int window_pixel_1d_offset = (window_index_X + window_half_width) 
											   + (window_index_Y + window_half_width) * WINDOW_WIDTH;
					ref_pixel_window_values[window_pixel_1d_offset] = grayscale_reference_image[current_1d_ref_pixel_idx];
				}
			}*/

			int reference_center_pixel_idx_1d= get_index_2d_to_1d(reference_center_pixel_idx_2d.x, reference_center_pixel_idx_2d.y, image_width);
			float own_disp_offset = initial_random_guess_image[reference_center_pixel_idx_1d];
			int2 own_search_center_pixel_idx_2d = {reference_center_pixel_idx_2d.x - own_disp_offset,
												   reference_center_pixel_idx_2d.y};

			//assuming propagation_direction == 0  //LEFTWARDS
			int2 neighbour_pixel_idx_2d = {reference_center_pixel_idx_2d.x -1,
										   reference_center_pixel_idx_2d.y};
			if(propagation_direction == 1){ //DOWNWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y - 1;
			}else if(propagation_direction == 2){//RIGHTWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x + 1;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y;
			}else if(propagation_direction == 3){//UPWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x + 1;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y;
			}


			float matching_costs_using_curent_Pd = computeSADmatchingCosts(/*ref_pixel_window_values,*/
																			grayscale_reference_image,
																		    grayscale_search_image,
																		    reference_center_pixel_idx_2d,
																		    own_search_center_pixel_idx_2d,
																		    window_half_width, image_width);
			float current_best_disp_value = own_disp_offset;
			float smallest_matching_costs = matching_costs_using_curent_Pd;


			int neighbour_pixel_ind_1d = get_index_2d_to_1d(neighbour_pixel_idx_2d.x, neighbour_pixel_idx_2d.y, image_width);
			float neighbour_disp_offset = initial_random_guess_image[neighbour_pixel_ind_1d];
			int2 neighbour_search_center_pixel_idx_2d = {reference_center_pixel_idx_2d.x - neighbour_disp_offset, 
														 reference_center_pixel_idx_2d.y};
			int row_range_limit = reference_center_pixel_idx_2d.x - window_half_width;

			//check if the disparity proposed by the neighbouring pixel is in the allowed range
			//NOTE: assuming that search image pixels are shifted to the left relative to the reference image

				if(neighbour_disp_offset <= row_range_limit){
					
					float matching_costs_using_neighbor_Qd = computeSADmatchingCosts(/*ref_pixel_window_values,*/
																					 grayscale_reference_image,
																		    		 grayscale_search_image,
																		    		 reference_center_pixel_idx_2d,
																		    		 neighbour_search_center_pixel_idx_2d,
																		    		 window_half_width, image_width);
					
					if(matching_costs_using_curent_Pd - matching_costs_using_neighbor_Qd > 0.0){
						current_best_disp_value = neighbour_disp_offset;
						smallest_matching_costs = matching_costs_using_neighbor_Qd;
					}
				}

				// Random search step
				//******
				//if(propagation_direction == 1 || propagation_direction == 3){
					
					for(int new_disp_offset = row_range_limit; new_disp_offset >= 1; new_disp_offset /= 2){
		                
		                int random_x_pixel_position = abs(reference_center_pixel_idx_2d.x - new_disp_offset);
		                int new_x_pixel_position = min(random_x_pixel_position, row_range_limit);
						int2 alternative_search_center_pixel_idx_2d = {new_x_pixel_position, 
																       reference_center_pixel_idx_2d.y};

						float matching_costs_using_new_randomD = computeSADmatchingCosts(/*ref_pixel_window_values,*/
																       					 grayscale_reference_image,
																		                 grayscale_search_image,
																		                 reference_center_pixel_idx_2d,
																		                 alternative_search_center_pixel_idx_2d,
																		                 window_half_width, image_width);
						if(smallest_matching_costs - matching_costs_using_new_randomD > 0.0){
							current_best_disp_value = abs(reference_center_pixel_idx_2d.x - new_x_pixel_position);
							smallest_matching_costs = matching_costs_using_new_randomD;
						}
					}
				//}

			//write out final result
			disparity_out_image[reference_center_pixel_idx_1d] =  current_best_disp_value;	
		}
	}
}

__kernel void patchMatchPropagation(__global unsigned char* rgb_color_reference_image, //rgb8f_in_image
									  __global unsigned char* rgb_color_search_image,    //rgb8f_in_image
									  __global float* lab_color_reference_image,	     // lab8f_in_image 
								  	  __global float* lab_color_search_image,		     // lab8f_in_image 
								  	  __global float* initial_random_guess_image,        //r32f_in_image
								      __global float* disparity_out_image,		         // r32f_out_image
								  	  int image_width,
								  	  int image_height,
								  	  int window_half_width,
								  	  int propagation_direction) {

	int thread_1d_index = get_global_id(0); //one thread per input image COLUMN OR per image ROW dependent on the itteration
	int num_threads = get_global_size(0);
	int2 reference_center_pixel_idx_2d;
	int start_column_ind; // = window_half_width; //leftwards propagaion
	int start_row_index; // = thread_1d_index;
	int col_increment; // = 1;
	int row_increment; // = num_threads;
	int end_column_ind = image_width - window_half_width;
	int end_row_ind = image_height - window_half_width;

	if(propagation_direction == 0 || propagation_direction == 2){ //leftwards OR rightwards
		start_column_ind = window_half_width + 1 ; 
	    start_row_index = thread_1d_index;
	    col_increment = 1;
	    row_increment = num_threads;
	}else if(propagation_direction == 1 || propagation_direction == 3){ //downwards OR  upwards propagaion
		start_column_ind = thread_1d_index;
		start_row_index = window_half_width + 1;
		col_increment = num_threads;
		row_increment = 1;
	}

	//for(reference_center_pixel_idx_2d.y = start_column_ind; reference_center_pixel_idx_2d.y < end_column_ind; reference_center_pixel_idx_2d.y += col_increment) {
		//for(reference_center_pixel_idx_2d.x = start_row_index; reference_center_pixel_idx_2d.x < end_row_ind; reference_center_pixel_idx_2d.x += row_increment){
		
	for(int column_ind_counter = start_column_ind; column_ind_counter < end_column_ind; column_ind_counter += col_increment) {
		for(int row_ind_counter = start_row_index; row_ind_counter < end_row_ind; row_ind_counter += row_increment){
				
			if(propagation_direction == 0){ //leftwards propagation
				reference_center_pixel_idx_2d.x = column_ind_counter;
				reference_center_pixel_idx_2d.y = row_ind_counter;
			}else if(propagation_direction == 1){ //downwards propagaion
				reference_center_pixel_idx_2d.x = column_ind_counter;
				reference_center_pixel_idx_2d.y = row_ind_counter;
			}else if(propagation_direction == 2){ //rightwards propagation
				reference_center_pixel_idx_2d.x = end_column_ind - column_ind_counter;
				reference_center_pixel_idx_2d.y = end_row_ind - row_ind_counter;
			}else if(propagation_direction == 3){ //upwards propagation
				reference_center_pixel_idx_2d.x = end_column_ind - column_ind_counter;
				reference_center_pixel_idx_2d.y = end_row_ind - row_ind_counter;
			}


			if(    (reference_center_pixel_idx_2d.y - window_half_width < 0)
			    || (reference_center_pixel_idx_2d.y + window_half_width >= image_height)
			    || (reference_center_pixel_idx_2d.x - window_half_width < 0)
			    || (reference_center_pixel_idx_2d.x + window_half_width >= image_width)
			  ) {
				continue;
			}

			
			int reference_center_pixel_idx_1d= get_index_2d_to_1d(reference_center_pixel_idx_2d.x, reference_center_pixel_idx_2d.y, image_width);
			float own_disp_offset = initial_random_guess_image[reference_center_pixel_idx_1d];
			int2 own_search_center_pixel_idx_2d = {reference_center_pixel_idx_2d.x - own_disp_offset,
												   reference_center_pixel_idx_2d.y};

			//assuming propagation_direction == 0  //LEFTWARDS
			int2 neighbour_pixel_idx_2d = {reference_center_pixel_idx_2d.x -1,
										   reference_center_pixel_idx_2d.y};
			if(propagation_direction == 1){ //DOWNWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y - 1;
			}else if(propagation_direction == 2){//RIGHTWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x + 1;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y;
			}else if(propagation_direction == 3){//UPWARDS
				neighbour_pixel_idx_2d.x = reference_center_pixel_idx_2d.x + 1;
				neighbour_pixel_idx_2d.y = reference_center_pixel_idx_2d.y;
			}

			//sums up upper part of E_p_pd 
			float own_accumulated_weighted_color_differences = 0.0;
			//sums up lower part of E_p_pd
			float own_accumulated_weight = 0.0;
			accumulate_ASW_costs(&own_accumulated_weighted_color_differences,
									   &own_accumulated_weight,  window_half_width,
									   reference_center_pixel_idx_2d,
									   own_search_center_pixel_idx_2d, image_width,
									   lab_color_reference_image, lab_color_search_image,
									   rgb_color_reference_image, rgb_color_search_image);



			float matching_costs_using_curent_Pd = own_accumulated_weighted_color_differences / own_accumulated_weight;
			float current_best_disp_value = own_disp_offset;
			float smallest_matching_costs = matching_costs_using_curent_Pd;


			int neighbour_pixel_ind_1d = get_index_2d_to_1d(neighbour_pixel_idx_2d.x, neighbour_pixel_idx_2d.y, image_width);
			float neighbour_disp_offset = initial_random_guess_image[neighbour_pixel_ind_1d];
			int2 neighbour_search_center_pixel_idx_2d = {reference_center_pixel_idx_2d.x - neighbour_disp_offset, 
														 reference_center_pixel_idx_2d.y};
			int row_range_limit = reference_center_pixel_idx_2d.x - window_half_width;

			//check if the disparity proposed by the neighbouring pixel is in the allowed range
			//NOTE: assuming that search image pixels are shifted to the left relative to the reference image

				if(neighbour_disp_offset <= row_range_limit){

					//sums up upper part of E_p_pd 
					float neighbour_accumulated_weighted_color_differences = 0.0;
					//sums up lower part of E_p_pd
					float neighbour_accumulated_weight = 0.0;
					accumulate_ASW_costs(&neighbour_accumulated_weighted_color_differences,
									     &neighbour_accumulated_weight,  window_half_width,
									     reference_center_pixel_idx_2d, //reference window stays the same => Code CAN be OPTIMIZED
									     neighbour_search_center_pixel_idx_2d, image_width,
									     lab_color_reference_image, lab_color_search_image,
									     rgb_color_reference_image, rgb_color_search_image);

					
					float matching_costs_using_neighbor_Qd = neighbour_accumulated_weighted_color_differences / neighbour_accumulated_weight;
					
					if(matching_costs_using_curent_Pd - matching_costs_using_neighbor_Qd > 0.0){
						current_best_disp_value = neighbour_disp_offset;
						smallest_matching_costs = matching_costs_using_neighbor_Qd;
						//disparity_out_image[reference_center_pixel_idx_1d] = neighbour_disp_offset; //initial_random_guess_image[neighbour_pixel_ind_1d];
					}
				}

				// Random search step
				//******
				//if(propagation_direction == 1 || propagation_direction == 3){
					
					for(int new_disp_offset = row_range_limit; new_disp_offset >= 1; new_disp_offset /= 2){
		                
		                int random_x_pixel_position = abs(reference_center_pixel_idx_2d.x - new_disp_offset);
		                int new_x_pixel_position = min(random_x_pixel_position, row_range_limit);
						int2 alternative_search_center_pixel_idx_2d = {new_x_pixel_position, 
																       reference_center_pixel_idx_2d.y};

						//sums up upper part of E_p_pd 
						float alternative_accumulated_weighted_color_differences = 0.0;
						//sums up lower part of E_p_pd
						float alternative_accumulated_weight = 0.0;
						accumulate_ASW_costs(&alternative_accumulated_weighted_color_differences,
											 &alternative_accumulated_weight,  window_half_width,
											 reference_center_pixel_idx_2d,
											 alternative_search_center_pixel_idx_2d, image_width,
											 lab_color_reference_image, lab_color_search_image,
											 rgb_color_reference_image, rgb_color_search_image);

						float matching_costs_using_new_randomD = alternative_accumulated_weighted_color_differences / alternative_accumulated_weight;
						if(smallest_matching_costs - matching_costs_using_new_randomD > 0.0){
							current_best_disp_value = abs(reference_center_pixel_idx_2d.x - new_x_pixel_position);
							smallest_matching_costs = matching_costs_using_new_randomD;
						}
					}
				//}

			//write out final result
			disparity_out_image[reference_center_pixel_idx_1d] =  current_best_disp_value;	
		}
	}
}

__constant float2 rand_helper_vector = {12.9898, 4.1414};

float rand(float2 n) { 
	//source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	float intermediate_result = sin(dot(n, rand_helper_vector)) * 43758.5453;
	return intermediate_result - floor(intermediate_result);
}

__kernel void randomDisparityMapInitialization(int image_width,
								  			   int image_height,
								  			   unsigned int max_disparity,
								  			   int window_half_width,
								               __global float* r32f_out_image) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
	int2 pixel_idx;
	// in case we do not have so many physical workers, let each one handle several pixels
	for(pixel_idx.y = thread_2d_index.y; pixel_idx.y < image_height; pixel_idx.y += thread_2d_sizes.y) {
		for(pixel_idx.x = thread_2d_index.x; pixel_idx.x < image_width; pixel_idx.x += thread_2d_sizes.x) {
			//unsigned int seed = (pixel_idx.y << 16) + (pixel_idx.x << 8);
			
			float2 normalized_pixel_pos = {convert_float(pixel_idx.x)  / image_width, convert_float(pixel_idx.y)   / image_height };
			
			int x_minus_max_disparity = pixel_idx.x - window_half_width;

			int maximum_allowed_disparity_by_x_position = min(convert_int(max_disparity), max(x_minus_max_disparity, 0) );

			float random_inital_value = maximum_allowed_disparity_by_x_position * rand(normalized_pixel_pos);
			//float random_inital_value = get_random_Z_value(max_disparity, seed); //TODO: change function name!
			int pixel_1d_index = get_index_2d_to_1d(pixel_idx.x, pixel_idx.y, image_width);

			r32f_out_image[pixel_1d_index] = random_inital_value;
		}
	}
}

/*__kernel void planeInitialization(__global float* r32f_in_image,
								  __global float* rgbd32f_out_image,
								  int image_width,
								  int image_height,
								  float max_disparity) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
	int2 pixel_idx;
	// in case we do not have so many physical workers, let each one handle several pixels
	for(pixel_idx.y = thread_2d_index.y; pixel_idx.y < image_height; pixel_idx.y += thread_2d_sizes.y) {
		for(pixel_idx.x = thread_2d_index.x; pixel_idx.x < image_width; pixel_idx.x += thread_2d_sizes.x) {
		
			unsigned int seed = (pixel_idx.y << 16) + pixel_idx.x;
			float pixel_idx_Z = get_random_Z_value(max_disparity, seed); 
			float3 random_normal = {0, 0, 1}; //inital test with fixed normal for fronto-parallel planes

			//convert 2d pixel idx to 1d pixel index
			int pixel_1d_index = get_index_2d_to_1d(pixel_idx.x, pixel_idx.y, image_width);
			rgbd32f_out_image[pixel_1d_index + 0] = pixel_idx_Z;
			rgbd32f_out_image[pixel_1d_index + 1] = random_normal.x;
			rgbd32f_out_image[pixel_1d_index + 2] = random_normal.y;
			rgbd32f_out_image[pixel_1d_index + 3] = random_normal.z;
		}
	}

}*/