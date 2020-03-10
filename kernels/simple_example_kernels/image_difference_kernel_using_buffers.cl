
__kernel void compute_difference_image (__global unsigned char* input_image_1,
							  			__global unsigned char* input_image_2,
							  			__global unsigned char* output_image,
							   		    int image_width, int image_height) {

	int2 pixel_2d_index = {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width; // "flatten" index 2D -> 1D
	int num_channels = 3; // for element offset calculation
	int pixel_1d_offset = num_channels * pixel_1d_index; // actual position of 1D pixel

	/*Note: We assume here that we launch the kernel with as many processing elements as we have pixels to work on,
	        otherwise we would only write to parts of the output image.
	
	  To make sure that you always work on all pixels, independently of the work group size, look at more general kernels
	  (e.g. the one in kernels/image_processing/binary/blending/difference_3x8.cl , which performs essentially the same
	        operation as this kernel here)
	*/

	for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
		output_image[pixel_1d_offset + channel_idx] = abs(input_image_1[pixel_1d_offset + channel_idx] -
														  input_image_2[pixel_1d_offset + channel_idx]);
	}
	

}