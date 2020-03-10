__constant int num_channels = 3;

/*** forward declarations of device functions ***/
void _subtract_absolute_pixels_3x8(int2 pixel_2d_index,  
				 		 	       __global unsigned char* input_image_1, __global unsigned char* input_image_2, 
							       __global unsigned char* output_image, int image_width, int image_height);


/*** main kernels ***/
__kernel void subtract_absolute_images_3x8(__global unsigned char* input_image_1,
									       __global unsigned char* input_image_2,
							 		       __global unsigned char* output_image,
							 		       int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			_subtract_absolute_pixels_3x8(pixel_2d_index, input_image_1, input_image_2, output_image, image_width, image_height);
		}
	}
}


/*** definition of device functions ***/
void _subtract_absolute_pixels_3x8(int2 pixel_2d_index,  
		  					       __global unsigned char* input_image_1, __global unsigned char* input_image_2, 
		  					       __global unsigned char* output_image, int image_width, int image_height) {

	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
	int pixel_1d_offset = num_channels * pixel_1d_index;

	// swap red and blue channels @ index 2 and 0 in order to convert BGR->RgB
	int3 pixel_color_1 = {input_image_1[pixel_1d_offset + 0],
		   			      input_image_1[pixel_1d_offset + 1],
					      input_image_1[pixel_1d_offset + 2]};

	int3 pixel_color_2 = {input_image_2[pixel_1d_offset + 0],
		   			      input_image_2[pixel_1d_offset + 1],
					      input_image_2[pixel_1d_offset + 2]};


	uint3 subtraction_result = abs(pixel_color_1 - pixel_color_2);

	output_image[pixel_1d_offset + 0] = subtraction_result.x;
	output_image[pixel_1d_offset + 1] = subtraction_result.y;
	output_image[pixel_1d_offset + 2] = subtraction_result.z;	
}