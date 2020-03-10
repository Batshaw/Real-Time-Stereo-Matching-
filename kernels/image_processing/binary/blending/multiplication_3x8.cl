__constant int num_channels = 3;

/*** forward declarations of device functions ***/
void _multiply_pixels_3x8(int2 pixel_2d_index,  
				 			__global unsigned char* input_image_1, __global unsigned char* input_image_2, 
							__global unsigned char* output_image, int image_width, int image_height);


/*** main kernels ***/
__kernel void multiply_images_3x8(__global unsigned char* input_image_1,
									     __global unsigned char* input_image_2,
							 		     __global unsigned char* output_image,
							 		int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			_multiply_pixels_3x8(pixel_2d_index, input_image_1, input_image_2, output_image, image_width, image_height);
		}
	}
}


/*** definition of device functions ***/
void _multiply_pixels_3x8(int2 pixel_2d_index,  
		  					     __global unsigned char* input_image_1, __global unsigned char* input_image_2, 
		  					     __global unsigned char* output_image, int image_width, int image_height) {

	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
	int pixel_1d_offset = num_channels * pixel_1d_index;

	float3 pixel_color_1 = {input_image_1[pixel_1d_offset + 0],
		   			        input_image_1[pixel_1d_offset + 1],
					        input_image_1[pixel_1d_offset + 2]};

	// one of the pixel colors needs to be normalized for the multiplication to work as expected
	float3 pixel_color_2 = {input_image_2[pixel_1d_offset + 0] / 255.0f,
		   			        input_image_2[pixel_1d_offset + 1] / 255.0f,
					        input_image_2[pixel_1d_offset + 2] / 255.0f};



	int3 multiplication_result = {pixel_color_1.x * pixel_color_2.x,
								  pixel_color_1.y * pixel_color_2.y,
								  pixel_color_1.z * pixel_color_2.z};

	output_image[pixel_1d_offset + 0] = multiplication_result.x;
	output_image[pixel_1d_offset + 1] = multiplication_result.y;
	output_image[pixel_1d_offset + 2] = multiplication_result.z;	
}