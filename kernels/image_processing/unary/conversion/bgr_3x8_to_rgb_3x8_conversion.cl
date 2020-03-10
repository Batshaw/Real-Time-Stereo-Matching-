
__constant float3 luminance_grayscale_conversion_weights = {0.212671, 0.71516,  0.072169};

__constant int num_channels = 3;

/*** forward declarations of device functions ***/
void _convert_pixel_bgr_3x8_to_rgb_3x8(int pixel_1d_offset,  __global unsigned char* input_image, __global unsigned char* output_image, int image_width, int image_height);


/*** main kernels ***/
__kernel void convert_image_bgr_3x8_to_rgb_3x8 (__global unsigned char* input_image,
							  					__global unsigned char* output_image,
							  					int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		int pixel_row_offset = pixel_2d_index.y * image_width;
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			int pixel_1d_index = pixel_2d_index.x + pixel_row_offset;
			int pixel_1d_offset = num_channels * pixel_1d_index;
			_convert_pixel_bgr_3x8_to_rgb_3x8(pixel_1d_offset, input_image, output_image, image_width, image_height);
		}
	}
}

/*** definition of device functions ***/
void _convert_pixel_bgr_3x8_to_rgb_3x8(int pixel_1d_offset, 
								 	   __global unsigned char* input_image,
								 	   __global unsigned char* output_image,
								 	   int image_width, int image_height) {


	// write out bgr to rgb by swapping first and last channel
	output_image[pixel_1d_offset    ] = input_image[pixel_1d_offset + 2];
	output_image[pixel_1d_offset + 2] = input_image[pixel_1d_offset    ];
	
}