
__constant float3 luminance_grayscale_conversion_weights = {0.212671f, 0.71516f,  0.072169f};

__constant int num_channels = 3;

/*** forward declarations of device functions ***/
void _convert_bgr_3x8_to_grayscale_3x8(int2 pixel_2d_index,  __global unsigned char* input_image, __global unsigned char* output_image, int image_width, int image_height);

/*** main kernels ***/
// input  : buffer bgr_3x8_input_image 3x8 bit unsigned char in bgr space
// output : buffer grayscale_3x8_output_image 3x8 bit unsigned char containing luminance greyscale
__kernel void convert_image_bgr_3x8_to_grayscale_3x8 (__global unsigned char* bgr_3x8_input_image,
							  						  __global unsigned char* grayscale_3x8_output_image,
							  			              int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
	int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};

	int2 pixel_2d_index = {-1, -1};

	// iterate over several pixels per thread, because the image might be larger than our total num threads
	for(pixel_2d_index.y = thread_2d_index.y; pixel_2d_index.y < image_height; pixel_2d_index.y += thread_2d_sizes.y) {
		for(pixel_2d_index.x = thread_2d_index.x; pixel_2d_index.x < image_width; pixel_2d_index.x += thread_2d_sizes.x) {
			_convert_bgr_3x8_to_grayscale_3x8(pixel_2d_index, bgr_3x8_input_image, grayscale_3x8_output_image, image_width, image_height);
		}
	}
}

/*** definition of device functions ***/
void _convert_bgr_3x8_to_grayscale_3x8(int2 pixel_2d_index, 
									__global unsigned char* bgr_3x8_input_image,
									__global unsigned char* grayscale_3x8_output_image,
									int image_width, int image_height) {

	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
	int pixel_1d_offset = num_channels * pixel_1d_index;

	// swap red and blue channels @ index 2 and 0 in order to convert BGR->RgB
	float3 pixel_color = {bgr_3x8_input_image[pixel_1d_offset + 2],
		   			      bgr_3x8_input_image[pixel_1d_offset + 1],
					      bgr_3x8_input_image[pixel_1d_offset + 0]};

	// use dot product for luma = w0*c0 + w1*c1 * w2*c2, because it is built-in
	int luminance_grayscale = dot(luminance_grayscale_conversion_weights, pixel_color);

	for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx){
		grayscale_3x8_output_image[pixel_1d_offset + channel_idx] = luminance_grayscale;
	}
}