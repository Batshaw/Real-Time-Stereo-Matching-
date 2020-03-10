__constant int num_channels = 3;
__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
__kernel void copy_bgr_3x8_buffer_to_image(__global unsigned char* bgr_3x8_input_buffer,
							  					__write_only image2d_t output_image,
							  					int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    int const pixel_1d_offset = num_channels * pixel_1d_index;

    uint4 pixel_color = {bgr_3x8_input_buffer[pixel_1d_offset + 2],
                        bgr_3x8_input_buffer[pixel_1d_offset + 1],
                        bgr_3x8_input_buffer[pixel_1d_offset + 0],
                        0};

    write_imageui(output_image, thread_2d_index, pixel_color);
}

__kernel void copy_image_to_buffer_bgr_3x8_buffer(__read_only image2d_t input_image,
                                           __global unsigned char* out_bgr_3x8_buffer,
                                           int image_width, int image_height) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    int const pixel_1d_offset = num_channels * pixel_1d_index;

    uint4 pixel_color = read_imageui(input_image, sampler, thread_2d_index);

    out_bgr_3x8_buffer[pixel_1d_offset + 2] = pixel_color.x;
    out_bgr_3x8_buffer[pixel_1d_offset + 1] = pixel_color.y;
    out_bgr_3x8_buffer[pixel_1d_offset + 0] = pixel_color.z;
}

__kernel void copy_image_to_buffer_1x8_buffer(__read_only image2d_t input_image,
                                           __global unsigned char* out_bgr_1x8_buffer,
                                           int image_width, int image_height) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

    uint4 pixel_color = read_imageui(input_image, sampler, thread_2d_index);

    out_bgr_1x8_buffer[pixel_1d_index] = pixel_color.x;
}

__kernel void copy_image_float_to_buffer_1x8_buffer(__read_only image2d_t input_image,
                                           __global unsigned char* out_bgr_1x8_buffer,
                                           int image_width, int image_height) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

    float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

    out_bgr_1x8_buffer[pixel_1d_index] = (int)pixel_color.x;
}

__kernel void copy_image_float_to_buffer_1x32f_buffer(__read_only image2d_t input_image,
                                                      __global float* out_bgr_1x8_buffer,
                                                      int image_width, int image_height) {

  int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

    float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

    out_bgr_1x8_buffer[pixel_1d_index] = pixel_color.x;
}


__kernel void copy_3D_image_to_buffer_1x8_buffer(__read_only image3d_t input_volume,
                                           __global unsigned char* out_bgr_1x8_buffer,
                                           int image_width, int image_height, int d) {

    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int const pixel_1d_index = x + y * image_width;

    float4 cost = read_imagef(input_volume, sampler, (int4)(x, y, d, 0));

    out_bgr_1x8_buffer[pixel_1d_index] = (uchar)cost.x;
}

