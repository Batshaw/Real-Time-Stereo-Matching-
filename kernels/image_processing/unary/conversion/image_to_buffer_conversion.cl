__kernel void convert_image_to_buffer (__read_only image2d_t grayscale_input_image,
                                       __global unsigned char* grayscale_1x8_output_image,
                                       int image_width, int image_height)
{
	int2 pixel_coord = {get_global_id(0), get_global_id(1)};
    int pixel_1d_index = pixel_coord.x + pixel_coord.y * image_width;
    uint4 c = read_imageui(grayscale_input_image, pixel_coord);
	grayscale_1x8_output_image[pixel_1d_index] = c.x;
}

__kernel void convert_image_to_buffer_1x32f (__read_only image2d_t grayscale_input_image,
                                       __global float* grayscale_1x8_output_image,
                                       int image_width, int image_height)
{
  int2 pixel_coord = {get_global_id(0), get_global_id(1)};
    int pixel_1d_index = pixel_coord.x + pixel_coord.y * image_width;
    uint4 c = read_imageui(grayscale_input_image, pixel_coord);
    grayscale_1x8_output_image[pixel_1d_index] = c.x;
}

__kernel void convert_image_float_to_buffer(__read_only image2d_t grayscale_input_image,
                                        __global unsigned char* grayscale_1x8_output_image,
                                        int image_width, int image_height)
{
	int2 pixel_coord = {get_global_id(0), get_global_id(1)};
    int pixel_1d_index = pixel_coord.x + pixel_coord.y * image_width;
    float4 c = read_imagef(grayscale_input_image, pixel_coord);
	grayscale_1x8_output_image[pixel_1d_index] = (int)c.x;
}
