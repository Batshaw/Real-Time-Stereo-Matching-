
__constant float3 luminance_grayscale_conversion_weights = {0.212671, 0.71516,  0.072169};

__constant int num_channels = 3;


__kernel void convert_image_bgr_3x8_to_grayscale_1x8_image (__global unsigned char* bgr_3x8_input_image,
                                                            __write_only image2d_t grayscale_1x8_output_image,
                                                            int image_width, int image_height)
{
	int2 pixel_coord = {get_global_id(0), get_global_id(1)};

	//_convert_bgr_3x8_to_grayscale_1x8(pixel_2d_index, bgr_3x8_input_image, grayscale_1x8_output_image, image_width, image_height);
    int pixel_1d_index = pixel_coord.x + pixel_coord.y * image_width;
    int pixel_1d_offset_3x8 = 3 * pixel_1d_index;
    float3 pixel_color = {bgr_3x8_input_image[pixel_1d_offset_3x8 + 2],
            		   	  bgr_3x8_input_image[pixel_1d_offset_3x8 + 1],
            			  bgr_3x8_input_image[pixel_1d_offset_3x8 + 0]};
            
    // use dot product for luma = w0*c0 + w1*c1 * w2*c2, because it is built-in
    int luminance_grayscale = dot(luminance_grayscale_conversion_weights, pixel_color);
    write_imagef(grayscale_1x8_output_image, pixel_coord, (float4)(luminance_grayscale / 255.0, 0., 0., 1.));
}
