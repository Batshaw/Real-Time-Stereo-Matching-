__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant int num_channels_rgb = 3;

__kernel void copy_grayscale_3x8_buffer_to_image_2D(__global unsigned char* grayscale_3x8_input_buffer,
							  					__write_only image2d_t output_image,
							  					int image_width, int image_height) {
    
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
                                                   
    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    int const pixel_1d_offset = num_channels_rgb * pixel_1d_index;
    //buffer includes rgb channel with the same values 

    float gray_val_unnormalized = (grayscale_3x8_input_buffer[pixel_1d_offset + 2] +
                                   grayscale_3x8_input_buffer[pixel_1d_offset + 1] +
                                   grayscale_3x8_input_buffer[pixel_1d_offset + 0]) / 3.0;

    float4 pixel_color = { gray_val_unnormalized / 255.0f,
                           0.0f,
                           0.0f,
                           1.0f};

    write_imagef(output_image, thread_2d_index, pixel_color);
}

__kernel void copy_3x8_buffer_to_image_2D(__global unsigned char* grayscale_3x8_input_buffer,
							  					__write_only image2d_t output_image,
							  					int image_width, int image_height) {
    
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
                                                   
    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    int const pixel_1d_offset = num_channels_rgb * pixel_1d_index;
    //buffer includes rgb channel with the same values 

    // float gray_val_unnormalized = (grayscale_3x8_input_buffer[pixel_1d_offset + 2] +
    //                                grayscale_3x8_input_buffer[pixel_1d_offset + 1] +
    //                                grayscale_3x8_input_buffer[pixel_1d_offset + 0]) / 3.0;

    float4 pixel_color = { grayscale_3x8_input_buffer[pixel_1d_offset + 2]/255.0f,
                           grayscale_3x8_input_buffer[pixel_1d_offset + 1]/255.0f,
                           grayscale_3x8_input_buffer[pixel_1d_offset + 0]/255.0f,
                           1.0f};

    write_imagef(output_image, thread_2d_index, pixel_color);
}

// __kernel void copy_image_2D_to_1x8_buffer(__read_only image2d_t input_image,
//                                            __global unsigned char* out_bgr_1x8_buffer,
//                                            int image_width, int image_height, float max_disparity) {

// 	float2 pos = {get_global_id(0), get_global_id(1)};
//     float4 input_value = read_imagef(input_image, sampler, (float2)(pos.x, pos.y));

//     float current_value_z = input_value.x; 
//     float3 current_normal_vector = {input_value.y, input_value.z, input_value.w};
//     float output_disparity = compute_disparity(pos, current_value_z, current_normal_vector);
//     float normalized_output_disparity = output_disparity /max_disparity;

//     int const pixel_1d_index = pos.x + pos.y * image_width;
//     out_bgr_1x8_buffer[pixel_1d_index] = normalized_output_disparity * 255.0f;//
// }

__kernel void copy_image_2D_to_1x8_buffer(__read_only image2d_t input_image,
                                           __global float* out_bgr_1x8_buffer,
                                           int image_width, int image_height) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

    float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

    out_bgr_1x8_buffer[pixel_1d_index] = pixel_color.x;



}

// __kernel void copy_image_float_to_buffer_1x8_buffer(__read_only image2d_t input_image,
//                                            __global unsigned char* out_bgr_1x8_buffer,
//                                            int image_width, int image_height) {

// 	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

//     int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

//     float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

//     out_bgr_1x8_buffer[pixel_1d_index] = (int)pixel_color.x;
// }


// __kernel void copy_image_2D_to_3x8_buffer(__read_only image2d_t input_image,
//                                            __global float* out_bgr_3x8_buffer,
//                                            int image_width, int image_height) {

// 	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

//     int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
//     int const pixel_1d_offset = num_channels_rgb * pixel_1d_index;

//     float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

//     out_bgr_3x8_buffer[pixel_1d_index + 2] = pixel_color.x;
//     out_bgr_3x8_buffer[pixel_1d_index + 1] = pixel_color.y;
//     out_bgr_3x8_buffer[pixel_1d_index + 0] = pixel_color.z;

// }

// __kernel void copy_r_1x8_buffer_to_image_2D(__global unsigned char* r_1x8_input_buffer,
// 							  					__write_only image2d_t output_image,
// 							  					int image_width, int image_height) {
// 	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

//     int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    

//     float4 pixel_color = {r_1x8_input_buffer[pixel_1d_index ] / 255.0f,
//                         0.0f,
//                         0.0f,
//                         1.0f};

//     write_imagef(output_image, thread_2d_index, pixel_color);
// }
