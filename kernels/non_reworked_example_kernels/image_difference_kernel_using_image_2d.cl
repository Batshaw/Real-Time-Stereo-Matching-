//Reference source: https://anteru.net/blog/2012/getting-started-with-opencl-part-3/


// this is a 

// the sampler variable used in read_imageui tells openCL how it should treat the image
__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE  //Access the image using integer coordinates
    | CLK_ADDRESS_CLAMP_TO_EDGE    //Instead of reading accross image borders, access the border elements
    | CLK_FILTER_NEAREST;          //If we do not sample exactly in the pixel center, do not interpolate

__kernel void compute_difference_image (__read_only image2d_t input_image_1,
							  		  	__read_only image2d_t input_image_2,
							      	  	__write_only image2d_t output_image,
							          	int image_width) {

	// retrieve 2D index from thread. Note: Here, we do not need to convert the index from 2D to 1D anymore, because 
	// cl-Images are 2D structures and can be accessed directly with 2D indices
	int2 pixel_2d_index = {get_global_id(0), get_global_id(1)};

	//RGBA value with dummy alpha channel       read rgb value and add dummy alpha channel using cl built in functions
	int4 sampled_color_from_image_1 = convert_int4(read_imageui(input_image_1, sampler, pixel_2d_index));
	//RGBA value with dummy alpha channel       read rgb value and add dummy alpha channel using cl built in functions
	int4 sampled_color_from_image_2 = convert_int4(read_imageui(input_image_2, sampler, pixel_2d_index));

	//compute image difference
	uint4 color_difference = abs(sampled_color_from_image_1 - sampled_color_from_image_2);


	// write out result
	write_imageui(output_image, pixel_2d_index, color_difference);
}