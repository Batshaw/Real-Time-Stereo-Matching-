__kernel void compute_SAD_cost_green_3x8(__global unsigned char* bgr_ref_img, // r32f_in_image
                                         __global unsigned char* bgr_search_img,	 // r32f_in_image
                                         __global float* out_cost,		 // r32f_in_image
                                         int image_width,
                                         int image_height,
                                         int window_half_width,
                                         int max_disp) {

    int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
    int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
    int2 reference_pixel_idx;

	// in case we do not have so many physical workers, let each one handle several pixels
    int d;
    for (d = 0; d < max_disp; d++)
    {
        for(reference_pixel_idx.y = thread_2d_index.y; reference_pixel_idx.y < image_height - window_half_width; reference_pixel_idx.y += thread_2d_sizes.y)
        {
            for(reference_pixel_idx.x = thread_2d_index.x; reference_pixel_idx.x < image_width  - window_half_width; reference_pixel_idx.x += thread_2d_sizes.x)
            {

                if(reference_pixel_idx.x > window_half_width && reference_pixel_idx.y > window_half_width){
                        int reference_pixel_1d_index = (reference_pixel_idx.x + reference_pixel_idx.y * image_width);
                        int search_pixel_x = reference_pixel_idx.x - d;
                        if (search_pixel_x < 0)
                            search_pixel_x + image_width;

                        int search_pixel_1d_index = search_pixel_x + reference_pixel_idx.y * image_width;
                        out_cost[d + reference_pixel_1d_index * max_disp] = abs(bgr_ref_img[3*reference_pixel_1d_index+1] - bgr_search_img[3*search_pixel_1d_index + 1]);
                }
            }
        }
    }
}
