__kernel void best_disp_winner_take_all(__global float* disp_cost,
                                        __global unsigned char* disp_img,
                                         int image_width,
                                         int image_height,
                                         int max_disp)
{
    int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
    int2 thread_2d_sizes = {get_global_size(0), get_global_size(1)};
    int2 reference_pixel_idx;

    for(reference_pixel_idx.y = thread_2d_index.y; reference_pixel_idx.y < image_height; reference_pixel_idx.y += thread_2d_sizes.y)
    {
        for(reference_pixel_idx.x = thread_2d_index.x; reference_pixel_idx.x < image_width; reference_pixel_idx.x += thread_2d_sizes.x)
        {
            int base_cost_1d_index = max_disp*(reference_pixel_idx.x + reference_pixel_idx.y * image_width);
            unsigned char min_cost = disp_cost[base_cost_1d_index];

            unsigned char best_d = 0;
            for (int d = 0; d < max_disp; ++d) {
                int cost_pos = d + base_cost_1d_index;

                int disp_cost_cost_pos = disp_cost[cost_pos];
                if (disp_cost_cost_pos < min_cost){
                    best_d = d;
                    min_cost = disp_cost_cost_pos;
                }
            }

            disp_img[reference_pixel_idx.x + reference_pixel_idx.y * image_width] = best_d;
            //disp_img[reference_pixel_idx.x + reference_pixel_idx.y * image_width] = min_cost;
            //disp_img[reference_pixel_idx.x + reference_pixel_idx.y * image_width] = disp_cost[10 + max_disp*(reference_pixel_idx.x + reference_pixel_idx.y * image_width)];
        }
    }
}
