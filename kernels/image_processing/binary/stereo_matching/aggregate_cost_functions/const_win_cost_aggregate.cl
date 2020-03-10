__kernel void const_win_cost_aggregate(__global float* cost_img_in,
                                       const int image_width, const int image_height,
                                       const int max_disp, const int half_win_size,
                                      __global float* cost_img_out) {
                      
    const int win_size = (half_win_size+1)*(half_win_size+1);
    int2 pixel_2d_index = {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
    int  pixel_1d_index  = max_disp * (pixel_2d_index.x + pixel_2d_index.y * image_width);

    for(int d = 0; d < max_disp; ++d){

        float cosg_avg = 0.0f;

        for (int neigh_y_idx = (pixel_2d_index.y - half_win_size); neigh_y_idx <= (pixel_2d_index.y + half_win_size); ++neigh_y_idx) {
            for (int neigh_x_idx = (pixel_2d_index.x - half_win_size); neigh_x_idx <= (pixel_2d_index.x + half_win_size); ++neigh_x_idx) {

              if (neigh_x_idx < 0 || neigh_x_idx > image_width || neigh_y_idx < 0 || neigh_y_idx > image_height) {
                  continue;
              }

              int neighbor_cost_1d_index = d + max_disp*(neigh_x_idx + neigh_y_idx * image_width);

              cosg_avg   += (float)(cost_img_in[neighbor_cost_1d_index]);
            }
        }

        cosg_avg = cosg_avg / (float)(win_size);

        cost_img_out[d + pixel_1d_index] = (cosg_avg);
    }
}
