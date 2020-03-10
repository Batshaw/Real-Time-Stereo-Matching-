__kernel void compute_adapt_weight_grey(
                                const __global unsigned char *img,
                                __global float* out_weight,
                                const float gamma_sim,
                                const float gamma_prox,
                                const int image_width, const int image_height,
                                const int half_win_size) {
                      
    const int win_size = (half_win_size+1)*(half_win_size+1);
    int2 idx_2d =  {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
    int  idx_1d  = idx_2d.x + idx_2d.y * image_width;

    unsigned char center_val = img[idx_1d];
    int w_idx = 0;
    int y_img = 0;
    int x_img = 0;
    for (int y = -half_win_size; y <= half_win_size; y++){
        y_img = y + idx_2d.y;
        if (y_img >= 0 && y_img < image_height){
            for (int x = -half_win_size; x <= half_win_size; x++){
                x_img = x +idx_2d.x;
                if (x_img >= 0 && x_img < image_width){
                    int cur_idx_1d =  x_img + y_img * image_width;
                    unsigned char dif  = abs(img[cur_idx_1d] - center_val);
                    float prox_w = exp(-sqrt((float)x*x+y*y)/gamma_prox);
                    out_weight[w_idx + idx_1d * win_size] = (float)prox_w * exp(-dif/gamma_sim);
                    w_idx += 1;
                }
            }
        }
    }

}
