__constant float Sx[9] = {
  1.0f,  0.0f,  -1.0f,
  2.0f,  0.0f,  -2.0f,
  1.0f,  0.0f,  -1.0f
};

__constant float Sy[9] = {
   1.0f,    2.0f,    1.0f,
   0.0f,    0.0f,    0.0f,
  -1.0f,    -2.0f,  -1.0f
};



__constant int maskSize = 1;


__kernel void sobel_filter (__global unsigned char* input_image,
                            __global unsigned char* output_image,
							              int image_width, int image_height){


	int2 const pixel_2d_index = {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width; // "flatten" index 2D -> 1D
	int num_channels = 3; // for element offset calculation
	int pixel_1d_offset = num_channels * pixel_1d_index; // actual position of 1D pixel


  float grad_x = 0.0f;
  float grad_y = 0.0f;



    for (int neigh_y_idx = (pixel_2d_index.y - maskSize); neigh_y_idx <= (pixel_2d_index.y + maskSize); ++neigh_y_idx) {
      for (int neigh_x_idx = (pixel_2d_index.x - maskSize); neigh_x_idx <= (pixel_2d_index.x + maskSize); ++neigh_x_idx) {
        if (neigh_x_idx < 0 || neigh_x_idx > image_width || neigh_y_idx < 0 || neigh_y_idx > image_height) {
            continue;
        }

        int neighbor_pixel_1d_index = neigh_x_idx + neigh_y_idx * image_width; // "flatten" neighbor index 2D -> 1D
        int neighbor_pixel_1d_offset = num_channels * neighbor_pixel_1d_index; // actual position of 1D neighbor pixel

        int local_x_index = (neigh_x_idx - pixel_2d_index.x) + maskSize;
        int local_y_index = (neigh_y_idx - pixel_2d_index.y) + maskSize;

        float neighbor_red   = input_image[neighbor_pixel_1d_offset + 2];
        float neighbor_green = input_image[neighbor_pixel_1d_offset + 1];
        float neighbor_blue  = input_image[neighbor_pixel_1d_offset + 0];

  	    float neighbor_grayscale = 0.2627f*neighbor_red + 0.678f*neighbor_green + 0.0593f*neighbor_blue;

        int window_size_1d = maskSize * 2 + 1;
        int sobel_window_index_1d = local_x_index + local_y_index * window_size_1d;

        grad_x += Sx[sobel_window_index_1d] * neighbor_grayscale;
        grad_y += Sy[sobel_window_index_1d] * neighbor_grayscale;
      }

    }

    float2 grad_2d = {grad_x, grad_y};
    float grad_mag = sqrt(grad_2d.x * grad_2d.x + grad_2d.y * grad_2d.y);
    grad_mag = clamp(grad_mag,0.0f,255.0f);
      
    output_image[pixel_1d_offset + 2] = grad_mag;
    output_image[pixel_1d_offset + 1] = grad_mag;
    output_image[pixel_1d_offset + 0] = grad_mag;
}

