/*__constant float mask[9] = {
  0.077847f,  0.123317f,  0.077847f,
  0.123317f,  0.195346f,  0.123317f,
  0.077847f,  0.123317f,  0.077847f
};

__constant int maskSize = 1;
*/
// __constant float mask[25] = {
// 0.003765,  0.015019,  0.023792,  0.015019,  0.003765,
// 0.015019,  0.059912,  0.094907,  0.059912,  0.015019,
// 0.023792,  0.094907,  0.150342,  0.094907,  0.023792,
// 0.015019,  0.059912,  0.094907,  0.059912,  0.015019,
// 0.003765,  0.015019,  0.023792,  0.015019,  0.003765
// };

// __constant int maskSize = 2;

__kernel void gaussian_blur(__global unsigned char* input_image,
                            __global unsigned char* output_image,
                            int image_width, int image_height,
							              __constant float* mask,
                            int maskSize) {

                      
	int2 pixel_2d_index = {get_global_id(0), get_global_id(1)}; // retrieve 2D index for current compute element
	int pixel_1d_index = pixel_2d_index.x + pixel_2d_index.y * image_width; // "flatten" index 2D -> 1D
	int num_channels = 3; // for element offset calculation
	int pixel_1d_offset = num_channels * pixel_1d_index; // actual position of 1D pixel
    


    float sum_red = 0.0f;
    float sum_green = 0.0f;
    float sum_blue = 0.0f;
    
    for (int neigh_y_idx = (pixel_2d_index.y - maskSize); neigh_y_idx <= (pixel_2d_index.y + maskSize); ++neigh_y_idx) {
        for (int neigh_x_idx = (pixel_2d_index.x - maskSize); neigh_x_idx <= (pixel_2d_index.x + maskSize); ++neigh_x_idx) {
          if (neigh_x_idx < 0 || neigh_x_idx > image_width || neigh_y_idx < 0 || neigh_y_idx > image_height) {
              continue;
          }


          int neighbor_pixel_1d_index = neigh_x_idx + neigh_y_idx * image_width; // "flatten" neighbor index 2D -> 1D
          int neighbor_pixel_1d_offset = num_channels * neighbor_pixel_1d_index; // actual position of 1D neighbor pixel
              
          // distance to mean pixel and add maskSize for a poistive (array) Index 
          int local_x_index = (neigh_x_idx - pixel_2d_index.x) + maskSize;
          int local_y_index = (neigh_y_idx - pixel_2d_index.y) + maskSize;
    
          int window_size_1d = maskSize * 2 + 1;
          int sobel_window_index_1d = local_x_index + local_y_index * window_size_1d;
    
          sum_red   += mask[sobel_window_index_1d] * input_image[neighbor_pixel_1d_offset + 0] ;
          sum_green += mask[sobel_window_index_1d] * input_image[neighbor_pixel_1d_offset + 1] ;
          sum_blue  += mask[sobel_window_index_1d] * input_image[neighbor_pixel_1d_offset + 2] ;
      }
    }

    //clamp values, if they are out of color range 
    sum_red   = clamp(sum_red,   0.0f,255.0f);
    sum_green = clamp(sum_green, 0.0f,255.0f);
    sum_blue  = clamp(sum_blue,  0.0f,255.0f);

    outputimage[pixel_1d_offset + 0] = sum_red;
    output__image[pixel_1d_offset + 1] = sum_green;
    output_image[pixel_1d_offset + 2] = sum_blue; 



}