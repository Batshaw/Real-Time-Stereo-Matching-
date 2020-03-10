__constant int forced_num_channels = 3; //RGBA

__constant float3 camera_position ={0.0f, 0.0f, 0.0f};


__kernel void reconstruct_3D_points_GPU(__global float* input_disparity_map,
                                        __global float* output_vector,
                                        int image_width, int image_height,
                                        float baseline, float focal_length,
                                        float disparity_scaling,
                                        __global unsigned char* input_color_map,
                                        __global unsigned char* output_color_map,
                                        int min_valid_disparity){
                                       
    int2 const pixel_2d_index = {get_global_id(0), get_global_id(1)};
    int pixel_1D_index = pixel_2d_index.x + pixel_2d_index.y * image_width;
    
    int pixel_1D_position_with_channel_offset = pixel_1D_index * forced_num_channels;
    float current_pixel_disparity = input_disparity_map[pixel_1D_index];
    
    float z_coord;
    if(min_valid_disparity > current_pixel_disparity )
    {
      z_coord = 0.0f;
    }
    else
    {
      z_coord = (focal_length * baseline) / (current_pixel_disparity * disparity_scaling);
    }
    
    // compute the ray back-projected from pixel to the camera (like ray-tracing)
    float3 current_image_plane_point = {pixel_2d_index.x - image_width*0.5f, pixel_2d_index.y - image_height * 0.5f, focal_length};
    

    float3 current_ray_direction = current_image_plane_point - camera_position;
    float3 normalized_ray_direction = normalize(current_ray_direction);

    // camera_position + t * normalized_ray_direction = xyz_posion
    float t = maxmag(0 , (z_coord / normalized_ray_direction.z)); // return 0 if |t| < 0
    float3 new_3D_position = t * normalized_ray_direction;
    
    unsigned char current_r = input_color_map[pixel_1D_position_with_channel_offset + 2];
    unsigned char current_g = input_color_map[pixel_1D_position_with_channel_offset + 1]; 
    unsigned char current_b = input_color_map[pixel_1D_position_with_channel_offset + 0]; 
  
    uint point_3d_base_offset = 3*pixel_1D_index; 
    output_vector[point_3d_base_offset + 0] = new_3D_position.x;
    output_vector[point_3d_base_offset + 1] = new_3D_position.y;
    output_vector[point_3d_base_offset + 2] = new_3D_position.z;

    output_color_map[point_3d_base_offset + 0] = current_r;
    output_color_map[point_3d_base_offset + 1] = current_g;
    output_color_map[point_3d_base_offset + 2] = current_b;
}



__kernel void reconstruct_3D_colored_triangles_GPU(__global float* input_disparity_map,
                                           __global float* output_vector,
                                           int image_width, int image_height,
                                           float baseline, float focal_length,
                                           float disparity_scaling,
                                           __global unsigned char* input_color_map,
                                           __global unsigned char* output_color_map,
                                           int min_valid_disparity,
                                           int use_distance_threshold){
                                       
    int2 const pixel_2d_index = {get_global_id(0), get_global_id(1)};

    float3 quad_depth_values[4];
    uchar3 quad_color_values[4];

    uint quad_write_index = 0;
    for(int square_index_y = 0; square_index_y < 2; ++square_index_y) {
        for(int square_index_x = 0; square_index_x < 2; ++square_index_x) { 


            int pixel_1D_read_index = (  pixel_2d_index.x + square_index_x) 
                                      + (pixel_2d_index.y + square_index_y) * image_width;
            
            int pixel_1D_position_with_channel_offset = pixel_1D_read_index * forced_num_channels;
            float current_pixel_disparity = input_disparity_map[pixel_1D_read_index];
            
            float z_coord;
            if(min_valid_disparity > current_pixel_disparity )
            {
              z_coord = 0.0f;
            }
            else
            {
              z_coord = (focal_length * baseline) / (current_pixel_disparity);
            }
            

            // compute the ray back-projected from pixel to the camera (like ray-tracing)
            float3 current_image_plane_point = {(pixel_2d_index.x + square_index_x) - image_width*0.5f, (pixel_2d_index.y  + square_index_y)- image_height * 0.5f, focal_length};
            

            float3 current_ray_direction = current_image_plane_point - camera_position;
            float3 normalized_ray_direction = normalize(current_ray_direction);

            // camera_position + t * normalized_ray_direction = xyz_posion
            float t = maxmag(0 , (z_coord / normalized_ray_direction.z)); // return 0 if |t| < 0
            float3 new_3D_position = t * normalized_ray_direction;
            
            quad_depth_values[quad_write_index] = new_3D_position;
    
            uchar3 current_rgb;
            current_rgb.x = input_color_map[pixel_1D_position_with_channel_offset + 2];
            current_rgb.y = input_color_map[pixel_1D_position_with_channel_offset + 1];
            current_rgb.z = input_color_map[pixel_1D_position_with_channel_offset + 0];
             
            quad_color_values[quad_write_index] = current_rgb;

            ++quad_write_index; 
        }
    }

    uint one_d_pixel_write_index =  pixel_2d_index.x + pixel_2d_index.y * (image_width-1);
    uint base_triangle_offset = 3 * 3 * 2 * one_d_pixel_write_index;


    float triangle_depth_gradients[6] = {0.0, 0.0, 0.0, 0.0, 0.0};
    triangle_depth_gradients[0] = fabs(quad_depth_values[0].z - quad_depth_values[1].z);
    triangle_depth_gradients[1] = fabs(quad_depth_values[1].z - quad_depth_values[2].z);
    triangle_depth_gradients[2] = fabs(quad_depth_values[0].z - quad_depth_values[2].z);

    triangle_depth_gradients[3] = fabs(quad_depth_values[2].z - quad_depth_values[1].z); 
    triangle_depth_gradients[4] = fabs(quad_depth_values[2].z - quad_depth_values[3].z); 
    triangle_depth_gradients[5] = fabs(quad_depth_values[1].z - quad_depth_values[3].z);


    float max_depth_triangle_1 = 0.0;
    float max_depth_triangle_2 = 0.0;
    for(int vertex_idx = 0; vertex_idx < 3; ++vertex_idx) {
        max_depth_triangle_1 = max(max_depth_triangle_1, quad_depth_values[vertex_idx].z);
        max_depth_triangle_2 = max(max_depth_triangle_2, quad_depth_values[vertex_idx + 1].z);
    }



    float max_depth_discont_triangle_1 = max(triangle_depth_gradients[2], max(triangle_depth_gradients[0], triangle_depth_gradients[1] ) );
    
    
    float max_depth_discont_triangle_2 = max(triangle_depth_gradients[3], max(triangle_depth_gradients[4], triangle_depth_gradients[5] ) );

    float max_depth_discontinuity = 0.03f;

    float max_depth_distance = 999999999.0;

    if(use_distance_threshold) {
        max_depth_distance = 3.2f;
    }
    if(max_depth_discont_triangle_1 < max_depth_discontinuity && max_depth_triangle_1 < max_depth_distance) {
    //if(true) {
        output_vector[base_triangle_offset + 0] = quad_depth_values[0].x;
        output_vector[base_triangle_offset + 1] = quad_depth_values[0].y;
        output_vector[base_triangle_offset + 2] = quad_depth_values[0].z;

        output_vector[base_triangle_offset + 3] = quad_depth_values[1].x;
        output_vector[base_triangle_offset + 4] = quad_depth_values[1].y;
        output_vector[base_triangle_offset + 5] = quad_depth_values[1].z;

        output_vector[base_triangle_offset + 6] = quad_depth_values[2].x;
        output_vector[base_triangle_offset + 7] = quad_depth_values[2].y;
        output_vector[base_triangle_offset + 8] = quad_depth_values[2].z;

        output_color_map[base_triangle_offset + 0] = quad_color_values[0].x;
        output_color_map[base_triangle_offset + 1] = quad_color_values[0].y;
        output_color_map[base_triangle_offset + 2] = quad_color_values[0].z;

        output_color_map[base_triangle_offset + 3] = quad_color_values[1].x;
        output_color_map[base_triangle_offset + 4] = quad_color_values[1].y;
        output_color_map[base_triangle_offset + 5] = quad_color_values[1].z;

        output_color_map[base_triangle_offset + 6] = quad_color_values[2].x;
        output_color_map[base_triangle_offset + 7] = quad_color_values[2].y;
        output_color_map[base_triangle_offset + 8] = quad_color_values[2].z;
    } else {
        output_vector[base_triangle_offset + 0] = 0.0f;
        output_vector[base_triangle_offset + 1] = 0.0f;
        output_vector[base_triangle_offset + 2] = 0.0f;

        output_vector[base_triangle_offset + 3] = 0.0f;
        output_vector[base_triangle_offset + 4] = 0.0f;
        output_vector[base_triangle_offset + 5] = 0.0f;

        output_vector[base_triangle_offset + 6] = 0.0f;
        output_vector[base_triangle_offset + 7] = 0.0f;
        output_vector[base_triangle_offset + 8] = 0.0f;

        output_color_map[base_triangle_offset + 0] = 0;
        output_color_map[base_triangle_offset + 1] = 0;
        output_color_map[base_triangle_offset + 2] = 0;

        output_color_map[base_triangle_offset + 3] = 0;
        output_color_map[base_triangle_offset + 4] = 0;
        output_color_map[base_triangle_offset + 5] = 0;

        output_color_map[base_triangle_offset + 6] = 0;
        output_color_map[base_triangle_offset + 7] = 0;
        output_color_map[base_triangle_offset + 8] = 0;
    }

    if(max_depth_discont_triangle_2 < max_depth_discontinuity && max_depth_triangle_2 < max_depth_distance) {
    //if(true) {
        output_vector[base_triangle_offset + 9] =  quad_depth_values[1].x;
        output_vector[base_triangle_offset + 10] = quad_depth_values[1].y;
        output_vector[base_triangle_offset + 11] = quad_depth_values[1].z;

        output_vector[base_triangle_offset + 12] = quad_depth_values[2].x;
        output_vector[base_triangle_offset + 13] = quad_depth_values[2].y;
        output_vector[base_triangle_offset + 14] = quad_depth_values[2].z;

        output_vector[base_triangle_offset + 15] = quad_depth_values[3].x;
        output_vector[base_triangle_offset + 16] = quad_depth_values[3].y;
        output_vector[base_triangle_offset + 17] = quad_depth_values[3].z;

        output_color_map[base_triangle_offset +  9] = quad_color_values[1].x;
        output_color_map[base_triangle_offset + 10] = quad_color_values[1].y;
        output_color_map[base_triangle_offset + 11] = quad_color_values[1].z;

        output_color_map[base_triangle_offset + 12] = quad_color_values[2].x;
        output_color_map[base_triangle_offset + 13] = quad_color_values[2].y;
        output_color_map[base_triangle_offset + 14] = quad_color_values[2].z;

        output_color_map[base_triangle_offset + 15] = quad_color_values[3].x;
        output_color_map[base_triangle_offset + 16] = quad_color_values[3].y;
        output_color_map[base_triangle_offset + 17] = quad_color_values[3].z;
    } else {
        output_vector[base_triangle_offset +  9] = 0.0f;
        output_vector[base_triangle_offset + 10] = 0.0f;
        output_vector[base_triangle_offset + 11] = 0.0f;

        output_vector[base_triangle_offset + 12] = 0.0f;
        output_vector[base_triangle_offset + 13] = 0.0f;
        output_vector[base_triangle_offset + 14] = 0.0f;

        output_vector[base_triangle_offset + 15] = 0.0f;
        output_vector[base_triangle_offset + 16] = 0.0f;
        output_vector[base_triangle_offset + 17] = 0.0f;

        output_color_map[base_triangle_offset +  9] = 0;
        output_color_map[base_triangle_offset + 10] = 0;
        output_color_map[base_triangle_offset + 11] = 0;

        output_color_map[base_triangle_offset + 12] = 0;
        output_color_map[base_triangle_offset + 13] = 0;
        output_color_map[base_triangle_offset + 14] = 0;

        output_color_map[base_triangle_offset + 15] = 0;
        output_color_map[base_triangle_offset + 16] = 0;
        output_color_map[base_triangle_offset + 17] = 0;
    }
}


__kernel void reconstruct_3D_textured_triangles_GPU(__global float* input_disparity_map,
                                                    __global float* output_vector,
                                                    int image_width, int image_height,
                                                    float baseline, float focal_length,
                                                    float disparity_scaling,
                                                    __global unsigned char* input_color_map,
                                                    __global unsigned char* output_color_map,
                                                    int min_valid_disparity){
                                       
    int2 const pixel_2d_index = {get_global_id(0), get_global_id(1)};

    float3 quad_depth_values[4];

    float2 quad_texture_coords[4];

    uint quad_write_index = 0;
    for(int square_index_y = 0; square_index_y < 2; ++square_index_y) {
        for(int square_index_x = 0; square_index_x < 2; ++square_index_x) { 


            int pixel_1D_read_index = (  pixel_2d_index.x + square_index_x) 
                                      + (pixel_2d_index.y + square_index_y) * image_width;
            
            int pixel_1D_position_with_channel_offset = pixel_1D_read_index * forced_num_channels;
            float current_pixel_disparity = input_disparity_map[pixel_1D_read_index];
            
            float z_coord;
            if(min_valid_disparity > current_pixel_disparity )
            {
              z_coord = 0.0f;
            }
            else
            {
              z_coord = (focal_length * baseline) / (current_pixel_disparity * disparity_scaling);
            }
            

            // compute the ray back-projected from pixel to the camera (like ray-tracing)
            float3 current_image_plane_point = {(pixel_2d_index.x + square_index_x) - image_width*0.5f, (pixel_2d_index.y  + square_index_y)- image_height * 0.5f, focal_length};
            

            float3 current_ray_direction = current_image_plane_point - camera_position;
            float3 normalized_ray_direction = normalize(current_ray_direction);

            // camera_position + t * normalized_ray_direction = xyz_posion
            float t = maxmag(0 , (z_coord / normalized_ray_direction.z)); // return 0 if |t| < 0
            float3 new_3D_position = t * normalized_ray_direction;
            
            quad_depth_values[quad_write_index] = new_3D_position;
    
            float2 current_uvs;
            current_uvs.x = ((  (pixel_2d_index.x + square_index_x) + 0.5 ) /(float)(image_width)) * (image_width / 2560.0);
            current_uvs.y = ((  (pixel_2d_index.y + square_index_y) + 0.5 ) /(float)(image_height)) * (image_height / 1440.0);

             
            quad_texture_coords[quad_write_index] = current_uvs;

            ++quad_write_index;
        }
    }

    uint one_d_pixel_write_index =  pixel_2d_index.x + pixel_2d_index.y * (image_width-1);
    uint base_triangle_offset = 5 * 3 * 2 * one_d_pixel_write_index;


    float triangle_depth_gradients[6] = {0.0, 0.0, 0.0, 0.0, 0.0};
    triangle_depth_gradients[0] = fabs(quad_depth_values[0].z - quad_depth_values[1].z);
    triangle_depth_gradients[1] = fabs(quad_depth_values[1].z - quad_depth_values[2].z);
    triangle_depth_gradients[2] = fabs(quad_depth_values[0].z - quad_depth_values[2].z);

    triangle_depth_gradients[3] = fabs(quad_depth_values[2].z - quad_depth_values[1].z); 
    triangle_depth_gradients[4] = fabs(quad_depth_values[2].z - quad_depth_values[3].z); 
    triangle_depth_gradients[5] = fabs(quad_depth_values[1].z - quad_depth_values[3].z);

    float max_depth_discont_triangle_1 = max(triangle_depth_gradients[2], max(triangle_depth_gradients[0], triangle_depth_gradients[1] ) );
    
    
    float max_depth_discont_triangle_2 = max(triangle_depth_gradients[3], max(triangle_depth_gradients[4], triangle_depth_gradients[5] ) );

    float max_depth_discontinuity = 0.1;
    if(max_depth_discont_triangle_1 < max_depth_discontinuity) {
    //if(false) {
        output_vector[base_triangle_offset +  0] = quad_depth_values[0].x; //x
        output_vector[base_triangle_offset +  1] = quad_depth_values[0].y; //y
        output_vector[base_triangle_offset +  2] = quad_depth_values[0].z; //z
        output_vector[base_triangle_offset +  3] = quad_texture_coords[0].x; //u
        output_vector[base_triangle_offset +  4] = quad_texture_coords[0].y; //v     

        output_vector[base_triangle_offset +  5] = quad_depth_values[1].x;
        output_vector[base_triangle_offset +  6] = quad_depth_values[1].y;
        output_vector[base_triangle_offset +  7] = quad_depth_values[1].z;
        output_vector[base_triangle_offset +  8] = quad_texture_coords[1].x; //u
        output_vector[base_triangle_offset +  9] = quad_texture_coords[1].y; //v     


        output_vector[base_triangle_offset + 10] = quad_depth_values[2].x;
        output_vector[base_triangle_offset + 11] = quad_depth_values[2].y;
        output_vector[base_triangle_offset + 12] = quad_depth_values[2].z;
        output_vector[base_triangle_offset + 13] = quad_texture_coords[2].x; //u
        output_vector[base_triangle_offset + 14] = quad_texture_coords[2].y; //v     
    } else {
        output_vector[base_triangle_offset +  0] = 0.0f; //x
        output_vector[base_triangle_offset +  1] = 0.0f; //y
        output_vector[base_triangle_offset +  2] = 0.0f; //z
        output_vector[base_triangle_offset +  3] = 0.0f; //u
        output_vector[base_triangle_offset +  4] = 0.0f; //v     

        output_vector[base_triangle_offset +  5] = 0.0f;
        output_vector[base_triangle_offset +  6] = 0.0f;
        output_vector[base_triangle_offset +  7] = 0.0f;
        output_vector[base_triangle_offset +  8] = 0.0f; //u
        output_vector[base_triangle_offset +  9] = 0.0f; //v     


        output_vector[base_triangle_offset + 10] = 0.0f;
        output_vector[base_triangle_offset + 11] = 0.0f;
        output_vector[base_triangle_offset + 12] = 0.0f;
        output_vector[base_triangle_offset + 13] = 0.0f; //u
        output_vector[base_triangle_offset + 14] = 0.0f; //v     
    }

    if(max_depth_discont_triangle_2 < max_depth_discontinuity) {
    //if(false) {
        output_vector[base_triangle_offset + 15] =  quad_depth_values[1].x;
        output_vector[base_triangle_offset + 16] = quad_depth_values[1].y;
        output_vector[base_triangle_offset + 17] = quad_depth_values[1].z;
        output_vector[base_triangle_offset + 18] = quad_texture_coords[1].x;
        output_vector[base_triangle_offset + 19] = quad_texture_coords[1].y;

        output_vector[base_triangle_offset + 20] = quad_depth_values[2].x;
        output_vector[base_triangle_offset + 21] = quad_depth_values[2].y;
        output_vector[base_triangle_offset + 22] = quad_depth_values[2].z;
        output_vector[base_triangle_offset + 23] = quad_texture_coords[2].x;
        output_vector[base_triangle_offset + 24] = quad_texture_coords[2].y;

        output_vector[base_triangle_offset + 25] = quad_depth_values[3].x;
        output_vector[base_triangle_offset + 26] = quad_depth_values[3].y;
        output_vector[base_triangle_offset + 27] = quad_depth_values[3].z;
        output_vector[base_triangle_offset + 28] = quad_texture_coords[3].x;
        output_vector[base_triangle_offset + 29] = quad_texture_coords[3].y;
    } else {
        output_vector[base_triangle_offset + 15] = 0.0f;
        output_vector[base_triangle_offset + 16] = 0.0f;
        output_vector[base_triangle_offset + 17] = 0.0f;
        output_vector[base_triangle_offset + 18] = 0.0f;
        output_vector[base_triangle_offset + 19] = 0.0f;

        output_vector[base_triangle_offset + 20] = 0.0f;
        output_vector[base_triangle_offset + 21] = 0.0f;
        output_vector[base_triangle_offset + 22] = 0.0f;
        output_vector[base_triangle_offset + 23] = 0.0f;
        output_vector[base_triangle_offset + 24] = 0.0f;

        output_vector[base_triangle_offset + 25] = 0.0f;
        output_vector[base_triangle_offset + 26] = 0.0f;
        output_vector[base_triangle_offset + 27] = 0.0f;
        output_vector[base_triangle_offset + 28] = 0.0f;
        output_vector[base_triangle_offset + 29] = 0.0f;
    }
}