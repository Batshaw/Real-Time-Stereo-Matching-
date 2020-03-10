__constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant sampler_t linear_sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// __constant float2 rand_helper_vector = {12.9898, 4.1414};
    __constant float2 rand_helper_vector = {12.9898,78.233};


// __constant float max_disparity = 60.0f;
// __constant float min_disparity = 15.0f;

float rand(float2 n) {
  // source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
  float intermediate_result =
      (0.5 * (1 + sin(dot(n, rand_helper_vector)))) * 43758.5453;
  return intermediate_result - floor(intermediate_result);
}

float rand_slanted_plane_value(float2 n) {
  // source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
  float intermediate_result = /* (0.5 * (1 + sin(dot(n, rand_helper_vector)))) * 43758.5453; */ sin(dot(n, rand_helper_vector));
  return intermediate_result;
}

float compute_disparity(float2 point, float value_z, float3 normal_vector) {
    
//   float3 Norm_normal_vector = normalize(normal_vector);
//   printf("Nx = %f Ny = %f Nz = %f length = %f nNx = %f nNy = %f nNz = %f\n", normal_vector.x, normal_vector.y, normal_vector.z, length(normal_vector),  Norm_normal_vector.x, Norm_normal_vector.y, Norm_normal_vector.z);

  //return value_z;

  normal_vector = (float3)(0.0f, 0.0f, 1.0f);
  
  float plane_component_a = -normal_vector.x / normal_vector.z;
  float plane_component_b = -normal_vector.y / normal_vector.z;
  float plane_component_c =
      (normal_vector.x * point.x + normal_vector.y * point.y +
       normal_vector.z * value_z) /
      normal_vector.z;

  float disparity = plane_component_a * point.x + plane_component_b * point.y +
                    plane_component_c;

    return value_z;
/*   return disparity; */
}

//every value of the defines except for ALPHA is the value of the paper divided by 255.0f (normalization)

#define GAMMA 0.0392156862745098f 
#define ALPHA 0.9f
#define TAU_COL 0.0392156862745098f
#define TAU_GRAD 0.00784313725490196f

float cost_aggregation_SAD(__read_only image2d_t current_image_gray,
                       __read_only image2d_t adjunct_image_gray,
                       __read_only image2d_t gradient_image_left,
                       __read_only image2d_t gradient_image_right,
                       float4 plane_values,
                       float2 ref_center_pixel,
                       float2 search_center_pixel,
                       int image_width,
                       int image_height,
                       int radius,
                       int compute_right,
                       int slanted){

    float absolute_difference = 0.0f;

    float4 ref_center_pixel_value = read_imagef(current_image_gray, linear_sampler, ref_center_pixel );        //Color
    float3 ref_center_normal = (float3)(plane_values.y, plane_values.z, plane_values.w);
    float ref_center_z = plane_values.x;

    // printf("Nx = %f Ny = %f Nz = %f length = %f\n", ref_center_normal.x, ref_center_normal.y, ref_center_normal.z, length(ref_center_normal));
    // printf("Z = %f\n", ref_center_z);

    // normal vector's parameters
    float a = -ref_center_normal.x / ref_center_normal.z;
    float b = -ref_center_normal.y / ref_center_normal.z;
    float c = (ref_center_normal.x * ref_center_pixel.x + ref_center_normal.y * ref_center_pixel.y + ref_center_normal.z * ref_center_z) / ref_center_normal.z;

    for(int win_y_idx = -radius; win_y_idx <= radius; ++win_y_idx) {
        for(int win_x_idx = -radius; win_x_idx <= radius; ++win_x_idx) {

            float2 current_reference_pixel; //neighbor pixel
            current_reference_pixel.x = ref_center_pixel.x + win_x_idx;
            current_reference_pixel.y = ref_center_pixel.y + win_y_idx;

            float4 ref_pixel_value = read_imagef(current_image_gray, linear_sampler, current_reference_pixel  );

            //compute disparity for the neighbor pixel
            float d;
            if(slanted == 0) {
                d = ref_center_z;
            } else d = (a * current_reference_pixel.x) + (b * current_reference_pixel.y) + c;

            float2 current_search_pixel; // coressponding pixel of neighbor pixel - d
/*             current_search_pixel.x = search_center_pixel.x + win_x_idx;
            current_search_pixel.y = search_center_pixel.y + win_y_idx; */
            if(compute_right == 0) {
                current_search_pixel.x = current_reference_pixel.x - d;
                current_search_pixel.y = current_reference_pixel.y;
            } else {
                current_search_pixel.x = current_reference_pixel.x + d;
                current_search_pixel.y = current_reference_pixel.y;
            }


            float4 search_pixel_value = read_imagef(adjunct_image_gray, linear_sampler, current_search_pixel );

            //Euclidean Distance of center pixel and its neighbor in reference image (current)
            //float color_distance = distance(ref_pixel_value, ref_center_pixel_value);
            // float color_distance = max(max(fabs((float)ref_pixel_value.x - (float)ref_center_pixel_value.x), fabs((float)ref_pixel_value.y - (float)ref_center_pixel_value.y)), fabs((float)ref_pixel_value.z - (float)ref_center_pixel_value.z));
            
            float3 color_distance_rgb = fabs(ref_center_pixel_value.xyz - ref_pixel_value.xyz);

            float color_distance = 0.3333333f*(color_distance_rgb.x + color_distance_rgb.y + color_distance_rgb.z );


            //float color_distance = 255.0f*fast_distance(ref_center_pixel_value.xyz,ref_pixel_value.xyz);

            // w(p,q)
            float weight = exp(-color_distance/GAMMA);
            //weight = 1.0f;
            //Pixel_dissimilarity between the neighbor pixel and its matching pixel
            float3 color_diff_rgb = fabs(search_pixel_value.xyz - ref_pixel_value.xyz);
          
            float color_diff = 0.3333333f*(color_diff_rgb.x + color_diff_rgb.y + color_diff_rgb.z );
            // float color_diff = max(max(fabs((float)ref_pixel_value.x - (float)search_pixel_value.x), fabs((float)ref_pixel_value.y - (float)search_pixel_value.y)), fabs((float)ref_pixel_value.z - (float)search_pixel_value.z));
            
            float current_ref_gradient_value = ref_pixel_value.w;
            float current_search_gradient_value = search_pixel_value.w;
            float gradient_diff = fabs(current_ref_gradient_value - current_search_gradient_value);// * 255.0f;

            float pixel_dissimilarity = (1-ALPHA)*min(color_diff, TAU_COL) + ALPHA*min(gradient_diff, TAU_GRAD);

            //float pixel_dissimilarity = (1-1.0)*min(color_distance, TAU_COL) + 1.0*min(gradient_diff, TAU_GRAD);
            
            absolute_difference += weight*pixel_dissimilarity;
        }
    }

/*     int window_side_length = (radius * 2 + 1);

    absolute_difference /= (window_side_length * window_side_length); */

    return absolute_difference;

}


__kernel void convert_RGBA_to_R(__read_only image2d_t plane_image,
                                __write_only image2d_t disparity_image,
                                int image_width,
                                int image_height,
                                float max_disparity) {
  float2 pos = {get_global_id(0), get_global_id(1)};
  float4 input_value = read_imagef(plane_image, nearest_sampler, (float2)(pos.x, pos.y));

  float current_value_z = input_value.x;
  float3 current_normal_vector = {input_value.y, input_value.z, input_value.w};
  float output_disparity = compute_disparity(pos, current_value_z, current_normal_vector) ;

  barrier(CLK_GLOBAL_MEM_FENCE);
  write_imagef(disparity_image, (int2)(pos.x, pos.y), (float4)(output_disparity, 0.0f, 0.0f, 1.0f));

  // Channel Order : CL_R -> Components of channel data : (r, 0.0, 0.0, 1.0)
}

bool spatial_propagation_step(float2 current_pos,
                              float2 neigh_pos,
                              __read_only image2d_t in_plane_image,
                              __read_only image2d_t current_image_gray,
                              __read_only image2d_t adjunct_image_gray,
                              __read_only image2d_t gradient_image_left,
                              __read_only image2d_t gradient_image_right,                              
                              __write_only image2d_t out_plane_image,
                              float current_cost,
                              float4* in_out_best_plane,
                              int image_width,
                              int image_height,
                              int radius,
                              int compute_right,
                              int slanted) {

    //float4 current_values =  read_imagef(in_plane_image, nearest_sampler, (float2)(current_pos.x, current_pos.y));
    //float current_z = current_values.x;
    //float3 current_normal_vector = (current_values.yzw);

  float4 current_values = *in_out_best_plane;
  float current_z = current_values.x;
  float3 current_normal_vector = current_values.yzw;

  //float current_SAD = 0.0f;
  //float new_neigh_disparity = 0.0f;
  float neighbor_costs = 0.0f;
  //float4 plane_values = {0.0f, 0.0f, 0.0f, 0.0f};
  //float4 neigh_plane_values = {0.0f, 0.0f, 0.0f, 0.0f};
  //float3 current_normal_vector = {0.0f, 0.0f, 0.0f};
//   float3 neigh_normal_vector = {0.0f, 0.0f, 0.0f};

  //float current_z = 0.0f;
  //float neigh_z = 0.0f;

//   float best_disparity = 0.0f;
//   float best_z = 0.0f;
//   //float best_SAD = 0.0f;
//   float3 best_normal_vector = {0.0f, 0.0f, 0.0f};

  //best_SAD = current_SAD;
  float best_z = current_z;
  float3 best_normal_vector = current_normal_vector;

  float4 neigh_plane_values = read_imagef(in_plane_image, nearest_sampler, (float2)(neigh_pos.x, neigh_pos.y));
  float neigh_z = neigh_plane_values.x;
  float3 neigh_normal_vector = (float3)(neigh_plane_values.y, neigh_plane_values.z, neigh_plane_values.w);
  float new_neigh_disparity = compute_disparity(current_pos, neigh_z, neigh_normal_vector);

  /*     float3 Norm_normal_vector = normalize(neigh_normal_vector);
  printf("Nx = %f Ny = %f Nz = %f length = %f nNx = %f nNy = %f nNz = %f\n", neigh_normal_vector.x, neigh_normal_vector.y, neigh_normal_vector.z, length(neigh_normal_vector),  Norm_normal_vector.x, Norm_normal_vector.y, Norm_normal_vector.z);
 */
  if (compute_right == 0) {
    neighbor_costs = cost_aggregation_SAD(
        current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, neigh_plane_values, current_pos,
        (float2)(current_pos.x - new_neigh_disparity, current_pos.y),
        image_width, image_height, radius, compute_right, slanted);
  } else {
    neighbor_costs = cost_aggregation_SAD(
        adjunct_image_gray, current_image_gray, gradient_image_right, gradient_image_left, neigh_plane_values, current_pos,
        (float2)(current_pos.x + new_neigh_disparity, current_pos.y),
        image_width, image_height, radius, compute_right, slanted);
  }

  if (current_cost > neighbor_costs) {
    best_z = neigh_z;
    best_normal_vector = neigh_normal_vector;
    *in_out_best_plane = (float4)(best_z, best_normal_vector.x, best_normal_vector.y, best_normal_vector.z);
    return true;
  }
    return false;
}



__kernel void random_initialization(int image_width,
                                    int image_height,
                                    __write_only image2d_t out_plane_image,
                                    float max_disparity,
                                    float min_disparity) {
    
    float2 pos = {get_global_id(0), get_global_id(1)};

    int pixel_1d_index = pos.x + pos.y * image_width;  // "flatten" index 2D -> 1D
    int num_channels = 3;  // for element offset calculation
    int pixel_1d_offset = num_channels * pixel_1d_index;  // actual position of 1D pixel

    float random_inital_disparity_value = rand(pos) * (max_disparity - min_disparity) + min_disparity;

    float3 random_plane_vector = {0.0, 0.0, 0.0};
    random_plane_vector.x = rand_slanted_plane_value(pos + 0);
    random_plane_vector.y = rand_slanted_plane_value(pos + 1);
    random_plane_vector.z = rand_slanted_plane_value(pos + 2); 


    if(random_plane_vector.z < 0) {
        random_plane_vector.z *= -1.0;  
    }

    random_plane_vector = fast_normalize(random_plane_vector);

    //slanted planes
    write_imagef(out_plane_image, (int2)(pos.x, pos.y), (float4)(random_inital_disparity_value,
                random_plane_vector.x, random_plane_vector.y, random_plane_vector.z));

   //fronto-planar
   //write_imagef(out_plane_image, (int2)(pos.x, pos.y),
   //          (float4)(random_inital_disparity_value, 0.0, 0.0, 1.0));
}



__kernel void spatial_propagation(__read_only image2d_t current_image_gray,
                          __read_only image2d_t adjunct_image_gray,
                          __read_only image2d_t in_plane_image,
                          __read_only image2d_t gradient_image_left,
                          __read_only image2d_t gradient_image_right,                          
                          __write_only image2d_t out_plane_image,
                          int image_width,
                          int image_height,
                          int num_propagation_iterations,
                          int radius,
                          int compute_right,
                          int is_red, int slanted) { //is red = 1

    int2 global_idx_2d = {get_global_id(0), get_global_id(1)}; //checkerboard base position state 0
    float2 work_idx_2d =   {global_idx_2d.x, global_idx_2d.y}; //checkerboard base position state 1

    if(is_red == 1){
        if(global_idx_2d.x % 2 == 0){
            work_idx_2d.y = work_idx_2d.y * 2 + 1;
        }
        else{
            work_idx_2d.y = work_idx_2d.y * 2;
        }
    }
    else{
        if(global_idx_2d.x % 2 == 0){
            work_idx_2d.y = work_idx_2d.y * 2;
        }
        else{
            work_idx_2d.y = work_idx_2d.y * 2 + 1;
        }
    }

    float4 current_plane_values = read_imagef(in_plane_image, nearest_sampler, (float2)(work_idx_2d.x, work_idx_2d.y));
    float3 current_normal_vector = (float3)(current_plane_values.yzw); 
    float current_z = current_plane_values.x;
    float current_disparity = compute_disparity(work_idx_2d, current_z, current_normal_vector);

    float current_cost = 0.0;

    if (compute_right == 0) {
        current_cost = cost_aggregation_SAD(
            current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, current_plane_values, work_idx_2d,
            (float2)(work_idx_2d.x - current_disparity, work_idx_2d.y), image_width,
            image_height, radius, compute_right, slanted);
    } 
    else{
        current_cost = cost_aggregation_SAD(
            adjunct_image_gray, current_image_gray, gradient_image_right, gradient_image_left, current_plane_values, work_idx_2d,
            (float2)(work_idx_2d.x + current_disparity, work_idx_2d.y), image_width,
            image_height, radius, compute_right, slanted);
    }

  //float4 out_best_plane = (0.0, 0.0, 0.0, 0.0);

    float4 out_best_plane =  read_imagef(in_plane_image, nearest_sampler, (float2)(work_idx_2d.x, work_idx_2d.y));
    //float current_z = current_values.x;
    //float3 current_normal_vector = (current_values.yzw);

    //  bool was_better = false;

    for(int displacement = 1; displacement < 6; displacement += 4) {

        float2 left_neighbor_idx = {work_idx_2d.x - displacement, work_idx_2d.y};
        // spatial_prop_costs(work_idx_2d, current_cost, left_neighbor_idx, 
        //             in_plane_image, current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right,                              
        //             out_plane_image, image_width, image_height,radius,compute_right);
      spatial_propagation_step(work_idx_2d, left_neighbor_idx, in_plane_image,
                               current_image_gray, adjunct_image_gray,
                               gradient_image_left, gradient_image_right,
                               out_plane_image,
                               current_cost,
                               &out_best_plane,
                               image_width, image_height,
                               radius, compute_right, slanted);

      
        float2 top_neighbor_idx = {work_idx_2d.x, work_idx_2d.y + displacement};
        // spatial_prop_costs(work_idx_2d, current_cost, top_neighbor_idx, 
        //             in_plane_image, current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right,                              
        //             out_plane_image, image_width, image_height,radius,compute_right);
       spatial_propagation_step(work_idx_2d, top_neighbor_idx, in_plane_image,
                               current_image_gray, adjunct_image_gray,
                               gradient_image_left, gradient_image_right,
                               out_plane_image, 
                               current_cost,
                               &out_best_plane,
                               image_width, image_height,
                               radius, compute_right, slanted);
    
      
        float2 right_neighbor_idx = {work_idx_2d.x + displacement, work_idx_2d.y};
        // spatial_prop_costs(work_idx_2d, current_cost, right_neighbor_idx, 
        //             in_plane_image, current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right,                              
        //             out_plane_image, image_width, image_height,radius,compute_right);      
       spatial_propagation_step(work_idx_2d, right_neighbor_idx, in_plane_image,
                               current_image_gray, adjunct_image_gray,
                               gradient_image_left, gradient_image_right,
                               out_plane_image, 
                               current_cost,
                               &out_best_plane,
                               image_width, image_height,
                               radius, compute_right, slanted);
      
      
        float2 bottom_neighbor_idx = {work_idx_2d.x, work_idx_2d.y - displacement};
        // spatial_prop_costs(work_idx_2d, current_cost, bottom_neighbor_idx, 
        //             in_plane_image, current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right,                              
        //             out_plane_image, image_width, image_height,radius,compute_right); 
      spatial_propagation_step(work_idx_2d, bottom_neighbor_idx, in_plane_image,
                               current_image_gray, adjunct_image_gray,
                               gradient_image_left, gradient_image_right,
                               out_plane_image, 
                               current_cost,
                               &out_best_plane,
                               image_width, image_height,
                               radius, compute_right, slanted);
      
    }

//   float3 best_normal_vec = (float3)(out_best_plane.yzw);
//     float3 Norm_normal_vector = normalize(best_normal_vec);
//   printf("Nx = %f Ny = %f Nz = %f length = %f nNx = %f nNy = %f nNz = %f\n", best_normal_vec.x, best_normal_vec.y, best_normal_vec.z, length(best_normal_vec),  Norm_normal_vector.x, Norm_normal_vector.y, Norm_normal_vector.z);

    write_imagef(out_plane_image, (int2)(work_idx_2d.x, work_idx_2d.y), out_best_plane);

}



void view_compare(float2 current_pos,
                __read_only image2d_t in_plane_left,
                __read_only image2d_t in_plane_right,
                __write_only image2d_t out_result_plane,
                __read_only image2d_t adjunct_image_gray,
                __read_only image2d_t current_image_gray,
                __read_only image2d_t gradient_image_left,
                __read_only image2d_t gradient_image_right,                
                int image_width,
                int image_height,
                float max_disparity,
                int radius,
                int compute_right, int slanted
            ) {

    
    float2 matching_pos = {0.0f, 0.0f};

    float4 current_plane_values = {0.0, 0.0, 0.0, 1.0};

    if(0 == compute_right) {
       current_plane_values = read_imagef(in_plane_left, nearest_sampler, current_pos);
    } else {
       current_plane_values = read_imagef(in_plane_right, nearest_sampler, current_pos);
    }
    float current_z = current_plane_values.x;
    float3 current_normal_vector = {current_plane_values.y, current_plane_values.z, current_plane_values.w};
    float current_disparity = compute_disparity(current_pos, current_z, current_normal_vector);
    
    float best_cost = 0.0f;
    float best_z = current_z;
    float3 best_normal_vector = current_normal_vector;

    matching_pos.x = current_pos.x;
    matching_pos.y = current_pos.y;
    float matching_pos_cost = 0.0f;

    if(0 == compute_right) {
        best_cost = cost_aggregation_SAD(current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, current_plane_values,
                                         current_pos, (float2)(current_pos.x - current_disparity, current_pos.y), image_width, image_height, radius, 0, slanted);
    }
    else {
        best_cost = cost_aggregation_SAD(adjunct_image_gray, current_image_gray, gradient_image_left, gradient_image_right, current_plane_values,
                                         (float2)(current_pos.x + current_disparity, current_pos.y), current_pos, image_width, image_height, radius, 1, slanted);
    }

    if(0 == compute_right) {
        for(matching_pos.x = current_pos.x; matching_pos.x >= current_pos.x - max_disparity; matching_pos.x -= 1.0) {

            float4 matching_pos_value = read_imagef(in_plane_right, nearest_sampler, (float2)(matching_pos.x, matching_pos.y));
            float matching_pos_z = matching_pos_value.x;
            float3 matching_pos_normal_vector = {matching_pos_value.y, matching_pos_value.z, matching_pos_value.w};
            float matching_pos_disparity = compute_disparity(matching_pos, matching_pos_z, matching_pos_normal_vector);
            float matching_x_value = 0.0f;

            if(0 == compute_right) {
                matching_x_value = matching_pos.x + matching_pos_disparity;
            }
           
            
            if( fabs(matching_x_value - current_pos.x) < 1.0 ) {
              

            //if( (int) ( (matching_x_value) ) == (int)((current_pos.x)) ) {
                if(0 == compute_right) {
                    matching_pos_cost = cost_aggregation_SAD(current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, matching_pos_value,
                                                             current_pos, (float2)(current_pos.x - matching_pos_disparity, current_pos.y), image_width, image_height, radius, 0, slanted);                
                }
              
                if(best_cost > matching_pos_cost) {
                  if(current_pos.x == 100 && current_pos.y == 100) {
                      //printf("MP: %.4f\n", matching_pos.x);
                      //printf("Updating cost to %.4f and z: %.4f\n", best_cost, best_z);
                    }     
                    best_z = matching_pos_z;
                    best_normal_vector = matching_pos_normal_vector;
                    best_cost = matching_pos_cost; // update best costs
                }


            }
        }
    } else {
        for(matching_pos.x = current_pos.x; matching_pos.x <= current_pos.x + max_disparity; matching_pos.x += 1.0) {

            float4 matching_pos_value = read_imagef(in_plane_left, nearest_sampler, (float2)(matching_pos.x, matching_pos.y));
            float matching_pos_z = matching_pos_value.x;
            float3 matching_pos_normal_vector = {matching_pos_value.y, matching_pos_value.z, matching_pos_value.w};
            float matching_pos_disparity = compute_disparity(matching_pos, matching_pos_z, matching_pos_normal_vector);
            float matching_x_value = 0.0f;

            if(1 == compute_right)  {
                matching_x_value = matching_pos.x - matching_pos_disparity;
            }
            
            if( fabs(matching_x_value - current_pos.x) < 1.0 ) {
                if(1 == compute_right) {
                    matching_pos_cost = cost_aggregation_SAD(adjunct_image_gray, current_image_gray, gradient_image_right, gradient_image_left, matching_pos_value,
                                                             current_pos, (float2)(current_pos.x + matching_pos_disparity, current_pos.y), image_width, image_height, radius, 1, slanted);                
                }
                
                if(best_cost > matching_pos_cost) {
                    best_z = matching_pos_z;
                    best_normal_vector = matching_pos_normal_vector;
                    best_cost = matching_pos_cost; // update best costs
                }


            } //else {
              //  best_z = 0;
              //  best_normal_vector = (0.0, 0.0, 1.0);

            //}
        }
    }
    
    write_imagef(out_result_plane, (int2)(current_pos.x, current_pos.y), (float4)(best_z, best_normal_vector.x, best_normal_vector.y, best_normal_vector.z));
    // write_imagef(out_result_plane, (int2)(current_pos.x, current_pos.y), (float4)(0.0, 0.0, 0.0, 1.0));
    
}


__kernel void view_propagation(
                __read_only image2d_t in_plane_left,
                __read_only image2d_t in_plane_right,
                __write_only image2d_t out_plane_left,
                __write_only image2d_t out_plane_right,
                __read_only image2d_t adjunct_image_gray,
                __read_only image2d_t current_image_gray,
                __read_only image2d_t gradient_image_left,
                __read_only image2d_t gradient_image_right,                
                int image_width,
                int image_height,
                float max_disparity,
                int radius, int slanted
                ) {

    float2 pos = {get_global_id(0), get_global_id(1)};

//#if 1   
    view_compare(pos, in_plane_left, in_plane_right, out_plane_left, adjunct_image_gray, current_image_gray, gradient_image_left, gradient_image_right,  image_width, image_height, max_disparity, radius, 0, slanted);
    view_compare(pos, in_plane_left, in_plane_right, out_plane_right, adjunct_image_gray, current_image_gray, gradient_image_left, gradient_image_right, image_width, image_height, max_disparity, radius, 1, slanted);
  

// #else 
//     float4 current_plane_left = read_imagef(in_plane_left, nearest_sampler, (int2)(pos.x, pos.y));
//     float4 current_plane_right = read_imagef(in_plane_right, nearest_sampler, (int2)(pos.x, pos.y));

//     write_imagef(out_plane_left, (int2)(pos.x, pos.y), current_plane_left);
//     write_imagef(out_plane_right, (int2)(pos.x, pos.y), current_plane_right); 
// #endif
}


__kernel void plane_refinement(__read_only image2d_t current_image_gray,
                               __read_only image2d_t adjunct_image_gray,
                               __read_only image2d_t in_plane_image,
                               __write_only image2d_t out_plane_image,
                               __read_only image2d_t gradient_image_left,
                               __read_only image2d_t gradient_image_right,                               
                               int image_width,
                               int image_height,
                               float max_disparity,
                               int radius,
                               int compute_right,
                               int is_red,
                               int propagation_idx,
                               float steps,
                               int slanted) {
  float2 pos = {get_global_id(0), get_global_id(1)};
  int2 global_idx_2d = {get_global_id(0), get_global_id(1)}; //checkerboard base position state 01

 if(is_red == 1){
    if(global_idx_2d.x % 2 == 0){
        pos.y = pos.y * 2 + 1;
    }
    else{
        pos.y = pos.y * 2;
    }
  }
  else{
    if(global_idx_2d.x % 2 == 0){
        pos.y = pos.y * 2;
    }
    else{
        pos.y = pos.y * 2 + 1;
    }
  }
  
  float4 current_plane_values = read_imagef(in_plane_image, nearest_sampler, (float2)(pos.x, pos.y));
  float current_value_z = current_plane_values.x;
  float3 current_normal_vector = {current_plane_values.y, current_plane_values.z, current_plane_values.w};

//   float3 Norm_normal_vector = normalize(current_normal_vector);
//   printf("Nx = %f Ny = %f Nz = %f length = %f nNx = %f nNy = %f nNz = %f\n", current_normal_vector.x, current_normal_vector.y, current_normal_vector.z, length(current_normal_vector),  Norm_normal_vector.x, Norm_normal_vector.y, Norm_normal_vector.z);
  
  float max_change_n = 1.0f;
  float new_z = 0.0;
  float3 new_normal_vector = {0.0f, 0.0f, 1.0f};
  float new_costs = 0.0f;


  float current_disparity = compute_disparity(pos, current_value_z, current_normal_vector);

  float current_costs = 0.0;
  if(compute_right == 0) {
    current_costs = cost_aggregation_SAD(current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, current_plane_values, pos,
                                      (float2)(pos.x - current_disparity, pos.y),
                                      image_width, image_height, radius, compute_right, slanted);
  } else {
    current_costs = cost_aggregation_SAD(adjunct_image_gray, current_image_gray, gradient_image_right, gradient_image_left, current_plane_values, pos,
                                      (float2)(pos.x + current_disparity, pos.y),
                                      image_width, image_height, radius, compute_right, slanted);    
  }



  for(float max_change_z = max_disparity / 2.0; max_change_z >= 0.1f; max_change_z /= steps) {
    float delta_z = rand_slanted_plane_value(pos) * max_change_z;
    new_z = (current_value_z + delta_z);

    /*
    if(new_z < 0 || new_z > max_disparity) {
        continue;
    } */
    new_z = clamp(new_z, 0.0f, max_disparity);

    float3 delta_normal_vector = (rand_slanted_plane_value(max_change_n * (pos + 4354.0f)), rand_slanted_plane_value(max_change_n * (pos + 768354.0f)), rand_slanted_plane_value(max_change_n * (pos + 872853.0f))) * max_change_n;



    float3 unnormalized_normal_vector =
        ((delta_normal_vector) +  (current_normal_vector) );

    if(unnormalized_normal_vector.z < 0) {
        unnormalized_normal_vector.z = 0;
    }
    new_normal_vector = fast_normalize(unnormalized_normal_vector);

    //float current_disparity = compute_disparity(pos, current_value_z, current_normal_vector);
    float new_disparity = compute_disparity(pos, new_z, new_normal_vector);
    float4 new_plane_values = (float4)(new_z, new_normal_vector.x, new_normal_vector.y, new_normal_vector.z);
    // float new_disparity = compute_disparity(pos, new_z,
    // current_normal_vector);

    if (compute_right == 0) {
      new_costs = cost_aggregation_SAD(current_image_gray, adjunct_image_gray, gradient_image_left, gradient_image_right, new_plane_values, pos,
                               (float2)(pos.x - new_disparity, pos.y),
                               image_width, image_height, radius, 0, slanted);
    } else {
      new_costs = cost_aggregation_SAD(adjunct_image_gray, current_image_gray, gradient_image_right, gradient_image_left, new_plane_values, pos,
                               (float2)(pos.x + new_disparity, pos.y),
                               image_width, image_height, radius, 1, slanted);
    }


    
    if (new_costs < current_costs) {
      current_value_z = new_z;
      current_normal_vector = new_normal_vector;
      current_costs = new_costs;
    }
    
    

    max_change_n /= 2.0f;
  }

    // float3 Norm_normal_vector = normalize(current_normal_vector);
//   printf("Nx = %f Ny = %f Nz = %f length = %f nNx = %f nNy = %f nNz = %f\n", current_normal_vector.x, current_normal_vector.y, current_normal_vector.z, length(current_normal_vector),  Norm_normal_vector.x, Norm_normal_vector.y, Norm_normal_vector.z);

  write_imagef(out_plane_image, (int2)(pos.x, pos.y),
               (float4)(current_value_z, current_normal_vector.x,
                        current_normal_vector.y, current_normal_vector.z));
}

__kernel void temporal_propagation( __read_only image2d_t left_image,
                                    __read_only image2d_t right_image,
                                    __read_only image2d_t current_plane_left,
                                    __read_only image2d_t current_plane_right,
                                    __read_only image2d_t last_plane_left,
                                    __read_only image2d_t last_plane_right,
                                    __read_only image2d_t gradient_image_left,
                                    __read_only image2d_t gradient_image_right,                                    
                                    __write_only image2d_t output_plane_left,
                                    __write_only image2d_t output_plane_right,
                                    int image_width,
                                    int image_height,
                                    int radius , int is_red, int slanted) {

    float2 pos = {get_global_id(0), get_global_id(1)};

    int2 global_idx_2d = {get_global_id(0), get_global_id(1)}; //checkerboard base position state 0
    //float2 work_idx_2d =   {global_idx_2d.x, global_idx_2d.y}; //checkerboard base position state 1

    if(is_red == 1){
        if(global_idx_2d.x % 2 == 0){
            pos.y = pos.y * 2 + 1;
        }
        else{
        pos.y = pos.y * 2;
        }
    }
    else{
        if(global_idx_2d.x % 2 == 0){
            pos.y = pos.y * 2;
        }
        else{
            pos.y = pos.y * 2 + 1;
        }
    }


    // TEMPORAL FOR LEFT IMAGE
    float4 current_value_left = read_imagef(current_plane_left, nearest_sampler, (float2)(pos.x,pos.y));    
    float current_value_z_left = current_value_left.x;
    float3 current_normal_vector_left = {current_value_left.y , current_value_left.z, current_value_left.w};
    float current_disparity_left = compute_disparity(pos, current_value_z_left, current_normal_vector_left);

    float current_SAD_left = cost_aggregation_SAD(left_image, right_image, gradient_image_left, gradient_image_right, current_value_left, pos, 
                                       (float2)(pos.x - current_disparity_left , pos.y), 
                                       image_width, image_height, radius, 0, slanted);

    float4 last_value_left = read_imagef(last_plane_left, nearest_sampler, (float2)(pos.x,pos.y));    
    float last_value_z_left = last_value_left.x;
    float3 last_normal_vector_left = {last_value_left.y , last_value_left.z, last_value_left.w};
    float last_disparity_left = compute_disparity(pos, last_value_z_left, last_normal_vector_left);

    float last_SAD_left = cost_aggregation_SAD(left_image, right_image, gradient_image_left, gradient_image_right, last_value_left, pos, 
                                       (float2)(pos.x - last_disparity_left , pos.y), 
                                       image_width, image_height, radius, 0, slanted);


    if(last_SAD_left < current_SAD_left)
    {
        current_value_z_left = last_value_z_left;
        current_normal_vector_left = last_normal_vector_left;
    }
    write_imagef(output_plane_left, (int2)(pos.x, pos.y),
                (float4)(current_value_z_left, current_normal_vector_left.x,
                current_normal_vector_left.y, current_normal_vector_left.z));

    // TEMPORAL FOR RIGHT IMAGE
    float4 current_value_right = read_imagef(current_plane_right, nearest_sampler, (float2)(pos.x,pos.y));
    float current_value_z_right = current_value_right.x;
    float3 current_normal_vector_right = {current_value_right.y , current_value_right.z, current_value_right.w};
    float current_disparity_right = compute_disparity(pos, current_value_z_right, current_normal_vector_right);

    float current_SAD_right = cost_aggregation_SAD(right_image, left_image, gradient_image_right, gradient_image_left, current_value_right, pos, 
                                       (float2)(pos.x + current_disparity_right , pos.y), 
                                       image_width, image_height, radius, 1, slanted);

    float4 last_value_right = read_imagef(last_plane_right, nearest_sampler, (float2)(pos.x,pos.y));
    float last_value_z_right = last_value_right.x;
    float3 last_normal_vector_right = {last_value_right.y , last_value_right.z, last_value_right.w};
    float last_disparity_right = compute_disparity(pos, last_value_z_right, last_normal_vector_right);

    float last_SAD_right = cost_aggregation_SAD(right_image, left_image, gradient_image_right, gradient_image_left, last_value_right, pos, 
                                       (float2)(pos.x + last_disparity_right , pos.y), 
                                       image_width, image_height, radius, 1, slanted);

    if(last_SAD_right < current_SAD_right)
    {
        current_value_z_right = last_value_z_right;
        current_normal_vector_right = last_normal_vector_right;
    }
    write_imagef(output_plane_right, (int2)(pos.x, pos.y),
                (float4)(current_value_z_right, current_normal_vector_right.x,
                current_normal_vector_right.y, current_normal_vector_right.z));
}


#define OCCLUDED_REGION 1
#define MISMATCH_REGION 2

__kernel void outlier_detection_after_converting_RGBA_in_R(
                                __read_only image2d_t disparity_left,
                                __read_only image2d_t disparity_right,
                                __write_only image2d_t disparity_image,
                                __write_only image2d_t outlier_mask,
                                float dMin,
                                float dMax) {
  float2 pos = {get_global_id(0), get_global_id(1)};

  float tolerance = 5.0f;

  float left_disp = read_imagef(disparity_left, nearest_sampler, (float2)(pos.x, pos.y)).x ;


  float2 new_pos = {pos.x - left_disp, pos.y};
  float right_disp = read_imagef(disparity_right, nearest_sampler, round(new_pos)).x ;
  
  float outlier_mask_value = 0.0;

 if( /*(pos.x - (left_disp) < 0.0f) ||  */fabs((left_disp) - (right_disp)) > tolerance ) { 
//if( (left_disp) > 50.0f) {  

      
      bool occlusion = true; 
      
      for(float d = dMin; (d < dMax) && occlusion; d += 1.0f) {

          //right_disparity = read_imagef(disparity_right, nearest_sampler,(float2)(pos.x - d, pos.y)).x; 

          new_pos = (pos.x - d, pos.y);
          right_disp = /*255.0f * */ read_imagef(disparity_right, nearest_sampler, new_pos).x;
          
          if( pos.x - d >= 0 &&  fabs(d - right_disp) < tolerance){
              occlusion = false;
          }
      }

      if(occlusion) {
          left_disp = 0.0f; // TODO: marker region for later stages
          outlier_mask_value = OCCLUDED_REGION;
          write_imagef(outlier_mask, (int2)(pos.x, pos.y),(float4)(OCCLUDED_REGION, 0.0f, 0.0f, 1.0f));
      } else {
          left_disp = 0.0f; // TODO: marker region differently for later stages
          outlier_mask_value = MISMATCH_REGION;
          write_imagef(outlier_mask, (int2)(pos.x, pos.y),(float4)(MISMATCH_REGION, 0.0f, 0.0f, 1.0f));
      }

    

     //left_disp = 0.0f;
  } //else {
    //left_disp = 60.0f;
  //}

  write_imagef(disparity_image, (int2)(pos.x, pos.y),(float4)(left_disp, 0.0f, 0.0f, 1.0f));

  //write_imagef(outlier_mask, (int2)(pos.x, pos.y),(float4)(outlier_mask_value, 0.0f, 0.0f, 1.0f));





  //////
/*
  float left_disparity = read_imagef(disparity_left, nearest_sampler, (float2)(pos.x, pos.y)).x;

  float2 new_pos_test = {pos.x - left_disparity, pos.y};
  float right_disp_test = read_imagef(disparity_right, nearest_sampler, (new_pos_test)).x;
  float2 new_new_pos = {new_pos_test.x + right_disp_test, pos.y};

  float recovered_original_disparity = read_imagef(disparity_left, nearest_sampler, new_new_pos).x;

  write_imagef(disparity_image, (int2)(pos.x, pos.y),(float4)(recovered_original_disparity , 0.0f, 0.0f, 1.0f));
*/
}

#define INVALID 1.0f
#define VALID 0.0f
//post processing
__kernel void outlier_detection(__read_only image2d_t disparity_left,
                                __read_only image2d_t disparity_right,
                                __write_only image2d_t disparity_image,
                                __write_only image2d_t outlier_mask,
                                float dMin,
                                float dMax,
                                int image_width,
                                int image_height) {
    
    float2 pos = {get_global_id(0), get_global_id(1)};

    float4 left_value = read_imagef(disparity_left, nearest_sampler, (float2)(pos.x, pos.y));
    float left_value_z = left_value.x;
    float3 left_normal_vector = {left_value.y , left_value.z, left_value.w};
    float left_disp = compute_disparity(pos, left_value_z, left_normal_vector);

    float2 corresponding_pos = {pos.x - left_disp, pos.y};
    float4 right_value = read_imagef(disparity_right, nearest_sampler, (float2)(corresponding_pos.x, corresponding_pos.y));
    float right_value_z = right_value.x;
    float3 right_normal_vector = {right_value.y , right_value.z, right_value.w};
    float right_disp = compute_disparity(corresponding_pos, right_value_z, right_normal_vector);

    float outlier_mask_value = 0.0f;
  
    float tolerance = 1.0f;
    float pixel_validation = VALID;

    // if(fabs(fabs(left_value_z) - fabs(right_value_z)) > tolerance ) {  
    if(fabs(left_disp - right_disp) > tolerance ) {   
        //bool invalid_pixel = true;
        pixel_validation = INVALID;
    }
    write_imagef(outlier_mask, (int2)(pos.x, pos.y),(float4)(pixel_validation, 0.0f, 0.0f, 1.0f));
    //write_imagef(disparity_image, (int2)(pos.x, pos.y),(float4)(left_value_z, left_normal_vector.x, left_normal_vector.y, left_normal_vector.z));
}

__kernel void fill_invalid_pixel(__read_only image2d_t disparity_left,
                                __read_only image2d_t disparity_right,
                                __write_only image2d_t disparity_image,
                                __read_only image2d_t outlier_mask,
                                //float dMin,
                                //float dMax,
                                int image_width,
                                int image_height) {

    float2 pos = {(float)get_global_id(0), (float)get_global_id(1)};
    
   // float2 local_pos = {get_local_id(0), get_local_id(1)};
   // __local float band[]16[16*29]; //ca 450 , image width

    float4 current_outlier_value = read_imagef(outlier_mask, nearest_sampler, (float2)(pos.x, pos.y));

    float4 left_value = read_imagef(disparity_left, nearest_sampler, (float2)(pos.x, pos.y));
    float fill_value_z = left_value.x;
    float3 fill_normal_vector = {left_value.y , left_value.z, left_value.w};
    float lowest_disp = compute_disparity(pos, fill_value_z, fill_normal_vector);
    
    float2 left_neigh_pos = {pos.x - 1, pos.y};
    float2 right_neigh_pos = {pos.x + 1, pos.y};
    float2 down_neigh_pos = {pos.x, pos.y - 1};
    float2 up_neigh_pos = {pos.x, pos.y + 1};

    if(current_outlier_value.x == INVALID) {   
        //searching next valid neighbor pixel in outlier_mask (READ)
         for(int horizontal_line_left = pos.x - 1 ; horizontal_line_left >=  /* 0 */ pos.x - 15; horizontal_line_left -= 1){
            float4 left_neigh_valid = read_imagef(outlier_mask, nearest_sampler, (float2)(horizontal_line_left, pos.y));

            if(left_neigh_valid.x == VALID){
                left_neigh_pos = (float2)(horizontal_line_left, pos.y);
                break;
            }
        }

        for(int horizontal_line_right = pos.x + 1; horizontal_line_right < /* image_width */ pos.x + 15; horizontal_line_right += 1){
            float4 right_neigh_valid = read_imagef(outlier_mask, nearest_sampler, (float2)(horizontal_line_right, pos.y));

            if(right_neigh_valid.x == VALID){
                right_neigh_pos = (float2)(horizontal_line_right, pos.y);
                break;
            }
        } 

        //  for(int vertical_line_up = pos.y + 1 ; vertical_line_up < image_height /* pos.y + 2 */; vertical_line_up += 1){
        //     float4 up_neigh_valid = read_imagef(outlier_mask, nearest_sampler, (float2)(pos.x, vertical_line_up));

        //     if(up_neigh_valid.x == VALID){
        //         up_neigh_pos = (float2)(pos.x , vertical_line_up);
        //         break;
        //     }
        // } 
        // for(int vertical_line_down = pos.y - 1 ; vertical_line_down >= 0 /* pos.y - 2 */; vertical_line_down -= 1){
        //     float4 down_neigh_valid = read_imagef(outlier_mask, nearest_sampler, (float2)(pos.x, vertical_line_down));

        //     if(down_neigh_valid.x == VALID){
        //         down_neigh_pos = (float2)(pos.x , vertical_line_down);
        //         break;
        //     }
        // } 
        
        float4 left_neigh_value = read_imagef(disparity_left, nearest_sampler, (float2)(left_neigh_pos.x, left_neigh_pos.y));
        float left_neigh_value_z = left_neigh_value.x;
        float3 left_neigh_normal_vector = {left_neigh_value.y , left_neigh_value.z, left_neigh_value.w};
        float left_neigh_disp = compute_disparity(pos, left_neigh_value_z, left_neigh_normal_vector);
        
        float4 right_neigh_value = read_imagef(disparity_left, nearest_sampler, (float2)(right_neigh_pos.x, right_neigh_pos.y));
        float right_neigh_value_z = right_neigh_value.x;
        float3 right_neigh_normal_vector = {right_neigh_value.y , right_neigh_value.z, right_neigh_value.w};
        float right_neigh_disp = compute_disparity(pos, right_neigh_value_z, right_neigh_normal_vector);

/*         float4 up_neigh_value = read_imagef(disparity_left, nearest_sampler, (float2)(up_neigh_pos.x, up_neigh_pos.y));
        float up_neigh_value_z = up_neigh_value.x;
        float3 up_neigh_normal_vector = {up_neigh_value.y , up_neigh_value.z, up_neigh_value.w};
        float up_neigh_disp = compute_disparity(pos, up_neigh_value_z, up_neigh_normal_vector);

        float4 down_neigh_value = read_imagef(disparity_left, nearest_sampler, (float2)(down_neigh_pos.x, down_neigh_pos.y));
        float down_neigh_value_z = down_neigh_value.x;
        float3 down_neigh_normal_vector = {down_neigh_value.y , down_neigh_value.z, down_neigh_value.w};
        float down_neigh_disp = compute_disparity(pos, down_neigh_value_z, down_neigh_normal_vector);
 */

        lowest_disp = left_neigh_disp;
        fill_value_z = left_neigh_value_z;
        fill_normal_vector = left_neigh_normal_vector;
       
        if(lowest_disp > right_neigh_disp) {
            lowest_disp = right_neigh_disp;
            fill_value_z = right_neigh_value_z;
            fill_normal_vector = right_neigh_normal_vector;
        }


/*     
        if(up_neigh_disp < down_neigh_disp){
            down_neigh_disp = up_neigh_disp;
            down_neigh_value_z = up_neigh_value_z;
            down_neigh_normal_vector = up_neigh_normal_vector;
        }
        if(down_neigh_disp < lowest_disp){
            lowest_disp = down_neigh_disp;
            fill_value_z = down_neigh_value_z;
            fill_normal_vector = down_neigh_normal_vector;
        } */

        //Testing
        // left_value_z = 0.0f;
        

        
    } 

    //work_group_barrier(CLK_LOCAL_MEM_FENCE);

    write_imagef(disparity_image, (int2)(pos.x, pos.y),(float4)(fill_value_z, fill_normal_vector.x, fill_normal_vector.y, fill_normal_vector.z));
}

float colorDiff(float4 p1, float4 p2) 
{
    // return biggest channel difference
    return max(max(fabs((float)p1.x - (float)p2.x), fabs((float)p1.y - (float)p2.y)), fabs((float)p1.z - (float)p2.z));
}

#define tau1 20
#define tau2 6
#define L1 32
#define L2 17
#define votingThreshold 20
#define votingRatioThreshold 0.4

__kernel void compute_limits(__read_only image2d_t input_image,
                             __write_only image2d_t limits_image,
                             int image_width,
                             int image_height)
{
  
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float4 p = read_imagef(input_image, nearest_sampler, (int2)(x, y));

    // compute limits for Up, Down, Left, Right arm in this order
    char directionH[4] = {-1, 1, 0, 0};
    char directionW[4] = {0, 0, -1, 1};
    int dist[4];
    for(char direction = 0; direction < 4; direction++) {
        float4 p2 = p;   
        int d = 1;
        int h1 = y + directionH[direction];
        int w1 = x + directionW[direction];
        //int imgWidth = get_image_width(input_image);
        //int imgHeight = get_image_height(input_image);

        bool inside = (0 <= h1) && (h1 < image_height) && (0 <= w1) && (w1 < image_width);
        if(inside) {
            bool colorCond = true; bool wLimitCond = true; bool fColorCond = true;
            while(colorCond && wLimitCond && fColorCond && inside) {
                float4 p1 = read_imagef(input_image, nearest_sampler, (int2)(w1, h1));
                // check if color similar enough
                colorCond = colorDiff(p, p1) < tau1 && colorDiff(p1, p2) < tau1;
                // check if we exceed the length
                wLimitCond = d < L1;
                // check for color similarities of further away neighbors
                fColorCond = (d <= L2) || (d > L2 && colorDiff(p, p1) < tau2);
                p2 = p1; h1 += directionH[direction]; w1 += directionW[direction];
                // check if we are still inside the image
                inside = (0 <= h1) && (h1 < image_height) && (0 <= w1) && (w1 < image_width);
                d++;
            }
            d--;
        }
        dist[direction] = d - 1;
    }

    write_imagef(limits_image, (int2)(x, y), (float4)(dist[0], dist[1], dist[2], dist[3])); 

}





/*

#define OCCLUDED_REGION 1
#define MISMATCH_REGION 2

__kernel void outlier_detection(__read_only image2d_t disparity_left,
                                __read_only image2d_t disparity_right,
                                __write_only image2d_t disparity_image,
                                __write_only image2d_t outlier_mask,
                                const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);
    int tolerance = 0;

    uint disparity = read_imagef(disparity_left, nearest_sampler, (int2)(x, y)).x;
    uint right_disparity = read_imagef(disparity_right, nearest_sampler, (int2)(x - disparity, y)).x;
    // initialize all outlier_mask values with 0 (i hate OpenCL... no zero initialization for device mem)
    //write_imagef(outlier_mask, (int2)(x, y), (uint4)(0, 0, 0, 0)); 
    if(x - ((int)disparity) < 0 || abs(((int)disparity) - ((int)right_disparity)) > tolerance) {
        bool occlusion = true;
        for(int d = dMin; d < dMax && occlusion; d++) {

            right_disparity = read_imagef(disparity_right, nearest_sampler, (int2)(x - d, y)).x;
            if(x - d >= 0 && d == (int)right_disparity) {
                occlusion = false;
            }
        }
        if(occlusion) {
            disparity = 0; // TODO: marker region for later stages
            write_imagef(outlier_mask, (int2)(x, y), (uint4)(OCCLUDED_REGION, 0, 0, 0)); 
        }
        else {
            disparity = 0; // TODO: marker region differently for later stages
            write_imagef(outlier_mask, (int2)(x, y), (uint4)(MISMATCH_REGION, 0, 0, 0)); 
        }
    }

    write_imagef(disparity_image, (int2)(x, y), (uint4)(disparity, 0, 0, 0)); 
}

*/


//will be added later in the compiling time
#define NUM_DISPARITIES 255

__kernel void region_voting(__read_only image2d_t disparity_src,
                            __read_only image2d_t outlier_mask_src,
			                      __read_only image2d_t limits_image,
                            __write_only image2d_t disparity_target,
                            __write_only image2d_t outlier_mask_target,
                            const int dMin,
                            const int dMax,
			                      const int horizontal)
{
  
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uchar bins[NUM_DISPARITIES] = {0};

    float outlier = read_imagef(outlier_mask_src, nearest_sampler, (int2)(x, y)).x;
    // copy the outlier mask 1 to 1
    write_imagef(outlier_mask_target, (int2)(x, y), (float4)(outlier, 0, 0, 0));

    if(outlier == 0) {
    	float disparity = read_imagef(disparity_src, nearest_sampler, (int2)(x, y)).x ;
	    write_imagef(disparity_target, (int2)(x, y), (float4)(disparity, 0, 0, 0));
	    return;
    }

    float4 limits = read_imagef(limits_image, nearest_sampler, (int2)(x, y));
    int oLA, oLB, iLA, iLB;
    if(horizontal == 1) { 
      oLA = -((int)limits.x);
      oLB = ((int)limits.y);
    }
    else {
      oLA = -((int)limits.z);
      oLB = ((int)limits.w);
    }

    // voting stage

    int vote = 0;
    for(int outer = oLA; outer <= oLB; outer++) {
    	if(horizontal == 1) {
            float4 inner_limits = read_imagef(limits_image, nearest_sampler, (int2)(x, y + outer));
            iLA = -((int)inner_limits.z);
            iLB = (int)inner_limits.w;
	}
        else {
            float4 inner_limits = read_imagef(limits_image, nearest_sampler, (int2)(x + outer, y));
            iLA = -((int)inner_limits.x);
            iLB = (int)inner_limits.y;
        }
        
        for(int inner = iLA; inner <= iLB; inner++) {
            int height, width;
            if(horizontal == 1) {
                height = y + outer;
                width = x + inner;
            }
            else {
                height = y + inner;
                width = x + outer;
            }
            
            outlier = read_imagef(outlier_mask_src, nearest_sampler, (int2)(width, height)).x;
            if(outlier == 0.0f) { // we dont count outliers (1 and 2)
                vote += 1;
    	        float disparity = read_imagef(disparity_src, nearest_sampler, (int2)(width, height)).x ;
                bins[(int)disparity] += 1;
            }
        }
    }    

    // thresholding
    float disparity = read_imagef(disparity_src, nearest_sampler, (int2)(x, y)).x ;
    if(vote <= votingThreshold) {
	write_imagef(disparity_target, (int2)(x, y), (float4)(disparity, 0, 0, 0));
        // delete outlier ?
        if(disparity != 0)
            write_imagef(outlier_mask_target, (int2)(x, y), (float4)(0, 0, 0, 0));
    }
    else {
        float voteRatio;
        float voteRatioMax = 0.0;
        for(int d = dMin; d <= dMax; d++) {
            voteRatio = bins[d - dMin] / (float)vote;
            if(voteRatio > voteRatioMax) {
                voteRatioMax = voteRatio;
                if(voteRatioMax > votingRatioThreshold) {
                    disparity = d;
                }
            }
            bins[d - dMin] = 0;
        }
	write_imagef(disparity_target, (int2)(x, y), (float4)(disparity, 0, 0, 0));
        // delete outlier
        if(disparity != 0)
            write_imagef(outlier_mask_target, (int2)(x, y), (float4)(0, 0, 0, 0));
    }

}

__kernel void copy_plane_image_to_last_plane(__read_only image2d_t final_plane_left,
                                                __read_only image2d_t final_plane_right,
                                                __write_only image2d_t last_plane_left,
                                                __write_only image2d_t last_plane_right) {
    
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float4 value_left = read_imagef(final_plane_left, nearest_sampler, (float2)(x, y));
    float4 value_right = read_imagef(final_plane_right, nearest_sampler, (float2)(x, y));

    write_imagef(last_plane_left, (int2)(x, y), value_left);
    write_imagef(last_plane_right, (int2)(x, y), value_right);
}

// use Sobel_Filter to create Gradient_images from input images
__kernel void gradient_filter(read_only image2d_t input_image,
                              __write_only image2d_t gradient_image) {

    float x = get_global_id(0);
    float y = get_global_id(1);

    float4 p00 = read_imagef(input_image, nearest_sampler, (float2)(x - 1 + 0.5f, y - 1+ 0.5f));
    float4 p10 = read_imagef(input_image, nearest_sampler, (float2)(x+ 0.5f, y - 1+ 0.5f));
    float4 p20 = read_imagef(input_image, nearest_sampler, (float2)(x + 1+ 0.5f, y - 1+ 0.5f));

    float4 p01 = read_imagef(input_image, nearest_sampler, (float2)(x - 1+ 0.5f, y+ 0.5f));
    float4 p21 = read_imagef(input_image, nearest_sampler, (float2)(x + 1+ 0.5f, y+ 0.5f));

    float4 p02 = read_imagef(input_image, nearest_sampler, (float2)(x - 1+ 0.5f, y + 1+ 0.5f));
    float4 p12 = read_imagef(input_image, nearest_sampler, (float2)(x+ 0.5f, y + 1+ 0.5f));
    float4 p22 = read_imagef(input_image, nearest_sampler, (float2)(x + 1+ 0.5f, y + 1+ 0.5f));

    float3 gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;
    float3 gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;
    //for grayscale images
    float gs_x = 0.3333f * (gx.x + gx.y + gx.z);
    float gs_y = 0.3333f * (gy.x + gy.y + gy.z);
	float g = native_sqrt(gs_x * gs_x + gs_y * gs_y);
    write_imagef(gradient_image, (int2)(x, y), (float4)(g, 0.0, 0.0, 1.0));
    //for rgb images
    //float3 g = native_sqrt(gx * gx + gy * gy);
    //write_imagef(gradient_image, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0));
}

__kernel void gradient_filter_2_inputs( __read_only image2d_t input_image_1,
                                        __read_only image2d_t input_image_2,
                                        __write_only image2d_t gradient_image_1,
                                        __write_only image2d_t gradient_image_2) {

    float x = get_global_id(0);
    float y = get_global_id(1);

    float4 p00 = read_imagef(input_image_1, nearest_sampler, (float2)(x - 1 + 0.5f, y - 1+ 0.5f));
    float4 p10 = read_imagef(input_image_1, nearest_sampler, (float2)(x+ 0.5f, y - 1+ 0.5f));
    float4 p20 = read_imagef(input_image_1, nearest_sampler, (float2)(x + 1+ 0.5f, y - 1+ 0.5f));

    float4 p01 = read_imagef(input_image_1, nearest_sampler, (float2)(x - 1+ 0.5f, y+ 0.5f));
    float4 p21 = read_imagef(input_image_1, nearest_sampler, (float2)(x + 1+ 0.5f, y+ 0.5f));

    float4 p02 = read_imagef(input_image_1, nearest_sampler, (float2)(x - 1+ 0.5f, y + 1+ 0.5f));
    float4 p12 = read_imagef(input_image_1, nearest_sampler, (float2)(x+ 0.5f, y + 1+ 0.5f));
    float4 p22 = read_imagef(input_image_1, nearest_sampler, (float2)(x + 1+ 0.5f, y + 1+ 0.5f));

    float3 gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;
    float3 gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;
    //for grayscale images
    float gs_x = 0.3333f * (gx.x + gx.y + gx.z);
    float gs_y = 0.3333f * (gy.x + gy.y + gy.z);
	float g = native_sqrt(gs_x * gs_x + gs_y * gs_y);
    write_imagef(gradient_image_1, (int2)(x, y), (float4)(g, 0.0, 0.0, 1.0));

    p00 = read_imagef(input_image_2, nearest_sampler, (float2)(x - 1 + 0.5f, y - 1+ 0.5f));
    p10 = read_imagef(input_image_2, nearest_sampler, (float2)(x+ 0.5f, y - 1+ 0.5f));
    p20 = read_imagef(input_image_2, nearest_sampler, (float2)(x + 1+ 0.5f, y - 1+ 0.5f));

    p01 = read_imagef(input_image_2, nearest_sampler, (float2)(x - 1+ 0.5f, y+ 0.5f));
    p21 = read_imagef(input_image_2, nearest_sampler, (float2)(x + 1+ 0.5f, y+ 0.5f));

    p02 = read_imagef(input_image_2, nearest_sampler, (float2)(x - 1+ 0.5f, y + 1+ 0.5f));
    p12 = read_imagef(input_image_2, nearest_sampler, (float2)(x+ 0.5f, y + 1+ 0.5f));
    p22 = read_imagef(input_image_2, nearest_sampler, (float2)(x + 1+ 0.5f, y + 1+ 0.5f));

    gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;
    gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;
    //for grayscale images
    gs_x = 0.3333f * (gx.x + gx.y + gx.z);
    gs_y = 0.3333f * (gy.x + gy.y + gy.z);
	g = native_sqrt(gs_x * gs_x + gs_y * gs_y);
    write_imagef(gradient_image_2, (int2)(x, y), (float4)(g, 0.0, 0.0, 1.0));
}

// use Sobel_Filter to create Gradient_images from input images
__kernel void rgb_and_gradient_combiner( __read_only image2d_t input_image_rgb_left,
                                        __read_only image2d_t input_image_rgb_right,
                                        __read_only image2d_t input_image_gradient_left,
                                        __read_only image2d_t input_image_gradient_right,
                                        __write_only image2d_t output_rgb_and_gradient_image_left,
                                        __write_only image2d_t output_rgb_and_gradient_image_right) {

    float x = get_global_id(0);
    float y = get_global_id(1);

    float4 rgb_grad_left = read_imagef(input_image_rgb_left, nearest_sampler, (float2)(x, y));
    float4 rgb_grad_right = read_imagef(input_image_rgb_right, nearest_sampler, (float2)(x, y));
    
    //fill in gradient
    rgb_grad_left.w  =  read_imagef(input_image_gradient_left, nearest_sampler, (float2)(x, y)).x;
    rgb_grad_right.w = read_imagef(input_image_gradient_right, nearest_sampler, (float2)(x, y)).x;

    write_imagef(output_rgb_and_gradient_image_left, (int2)(x, y), rgb_grad_left);
    write_imagef(output_rgb_and_gradient_image_right, (int2)(x, y), rgb_grad_right);
}

/* bilateral filter with local buffer
based on: https://github.com/xidexia/Bilateral-Filtering/blob/master/bilateral.cl */
__kernel void median_3x3(__read_only image2d_t in_values,
                         __write_only image2d_t out_values,
                                         //__local float *buffer, 
                                         //__local float *spatial, 
                        int w, int h, 
                        int work_group_0, int work_group_1){

    float sigma = 2.0f;
    const int halo = 5; 

    const int buf_w = work_group_0 + 2 * halo; 
    const int buf_h = work_group_1 + 2 * halo; 

    __local float buffer[( 16 + 2 * 5 ) * ( 16+ 2 * 5 )]; // halo = 5 , local_size  = 16x16
    __local float spatial[( 16 + 2 * 5 ) * ( 16 + 2 * 5 )];
    
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    const int local_y = get_local_size(1);

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    // Write buffer
    const int size = (2*halo+1);

    
    if (idx_1D<buf_w){
        for (int row = 0; row < buf_h; ++row) {
            float4 tmp_values = read_imagef(in_values, nearest_sampler, (float2)(buf_corner_x + idx_1D, buf_corner_y +row));
            float  tmp_distance = tmp_values.x;
            float tmp_spatial = tmp_values.w;

            buffer[row * buf_w + idx_1D] = tmp_distance; 
            spatial[row * buf_w + idx_1D] = tmp_spatial;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread in the valid region (x < w, y < h) should calculate the weighted average
    if ((y < h) && (x < w)) { // stay in bounds
        float num = 0.0f;
        float den = 0.0f;
        float result = 0.0f;
        float4 disp_values = read_imagef(in_values, nearest_sampler, (float2)(x, y));
        float pixel = disp_values.x;

        int idx = 0;
        for (int i = -halo; i<=halo; ++i){
            for (int j = -halo; j<=halo; ++j){
                // get value of neighbourhood
                float tmp_p = buffer[buf_x+i + buf_w*(buf_y+j)];
                float dif = tmp_p-pixel;
                float value = spatial[buf_x+i + buf_w*(buf_y+j)] * exp(-0.5*(dif*dif)/(sigma*sigma));
                num += tmp_p*value;
                den += value;
                ++idx;
            }
        }
        result = num/den;
        write_imagef(out_values, (int2)(x,y), (float4)(result, 0.0f, 0.0f, 1.0f));
    }
}

/* bilateral filter without local buffer
source: https://github.com/xidexia/Bilateral-Filtering/blob/master/bilateral.cl */
__kernel void median_3x3_bilateral(__read_only image2d_t in_values,
                        __write_only image2d_t out_values,
                        int w, int h){

    float sigma = 2.0f;
    const int halo = 5;
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((y < h) && (x < w)) { // stay in bounds
        float num = 0;
        float den = 0;
        float result = 0;
        float4 disp_values = read_imagef(in_values, nearest_sampler, (float2)(x, y));
        float pixel = disp_values.x;
        
        for (int i = -halo; i<=halo; ++i){
            for (int j = -halo; j<=halo; ++j){
                float tmp_p = read_imagef(in_values, nearest_sampler, (float2)(x + i, y + j)).x;
                float dif = tmp_p-pixel;

                //float value = exp(-0.5*(i*i+j*j)/9) * exp(-0.5*(dif*dif)/(sigma*sigma));
                float spatial_grad = disp_values.w;
                float value = spatial_grad * exp(-0.5*(dif*dif)/(sigma*sigma));
                num += tmp_p*value;
                den += value;   
            }
        }
        result = num/den;
        write_imagef(out_values, (int2)(x,y), (float4)(result, 0.0f, 0.0f, 1.0f));
    }

}

__kernel void convert_image_to_disp_image_unnormalized(__global unsigned char* grayscale_3x8_input_buffer,
                                                       __write_only image2d_t out_disp_image, 
                                                       int image_width, 
                                                       int image_height) {
    
    int2 thread_2d_index = {get_global_id(0), get_global_id(1)};
    //float2 thread_2d_index_32f = {get_global_id(0), get_global_id(1)};
    int const num_channels_rgb = 3;
    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    int const pixel_1d_offset = num_channels_rgb * pixel_1d_index;   

    //3 Channel
    float gray_val_unnormalized = (grayscale_3x8_input_buffer[pixel_1d_offset + 2] +
                                   grayscale_3x8_input_buffer[pixel_1d_offset + 1] +
                                   grayscale_3x8_input_buffer[pixel_1d_offset + 0]) / 3.0f;
    
    //1 Channel
    //float gray_val_unnormalized = grayscale_3x8_input_buffer[pixel_1d_index];
    
    // if (gray_val_unnormalized == 0.0f){
    //     gray_val_unnormalized = 120.0f; //rand_slanted_plane_value(thread_2d_index_32f) *250.0f;
    // }

    float  value_without_vis_scaling = gray_val_unnormalized / 4 ;

    float4 pixel_color = { value_without_vis_scaling, 0.0f, 0.0f, 1.0f };                      

    write_imagef(out_disp_image, thread_2d_index, pixel_color);
}

__kernel void convert_disp_image_to_plane_image(__read_only image2d_t in_disp_image,
                                                __write_only image2d_t out_plane_image_a, 
                                                __write_only image2d_t out_plane_image_b,
                                                int min_disparity , int max_disparity) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);                          
    float2 pos = {x,y};
    //fronto-parallel plane
    float3 normal_vector = {0.0f, 0.0f, 1.0f};
    float4 disparity_value = read_imagef(in_disp_image, nearest_sampler, (float2)(x, y)); // 1 channel img
    float z = disparity_value.x;

    if(/* z == 0.0f */ z < 4.0f){
        float rand_tmp = rand(pos) * (max_disparity - min_disparity) + min_disparity;
        z = rand_tmp;
    }


    
    write_imagef(out_plane_image_a, (int2)(x,y), (float4)(z,normal_vector.x, normal_vector.y, normal_vector.z));
    write_imagef(out_plane_image_b, (int2)(x,y), (float4)(z,normal_vector.x, normal_vector.y, normal_vector.z));
}
