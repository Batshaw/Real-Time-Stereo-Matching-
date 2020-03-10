__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__constant sampler_t SAD_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// __constant float3 normal_vector = {0.0f , 0.0f , 1.0f};

__constant float2 rand_helper_vector = {12.9898, 4.1414};

__constant float max_disparity = 60.0f;

float rand(float2 n) { 
	//source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	float intermediate_result = (0.5 * (1 + sin(dot(n, rand_helper_vector)) ) )  * 43758.5453;
	return intermediate_result - floor(intermediate_result);
}

float rand_slanted_plane_value(float2 n) { 
	//source: https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
	float intermediate_result = sin(dot( n, rand_helper_vector)) * 43758.5453;
	return intermediate_result - floor(intermediate_result);
}

float compute_disparity(float2 point, float value_z, float3 normal_vector) {

    float plane_component_a = - normal_vector.x / normal_vector.z;
    float plane_component_b = - normal_vector.y / normal_vector.z;
    float plane_component_c = (normal_vector.x * point.x + normal_vector.y * point.y + normal_vector.z * value_z) / normal_vector.z;

    float disparity = plane_component_a * point.x + plane_component_b * point.y + plane_component_c;


    return disparity;
}

float cost_aggregation_SAD(
        __read_only image2d_t current_image_gray,
        __read_only image2d_t adjunct_image_gray,
        float2 ref_center_pixel,
        float2 search_center_pixel,
        int image_width,
        int image_height){

    float absolute_difference = 0.0f;


    int radius = 8;


    for ( int win_y_idx = -radius; win_y_idx <= radius; ++win_y_idx){
        for (int win_x_idx = -radius; win_x_idx <= radius; ++win_x_idx){
            float2 current_reference_pixel;
            current_reference_pixel.x = ref_center_pixel.x + win_x_idx;
            current_reference_pixel.y = ref_center_pixel.y + win_y_idx;
            
            float2 current_search_pixel;
            current_search_pixel.x = search_center_pixel.x + win_x_idx;
            current_search_pixel.y = search_center_pixel.y + win_y_idx;

            float ref_pixel_value = read_imagef(current_image_gray, SAD_sampler, current_reference_pixel).x; //CL_R == float4 (r, 0.0, 0.0, 1.0)
            float search_pixel_value = read_imagef(adjunct_image_gray, SAD_sampler, current_search_pixel).x; //grayscale has only one channel

            absolute_difference += fabs(search_pixel_value - ref_pixel_value);
        }
    }

    int window_side_length =  (radius*2+1);

    absolute_difference /= window_side_length * window_side_length;

    return absolute_difference;
}

float cost_aggregation_SAD_Census(
        __read_only image2d_t current_image_gray,
        __read_only image2d_t adjunct_image_gray,
        float2 ref_center_pixel,
        float2 search_center_pixel,
        int image_width,
        int image_height) {

    float census_cost = 0.0f;
    float sum_of_absolute_difference_cost = 0.0f;


    int radius = 10;


    float center_ref_pixel_value = read_imagef(current_image_gray, SAD_sampler, ref_center_pixel).x;
    float center_search_pixel_value = read_imagef(adjunct_image_gray, SAD_sampler, search_center_pixel).x;

    for ( int win_y_idx = -radius; win_y_idx <= radius; ++win_y_idx){
        for (int win_x_idx = -radius; win_x_idx <= radius; ++win_x_idx){
            float2 current_reference_pixel;
            current_reference_pixel.x = ref_center_pixel.x + win_x_idx;
            current_reference_pixel.y = ref_center_pixel.y + win_y_idx;
            
            float2 current_search_pixel;
            current_search_pixel.x = search_center_pixel.x + win_x_idx;
            current_search_pixel.y = search_center_pixel.y + win_y_idx;

            float ref_pixel_value = read_imagef(current_image_gray, SAD_sampler, current_reference_pixel).x; //CL_R == float4 (r, 0.0, 0.0, 1.0)
            float search_pixel_value = read_imagef(adjunct_image_gray, SAD_sampler, current_search_pixel).x; //grayscale has only one channel

            if( (center_ref_pixel_value - ref_pixel_value) *  (center_search_pixel_value - search_pixel_value) < 0.0) {
                census_cost += 1;
            }
            float diff =  fabs(search_pixel_value - ref_pixel_value);
            sum_of_absolute_difference_cost += diff * diff;
        }
    }

    int window_side_length =  (radius*2+1);



    sum_of_absolute_difference_cost /= window_side_length * window_side_length;

    float sad_census_costs= (1 - exp(-sum_of_absolute_difference_cost / 10.0) + 1 - exp(-census_cost / 30.0)) / 2.0;


    return sad_census_costs;
}



void refinement(float2 pos,
            __read_only image2d_t plane_image,
            __read_only image2d_t adjunct_image_gray,
            __read_only image2d_t current_image_gray,
            __write_only image2d_t output_plane_image,
            int image_width,
            int image_height,
            int compute_right  
            ) {
        
    float max_change_n = 1.0f;
    float4 current_plane_values  = read_imagef(plane_image, sampler, (float2)(pos.x, pos.y));
    float current_value_z = current_plane_values.x;
    float3 current_normal_vector = {current_plane_values.y,current_plane_values.z, current_plane_values.w};
    float new_z = 0.0;
    float3 new_normal_vector = {0.0f, 0.0f, 0.0f};
    float current_SAD = 0.0f;
    float new_SAD = 0.0f;

    
    for(float max_change_z = max_disparity / 2.0f; max_change_z > 0.1f; max_change_z /= 2.0f) {

        float delta_z = rand_slanted_plane_value(pos) * max_change_z;
        new_z = current_value_z + delta_z;

        float3 delta_normal_vector;
        delta_normal_vector.x = rand_slanted_plane_value( max_change_n * (pos + 0) ) * max_change_n;
        delta_normal_vector.y = rand_slanted_plane_value( max_change_n * (pos + 1) ) * max_change_n;
        delta_normal_vector.z = rand_slanted_plane_value( max_change_n * (pos + 2) ) * max_change_n;

        float3 unnormalized_normal_vector = ((delta_normal_vector) + (current_normal_vector));
        
        new_normal_vector = normalize(unnormalized_normal_vector); 
        
        float current_disparity = compute_disparity(pos, current_value_z, current_normal_vector);
        float new_disparity = compute_disparity(pos, new_z, new_normal_vector);
        // float new_disparity = compute_disparity(pos, new_z, current_normal_vector);

        if(0 == compute_right) {
            current_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray,
                                                pos, (float2)(pos.x + current_disparity, pos.y), image_width, image_height);
            new_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray, 
                                                pos, (float2)(pos.x + new_disparity, pos.y), image_width, image_height);
        }
        else {
            current_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray,
                                                    pos, (float2)(pos.x - current_disparity, pos.y), image_width, image_height);
            new_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray, 
                                                    pos, (float2)(pos.x - new_disparity, pos.y), image_width, image_height);
        }

        if(new_SAD < current_SAD) {
            current_value_z = new_z;
            current_normal_vector = new_normal_vector;
        }

        max_change_n /= 2;         
                                        
    }

    write_imagef(output_plane_image, (int2)(pos.x, pos.y), (float4)(current_value_z, current_normal_vector.x, current_normal_vector.y, current_normal_vector.z));
}

void spatial_propagation_step(float2 current_pos, float2 neigh_pos , image2d_t plane_image,
        __read_only image2d_t adjunct_image_gray,
        __read_only image2d_t current_image_gray,
        __write_only image2d_t another_plane_image,
        int image_width,
        int image_height,
        int compute_right_disparity_map){

    float current_disparity = 0.0f;
    float current_SAD = 0.0f;
    float new_neigh_disparity = 0.0f;
    float neighbor_SAD = 0.0f;
    float4 plane_values = {0.0f, 0.0f , 0.0f, 0.0f};
    float4 neigh_plane_values = {0.0f, 0.0f , 0.0f, 0.0f};
    float3 current_normal_vector = {0.0f, 0.0f , 0.0f};
    float3 neigh_normal_vector = {0.0f, 0.0f , 0.0f};

    float current_z = 0.0f;
    float neigh_z = 0.0f;

    float best_disparity = 0.0f;
    float best_z = 0.0f;
    float3 best_normal_vector = {0.0f , 0.0f, 0.0f};
    float best_SAD = 0.0f;

                //spatial
    plane_values = read_imagef(plane_image, sampler, (float2)(current_pos.x, current_pos.y)) ;
    current_normal_vector = (plane_values.y , plane_values.z , plane_values.w);
    current_z = plane_values.x ;
    current_disparity = compute_disparity(current_pos , current_z ,  current_normal_vector);
    if(compute_right_disparity_map == 0) { 
        current_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray, 
                                            current_pos, (float2)(current_pos.x + current_disparity, current_pos.y), image_width, image_height);
    }
    else {
        current_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray, 
                                        current_pos, (float2)(current_pos.x - current_disparity, current_pos.y), image_width, image_height);
    }
    best_SAD = current_SAD;
    best_z = current_z;
    best_normal_vector = current_normal_vector;

    neigh_plane_values = read_imagef(plane_image, sampler, (float2)(neigh_pos.x, neigh_pos.y)) ;
    neigh_z = neigh_plane_values.x ;
    neigh_normal_vector = (neigh_plane_values.y , neigh_plane_values.z , neigh_plane_values.w);
    new_neigh_disparity =  compute_disparity(neigh_pos , neigh_z ,  neigh_normal_vector);     
                
    if(compute_right_disparity_map == 0) { 
        neighbor_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray, 
                                            current_pos, (float2)(current_pos.x + new_neigh_disparity, current_pos.y), image_width, image_height);
    }
    else{
        neighbor_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray,
                                            current_pos, (float2)(current_pos.x - new_neigh_disparity, current_pos.y), image_width, image_height);
    }

    if(best_SAD > neighbor_SAD){
        best_z = neigh_z;
        best_normal_vector = neigh_normal_vector;
    }

    write_imagef(another_plane_image, (int2)(current_pos.x, current_pos.y), (float4)(best_z, best_normal_vector.x, best_normal_vector.y, best_normal_vector.z)); // 0.0f, 0.0f, 1.0f));
}

void view_compare(float2 current_pos,
                __read_only image2d_t plane_left,
                __read_only image2d_t plane_right,
                __read_only image2d_t adjunct_image_gray,
                __read_only image2d_t current_image_gray,
                __write_only image2d_t result_plane,
                int image_width,
                int image_height,
                int compute_right
            ) {

    
    float2 matching_pos = {0.0f, 0.0f};

    float4 current_plane_values = read_imagef(plane_left, sampler, current_pos);
    float current_z = current_plane_values.x;
    float3 current_normal_vector = {current_plane_values.y, current_plane_values.z, current_plane_values.w};
    float current_disparity = compute_disparity(current_pos, current_z, current_normal_vector);
    float current_SAD = 0.0f;

    float best_z = current_z;
    float3 best_normal_vector = current_normal_vector;

    matching_pos.x = current_pos.x;
    matching_pos.y = current_pos.y;
    float matching_pos_SAD;

    if(0 == compute_right) {
        current_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray,
                                            current_pos, (float2)(current_pos.x + current_disparity, current_pos.y), image_width, image_height);
    }
    else {
        current_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray,
                                    current_pos, (float2)(current_pos.x - current_disparity, current_pos.y), image_width, image_height);
    }

    for(matching_pos.x = current_pos.x; matching_pos.x <= current_pos.x + max_disparity; matching_pos.x += 1.0) {

        float4 matching_pos_value = read_imagef(plane_right, sampler, (float2)(matching_pos.x, matching_pos.y));
        float matching_pos_z = matching_pos_value.x;
        float3 matching_pos_normal_vector = {matching_pos_value.y, matching_pos_value.z, matching_pos_value.w};
        float matching_pos_disparity = compute_disparity(current_pos, matching_pos_z, matching_pos_normal_vector);
        float matching_x_value = 0.0f;

        if(0 == compute_right) {
            matching_x_value = matching_pos.x - matching_pos_disparity;
        }
        else    matching_x_value = matching_pos.x + matching_pos_disparity;
        
        if(round(matching_x_value) == current_pos.x) {
            if(0 == compute_right) {
                matching_pos_SAD = cost_aggregation_SAD_Census(current_image_gray, adjunct_image_gray,
                                                            current_pos, (float2)(current_pos.x + matching_pos_disparity, current_pos.y), image_width, image_height);                
            }
            else {
                matching_pos_SAD = cost_aggregation_SAD_Census(adjunct_image_gray, current_image_gray,
                                                            current_pos, (float2)(current_pos.x - matching_pos_disparity, current_pos.y), image_width, image_height);                
            }
            
            if(current_SAD > matching_pos_SAD) {
                best_z = matching_pos_z;
                best_normal_vector = matching_pos_normal_vector;
            }
        }
    }
    
    write_imagef(result_plane, (int2)(current_pos.x, current_pos.y), (float4)(best_z, best_normal_vector.x, best_normal_vector.y, best_normal_vector.z));
    
}

__kernel void view_propagation(
                __read_only image2d_t plane_left,
                __read_only image2d_t plane_right,
                __read_only image2d_t adjunct_image_gray,
                __read_only image2d_t current_image_gray,
                __write_only image2d_t result_plane_left,
                __write_only image2d_t result_plane_right,
                int image_width,
                int image_height
                ) {

    float2 pos = {get_global_id(0), get_global_id(1)};

    view_compare(pos, plane_left, plane_right, adjunct_image_gray, current_image_gray, result_plane_left, image_width, image_height, 0);
    view_compare(pos, plane_left, plane_right, adjunct_image_gray, current_image_gray, result_plane_right, image_width, image_height, 1);

}


__kernel void random_initialization(
         int image_width,
         int image_height,
         __global float* output_image,
         __write_only image2d_t plane_image) {


    float2 pos = {get_global_id(0), get_global_id(1)};

	int pixel_1d_index = pos.x + pos.y * image_width; // "flatten" index 2D -> 1D
	int num_channels = 3; // for element offset calculation
	int pixel_1d_offset = num_channels * pixel_1d_index; // actual position of 1D pixel

	float random_inital_value_buffer = rand(pos);
    float random_inital_value_image  = rand(pos) * max_disparity;
    float plane_n_x = rand_slanted_plane_value(pos + 0);
    float plane_n_y = rand_slanted_plane_value(pos + 1);
    float plane_n_z = sqrt(1- pow(plane_n_x,2) - pow(plane_n_y,2));
    float plane_random_z_buffer = random_inital_value_buffer;
    float plane_random_z_image = random_inital_value_image;

    //write_imagef(plane_image, (int2)(pos.x, pos.y), (float4)(plane_random_z, plane_n_z, plane_n_y, plane_n_x));
    write_imagef(plane_image, (int2)(pos.x, pos.y), (float4)(plane_random_z_image, plane_n_x, plane_n_y, plane_n_z));

    
    output_image[pixel_1d_offset + 0] =  random_inital_value_buffer;
    output_image[pixel_1d_offset + 1] =  random_inital_value_buffer;
    output_image[pixel_1d_offset + 2] =  random_inital_value_buffer;

}


__kernel void propagation(
        __read_only image2d_t plane_image_left,
        __read_only image2d_t plane_image_right,
        __read_only image2d_t adjunct_image_gray,
        __read_only image2d_t current_image_gray,
        __write_only image2d_t plane_out_left,
        __write_only image2d_t plane_out_right,
        int image_width,
        int image_height,
        int num_propagation_iterations){

    float2 pos = {get_global_id(0), get_global_id(1)};

    int compute_right_disparity_map;

    if(num_propagation_iterations == 0) {  
                      
            for(int working_position_x = 0; working_position_x < image_width; ++working_position_x) { //left to right 
                
                pos.x = working_position_x;
                float2 left_neighbor_pos = {working_position_x - 1, pos.y};
                
                compute_right_disparity_map = 0;
                spatial_propagation_step( pos, left_neighbor_pos , plane_image_left , adjunct_image_gray,
                                        current_image_gray, plane_out_left, image_width, image_height, compute_right_disparity_map);
                compute_right_disparity_map = 1;
                spatial_propagation_step( pos, left_neighbor_pos , plane_image_right , adjunct_image_gray,
                                        current_image_gray, plane_out_right, image_width, image_height, compute_right_disparity_map);
                
            }
    }

    if(num_propagation_iterations == 1){

            for(int working_position_y = 0; working_position_y < image_height; ++working_position_y) { //top to bottom

                pos.y = working_position_y;
                float2 top_neighbor_pos = {pos.x , working_position_y + 1};

                compute_right_disparity_map = 0;
                spatial_propagation_step( pos, top_neighbor_pos , plane_image_left , adjunct_image_gray,
                                        current_image_gray, plane_out_left, image_width, image_height, compute_right_disparity_map) ;
                compute_right_disparity_map = 1;
                spatial_propagation_step( pos, top_neighbor_pos , plane_image_right , adjunct_image_gray,
                                        current_image_gray, plane_out_right, image_width, image_height, compute_right_disparity_map);

            } 
    }

    if(num_propagation_iterations == 2){

            for(int working_position_x = image_width - 1 ; working_position_x >= 0; --working_position_x) { //right to left
                
                pos.x = working_position_x;
                float2 right_neighbor_pos = {working_position_x + 1, pos.y};

                compute_right_disparity_map = 0;
                spatial_propagation_step( pos, right_neighbor_pos , plane_image_left , adjunct_image_gray,
                                        current_image_gray, plane_out_left, image_width, image_height, compute_right_disparity_map) ;
                compute_right_disparity_map = 1;
                spatial_propagation_step( pos, right_neighbor_pos , plane_image_right , adjunct_image_gray,
                                        current_image_gray, plane_out_right, image_width, image_height, compute_right_disparity_map);
            }
    }

    if(num_propagation_iterations == 3){

            for(int working_position_y = image_height -1 ; working_position_y >= 0; --working_position_y) { //bottom to top

                pos.y = working_position_y;
                float2 bottom_neighbor_pos = {pos.x , working_position_y - 1};

                compute_right_disparity_map = 0;
                spatial_propagation_step( pos, bottom_neighbor_pos , plane_image_left , adjunct_image_gray,
                                        current_image_gray, plane_out_left, image_width, image_height, compute_right_disparity_map) ;
                compute_right_disparity_map = 1;
                spatial_propagation_step( pos, bottom_neighbor_pos , plane_image_right , adjunct_image_gray,
                                        current_image_gray, plane_out_right, image_width, image_height, compute_right_disparity_map);
            } 

    }

}

__kernel void convert_RGBA_to_R(
        __read_only image2d_t plane_image,
        __write_only image2d_t disparity_image,
        int image_width,
        int image_height){

            float2 pos = {get_global_id(0), get_global_id(1)};
            float4 input_value = read_imagef(plane_image, sampler, (float2)(pos.x, pos.y));

            float current_value_z = input_value.x; 
            float3 current_normal_vector = {input_value.y, input_value.z, input_value.w};
            float output_disparity = compute_disparity(pos, current_value_z, current_normal_vector);
            
            write_imagef(disparity_image, (int2)(pos.x, pos.y), (float4)(output_disparity/max_disparity, 0.0f, 0.0f, 1.0f));

            //Channel Order : CL_R -> Components of channel data : (r, 0.0, 0.0, 1.0)
}

__kernel void plane_refinement(
            __read_only image2d_t plane_image_left,
            __read_only image2d_t plane_image_right,
            __read_only image2d_t adjunct_image_gray,
            __read_only image2d_t current_image_gray,
            __write_only image2d_t output_plane_image_left,
            __write_only image2d_t output_plane_image_right,
            int image_width,
            int image_height
        ) {

    float2 pos = {get_global_id(0), get_global_id(1)};

    refinement(pos, plane_image_left, adjunct_image_gray, current_image_gray, output_plane_image_left, image_width, image_height, 0);
    refinement(pos, plane_image_right, adjunct_image_gray, current_image_gray, output_plane_image_right, image_width, image_height, 1);
}


__kernel void copy_r_1x8_buffer_to_image_2D(__global unsigned char* r_1x8_input_buffer,
							  					__write_only image2d_t output_image,
							  					int image_width, int image_height) {
	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;
    

    float4 pixel_color = {r_1x8_input_buffer[pixel_1d_index ] / 255.0f,
                        0.0f,
                        0.0f,
                        1.0f};

    write_imagef(output_image, thread_2d_index, pixel_color);
}

__kernel void copy_image_to_buffer_1x8_buffer(__read_only image2d_t input_image,
                                           __global unsigned char* out_bgr_1x8_buffer,
                                           int image_width, int image_height) {

	int2 thread_2d_index = {get_global_id(0), get_global_id(1)};

    int const pixel_1d_index = thread_2d_index.x + thread_2d_index.y * image_width;

    float4 pixel_color = read_imagef(input_image, sampler, thread_2d_index);

    out_bgr_1x8_buffer[pixel_1d_index] = pixel_color.x;
}

