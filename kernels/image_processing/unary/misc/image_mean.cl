__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;    
     
__kernel void mean (__read_only image2d_t input_image,    
            __write_only image2d_t output_mean, int const windowRadius) {    
    float sum = 0.0f;    
    int x = (int) get_global_id(0);    
    int y = (int) get_global_id(1);    
     
    if(x < windowRadius || x > get_image_width(input_image) - windowRadius)    
        return;    
    if(y < windowRadius || y > get_image_height(input_image) - windowRadius)    
        return;    
        
    for(int i = -windowRadius; i <= windowRadius; ++i) {    
        for(int j = -windowRadius; j <= windowRadius; ++j) {    
            sum += read_imagef(input_image, sampler, (int2)(x + i, y + j)).x;    
        }    
    }    
    int numElem = (2 * windowRadius + 1) * (2 * windowRadius + 1);    
    sum /= numElem;    

    write_imagef(output_mean, (int2)(x, y), (float4)(sum, 0.0f, 0.0f, 1.0f));    
}    
