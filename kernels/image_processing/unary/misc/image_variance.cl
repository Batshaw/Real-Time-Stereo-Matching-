__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

__kernel void variance(__read_only image2d_t input_image,    
               __read_only image2d_t input_image_mean,    
               __write_only image2d_t output_variance,    
               int const windowRadius) {    
    float variance = 0.0f;    
    int x = (int) get_global_id(0);    
    int y = (int) get_global_id(1);    
    if(x < windowRadius || x > get_image_width(input_image) - windowRadius)
        return;
    if(y < windowRadius || y > get_image_height(input_image) - windowRadius)
        return;
    
    
    for(int i = -windowRadius; i <= windowRadius; ++i) {
        for(int j = -windowRadius; j <= windowRadius; ++j) {
            variance += pown(read_imagef(input_image, sampler, (int2)(x + i, y + j)).x, 2);
        }
    }
    int numElem = (2 * windowRadius + 1) * (2 * windowRadius + 1);
    variance /= numElem;
    variance -= pown(read_imagef(input_image_mean, sampler, (int2)(x, y)).x, 2);        
    write_imagef(output_variance, (int2)(x, y), (float4)(variance, 0.0f, 0.0f, 1.0f));
}
