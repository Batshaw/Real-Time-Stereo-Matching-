__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

__kernel void match(__read_only image2d_t image_left,
            __read_only image2d_t image_right,
            __read_only image2d_t mean_left,
            __read_only image2d_t mean_right,
            __read_only image2d_t variance_left,
            __read_only image2d_t variance_right,
            __write_only image2d_t disparityMap,
            int const windowRadius,
            uint const dMin,
            uint const dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    if(x < dMax + windowRadius || x > get_image_width(image_left) - windowRadius) {
        write_imageui(disparityMap, (int2)(x, y), (0, 0, 0, 0));    
        return;
    }
    if(y < windowRadius || y > get_image_height(image_left) - windowRadius) {
        write_imageui(disparityMap, (int2)(x, y), (0, 0, 0, 0));    
        return;
    }
    if(read_imagef(variance_left, sampler, (int2)(x, y)).x <= 0.0f)  // if image area is homogenous, skip search
    {
        write_imageui(disparityMap, (int2)(x, y), (0, 0, 0, 0));    
        return;
    }
    float max_ncc = 0.5f;
    int d = 0;
    //int l = max(windowRadius, x + dMin);
    //int u = max(windowRadius, x + dMax);
    float num = 0.0f;
    float den = 0.0f;
    float ncc = 0.0f;
    for(int k = dMin; k < dMax; ++k) {
        int rx = x - k; // if we search to the left, we need -k, if we search to the right, we night k
        int ry = y;
        for(int i = -windowRadius; i <= windowRadius; ++i) {
            for(int j = -windowRadius; j <= windowRadius; ++j) {
                num += read_imagef(image_left, sampler, (int2)(x + i, y + j)).x
                       * read_imagef(image_right, sampler, (int2)(rx + i, ry + j)).x;
            }
        }
        int numElem = (2 * windowRadius + 1) * (2 * windowRadius + 1);
        num /= numElem;
        num -= read_imagef(mean_left, sampler, (int2)(x, y)).x
               * read_imagef(mean_right, sampler, (int2)(rx, ry)).x;
        den = sqrt(read_imagef(variance_left, sampler, (int2)(x, y)).x 
                * read_imagef(variance_right, sampler, (int2)(rx, ry)).x);
        ncc = num / den;
        if(ncc > max_ncc) {
            d = x - rx;
            max_ncc = ncc;
        }
    }
    write_imageui(disparityMap, (int2)(x, y), (d, d, d, d));    
    //write_imageui(disparityMap, (int2)(x, y), (255, 0, 0, 255));    
}

