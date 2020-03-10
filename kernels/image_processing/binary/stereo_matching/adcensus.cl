#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

float ad(__read_only image2d_t left_image,
         __read_only image2d_t right_image,
         int wL, int wR, int h)
{
    float dist = 0.0;
    uint4 colorL = read_imageui(left_image, sampler, (int2)(wL, h));
    uint4 colorR = read_imageui(right_image, sampler, (int2)(wR, h));
    dist = abs((int)colorL.x - (int)colorR.x) +
           abs((int)colorL.y - (int)colorR.y) +
           abs((int)colorL.z - (int)colorR.z);
    dist /= 3.0;
    return dist;
}

float census(__read_only image2d_t left_image,
             __read_only image2d_t right_image,
             int wL, int wR, int h, int censusWinH, int censusWinW)
{
   float dist = 0.0;
   uint4 colorRefL = read_imageui(left_image, sampler, (int2)(wL, h));
   uint4 colorRefR = read_imageui(right_image, sampler, (int2)(wR, h));
   int avg_ref_l = (int)(0.33333333 * (colorRefL.x + colorRefL.y + colorRefL.z));
   int avg_ref_r = (int)(0.33333333 * (colorRefR.x + colorRefR.y + colorRefR.z));

   for(int y = -censusWinH / 2; y <= censusWinH / 2; ++y) {
       for(int x = -censusWinW / 2; x <= censusWinW / 2; ++x) {
           uint4 colorLP = read_imageui(left_image, sampler, (int2)(wL + x, h + y));
           uint4 colorRP = read_imageui(right_image, sampler, (int2)(wR + x, h + y));
           int avg_l = (int)(0.33333333 * (colorLP.x + colorLP.y + colorLP.z));
           int avg_r = (int)(0.33333333 * (colorRP.x + colorRP.y + colorRP.z));
           if ((avg_l - avg_ref_l) * (avg_r - avg_ref_r) < 0)
                dist += 1;

           //if(((int)colorLP.x - (int)colorRefL.x) * ((int)colorRP.x - (int)colorRefR.x) < 0)
           //    dist += 1;
           //if(((int)colorLP.y - (int)colorRefL.y) * ((int)colorRP.y - (int)colorRefR.y) < 0)
           //    dist += 1;
           //if(((int)colorLP.z - (int)colorRefL.z) * ((int)colorRP.z - (int)colorRefR.z) < 0)
           //    dist += 1;
       }
   }
   return dist;
}

__kernel void ad_census (__read_only image2d_t left_image,
                         __read_only image2d_t right_image,
                         __write_only image3d_t cost_volume,
                         const int direction, const int dMin, const int dMax,
                         const float lambdaAD, const float lambdaCensus,
                         const int censusWinH, const int censusWinW)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);
    //int d = (int) get_global_id(2);

    for(int d = 0; d < dMax - dMin; d++) {
        int wL = x;
        int wR = x;
        if(direction > 0)
            wL = x + d;
        else
            wR = x - d;

        float dist = 0.0;

        if (wL - censusWinW / 2 < 0 || wL + censusWinW >= get_image_width(left_image)
            || wR - censusWinW / 2 < 0 || wR + censusWinW / 2 >= get_image_width(left_image)
            || y - censusWinH /2 < 0 || y + censusWinH / 2 >= get_image_height(left_image)) {
            const float DEFAULT_BORDER_COST = 0.999;
            dist = DEFAULT_BORDER_COST; 
            write_imagef(cost_volume, (int4)(x, y, d, 0), (float4)(dist, 0.0, 0.0, 0.0));
            continue;
        }

        float costAd = ad(left_image, right_image, wL, wR, y);
        float costCensus = census(left_image, right_image, wL, wR, y, censusWinH, censusWinW);
        dist = (1 - exp(-costAd / lambdaAD) + 1 - exp(-costCensus / lambdaCensus)) / 2.0;
        write_imagef(cost_volume, (int4)(x, y, d, 0), (float4)(dist, 0.0, 0.0, 0.0));
    }
}

uint colorDiff(uint4 p1, uint4 p2) 
{
    // return biggest channel difference
    return max(max(abs((int)p1.x - (int)p2.x), abs((int)p1.y - (int)p2.y)), abs((int)p1.z - (int)p2.z));
}

__kernel void compute_limits(__read_only image2d_t input_image,
                             __write_only image2d_t limits_image,
                             const int tau1, const int tau2,
                             const int L1, const int L2)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint4 p = read_imageui(input_image, sampler, (int2)(x, y));

    // compute limits for Up, Down, Left, Right arm in this order
    char directionH[4] = {-1, 1, 0, 0};
    char directionW[4] = {0, 0, -1, 1};
    int dist[4];
    for(char direction = 0; direction < 4; direction++) {
        uint4 p2 = p;   
        int d = 1;
        int h1 = y + directionH[direction];
        int w1 = x + directionW[direction];
        int imgWidth = get_image_width(input_image);
        int imgHeight = get_image_height(input_image);

        bool inside = (0 <= h1) && (h1 < imgHeight) && (0 <= w1) && (w1 < imgWidth);
        if(inside) {
            bool colorCond = true; bool wLimitCond = true; bool fColorCond = true;
            while(colorCond && wLimitCond && fColorCond && inside) {
                uint4 p1 = read_imageui(input_image, sampler, (int2)(w1, h1));
                // check if color similar enough
                colorCond = colorDiff(p, p1) < tau1 && colorDiff(p1, p2) < tau1;
                // check if we exceed the length
                wLimitCond = d < L1;
                // check for color similarities of further away neighbors
                fColorCond = (d <= L2) || (d > L2 && colorDiff(p, p1) < tau2);
                p2 = p1; h1 += directionH[direction]; w1 += directionW[direction];
                // check if we are still inside the image
                inside = (0 <= h1) && (h1 < imgHeight) && (0 <= w1) && (w1 < imgWidth);
                d++;
            }
            d--;
        }
        dist[direction] = d - 1;
    }

    write_imageui(limits_image, (int2)(x, y), (uint4)(dist[0], dist[1], dist[2], dist[3])); 
}

__kernel void volume_to_image(__read_only image3d_t cost_volume,
                              __write_only image2d_t debug_image,
                              const int d)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float4 cost = read_imagef(cost_volume, sampler, (int4)(x, y, d, 0));
    write_imagef(debug_image, (int2)(x, y), cost.x);
}

__kernel void cost_to_disparity(__read_only image3d_t cost_volume,
                                __write_only image2d_t disparity_image,
                                const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint disparity = 0;
    float wta_cost = MAXFLOAT;
    for(int d = 0; d <= dMax - dMin; ++d) {
        float cost = read_imagef(cost_volume, sampler, (int4)(x, y, d, 0)).x;
        if(cost < wta_cost) {
            wta_cost = cost;
            disparity = (uint)d + (uint)dMin;
        }
    }
    write_imageui(disparity_image, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
}

__kernel void aggregate_hor_integral(__read_only image3d_t cost_volume_src,
                                     __write_only image3d_t cost_volume_target,
                                     const int dMin, const int dMax)
{
    int row = (int) get_global_id(0);
    const int width = get_image_width(cost_volume_src);
    float sum = 0.0f;
    for(int d = 0; d < dMax - dMin; d++) {
        for(int column = 0; column < width; column++) {
            sum += read_imagef(cost_volume_src, sampler, (int4)(column, row, d, 0)).x;
            write_imagef(cost_volume_target, (int4)(column, row, d, 0), (float4)(sum, 0.0, 0.0, 0.0)); 
        }
    }
}

__kernel void aggregate_hor_integral2(__read_only image3d_t cost_volume_src,
                                     __write_only image3d_t cost_volume_target,
                                     __read_only image2d_t limits_image,
                                     const int dMin, const int dMax)
{
    int row = (int) get_global_id(0);
    const int width = get_image_width(cost_volume_src);

    for(int d = 0; d < dMax - dMin; d++) {
        for(int column = 0; column < width; column++) {
            uint4 limits = read_imageui(limits_image, sampler, (int2)(column, row));
            float left = read_imagef(cost_volume_src, sampler, (int4)(column - limits.z, row, d, 0)).x;
            float right = read_imagef(cost_volume_src, sampler, (int4)(column - limits.w, row, d, 0)).x;
            write_imagef(cost_volume_target, (int4)(column, row, d, 0), (float4)(right - left, 0.0, 0.0, 0.0)); 
        }
    }
}


__kernel void aggregate_hor3(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, 
                        const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int lx = (int) get_local_id(0);
    int ly = (int) get_local_id(1);
    
    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

    __local float band[32][32*3];
    
    for(int d = 0; d < dMax - dMin; d++) {
        // transfer texture memory to local mem
        for(int i = 0; i < 3; i++) {
            int gx = x - 32 + i * 32;
            band[ly][lx + i * 32] = read_imagef(cost_volume_src, sampler, (int4)(gx, y, d, 0)).x;
            //printf("Disparity: %2d - Local: %3d %3d - Global: %3d %3d - %3d %3d to local: band[%2d][%2d]\n", d, lx, ly, x, y, gx, y, ly, i * 32 + lx);
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        
        float cost = 0.0;
        for(int dist = -((int)limits.z); dist <= (int)limits.w; dist++) {
            cost += band[ly][lx + 32 + dist];
        }
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));

        //work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void aggregate_hor2(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, 
                        const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int lx = (int) get_local_id(0);
    int ly = (int) get_local_id(1);
    
    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

    __local float band[16][16*5];
    
    for(int d = 0; d < dMax - dMin; d++) {
        // transfer texture memory to local mem
        for(int i = 0; i < 5; i++) {
            int gx = x - 32 + i * 16;
            band[ly][lx + i * 16] = read_imagef(cost_volume_src, sampler, (int4)(gx, y, d, 0)).x;
            //printf("Disparity: %2d - Local: %3d %3d - Global: %3d %3d - %3d %3d to local: band[%2d][%2d]\n", d, lx, ly, x, y, gx, y, ly, i * 16 + lx);
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        
        float cost = 0.0;
        for(int dist = -((int)limits.z); dist <= (int)limits.w; dist++) {
            cost += band[ly][lx + 2 * 16 + dist];
        }
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));

        //work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void aggregate_ver2(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, 
                        const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int lx = (int) get_local_id(0);
    int ly = (int) get_local_id(1);

    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

    __local float band[16*5][16];

    for(int d = 0; d < dMax - dMin; d++) {
        // transfer texture memory to local mem
        for(int i = 0; i < 5; i++) {
            int gy = y - 32 + i * 16;
            band[ly + i * 16][lx] = read_imagef(cost_volume_src, sampler, (int4)(x, gy, d, 0)).x;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        float cost = 0.0;
        for(int dist = -((int)limits.x); dist <= (int)limits.y; dist++) {
            cost += band[ly + 2 * 16 + dist][lx];
        }
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));

        //work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void aggregate_hor(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, 
                        const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

    for(int d = 0; d < dMax - dMin; d++) {
        float cost = 0.0;
        for(int dist = -((int)limits.z); dist <= (int)limits.w; dist++) {
            cost += read_imagef(cost_volume_src, sampler, (int4)(x + dist, y, d, 0)).x;
        }
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));
    }
}

__kernel void aggregate_ver(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, 
                        const int dMin, const int dMax)
{
int x = (int) get_global_id(0);
int y = (int) get_global_id(1);

uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

for(int d = 0; d < dMax - dMin; d++) {
    float cost = 0.0;

        for(int dist = -((int)limits.x); dist <= (int)limits.y; dist++) {
            cost += read_imagef(cost_volume_src, sampler, (int4)(x, y + dist, d, 0)).x;
        }
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));
    }
}

__kernel void agg_normalize(__read_only image3d_t cost_volume_src,
                        __write_only image3d_t cost_volume_target,
                        __read_only image2d_t limits_image, const int horizontalFirst,
                        const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));

    for(int d = 0; d < dMax - dMin; d++) {
        int count = 0;
        if(horizontalFirst == 1) {
            for(int dist = -((int)limits.x); dist <= (int)limits.y; dist++) {
                uint4 ud_limits = read_imageui(limits_image, sampler, (int2)(x, y + dist));
                count += 1 + ud_limits.z + ud_limits.w;
            }
        }
        else {
            for(int dist = -((int)limits.z); dist <= (int)limits.w; dist++) {
                uint4 lr_limits = read_imageui(limits_image, sampler, (int2)(x + dist, y));
                count += 1 + lr_limits.x + lr_limits.y;
            } 
        }
        float cost = read_imagef(cost_volume_src, sampler, (int4)(x, y, d, 0)).x;
        cost /= count;
        write_imagef(cost_volume_target, (int4)(x, y, d, 0), (float4)(cost, 0.0, 0.0, 0.0));
    }

}


// TODO: handle disparity direction in case of RIGHT -> LEFT optimization
__kernel void scanline_optimize(__read_only image2d_t image_1,
                                __read_only image2d_t image_2,
                                __read_only image3d_t cost_volume_src,
                                __write_only image3d_t cost_volume_target,
                                const int dMin, const int dMax,
                                const float Pi1, const float Pi2, const int tauSO,
                                const int direction, const int vertical, const int right)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);
    //int right = 0;
    int height1, height2, width1, width2;

    if(vertical) {
        if(y == 0 || y == get_image_height(image_1) - 1) { return; }
        height1 = y + direction;
        height2 = height1 - direction;
        width1 = x;
        width2 = x;
    } else {
        if(x == 0 || x == get_image_width(image_1) - 1) { return; }
        height1 = y;
        height2 = y;
        width1 = x + direction;
        width2 = width1 - direction;

    }

    float minOptCost = read_imagef(cost_volume_src, sampler, (int4)(width2, height2, 0, 0)).x;
    for(int disparity = 1; disparity < dMax - dMin; ++disparity) {
        float tmpCost = read_imagef(cost_volume_src, sampler, (int4)(width2, height2, disparity, 0)).x;
        if(minOptCost > tmpCost)
            minOptCost = tmpCost;
    }
    
    float minkCr = minOptCost;
    uint4 c1 = read_imageui(image_1, sampler, (int2)(width1, height1));
    uint4 c2 = read_imageui(image_1, sampler, (int2)(width2, height2));
    int d1 = colorDiff(c1, c2);

    for(int disparity = 0; disparity < dMax - dMin; ++disparity) {
        float cost = read_imagef(cost_volume_src, sampler, (int4)(width1, height1, disparity, 0)).x;
        cost -= minkCr;
        
        // compute p1, p2
        float p1, p2;

        int d2 = tauSO + 1;

        int disp = disparity;
        if(right > 0)
            disp = disparity;
        
        if(0 <= width1 + disparity && width1 + disparity < get_image_width(image_1)
           && 0 <= width2 + disparity && width2 + disparity < get_image_width(image_1)) {
            c1 = read_imageui(image_2, sampler, (int2)(width1 + disparity, height1));
            c2 = read_imageui(image_2, sampler, (int2)(width2 + disparity, height2));
            d2 = colorDiff(c1, c2);
        }
        
        if(d1 < tauSO) {
            if(d2 < tauSO) {
                p1 = Pi1; p2 = Pi2;
            } else {
                p1 = Pi1 / 4.0; p2 = Pi2 / 4.0;
            }
        } else {
            if(d2 < tauSO) {
                p1 = Pi1 / 4.0;
                p2 = Pi2 / 4.0;
            } else {
                p1 = Pi1 / 10.0;
                p2 = Pi2 / 10.0;
            }
        }
        // end of p1, p2 computation
        minOptCost = minkCr + p2;
        
        float tmpCost = read_imagef(cost_volume_src, sampler, (int4)(width2, height2, disparity, 0)).x;
        if(minOptCost > tmpCost)
            minOptCost = tmpCost;
        
        if(disparity != 0) {
            tmpCost = read_imagef(cost_volume_src, sampler, (int4)(width2, height2, disparity - 1, 0)).x + p1;
            if(minOptCost > tmpCost)
                minOptCost = tmpCost;
        }
        if(disparity != dMax - dMin - 1) {
            tmpCost = read_imagef(cost_volume_src, sampler, (int4)(width2, height2, disparity + 1, 0)).x + p1;
            if(minOptCost > tmpCost)
                minOptCost = tmpCost;
        }
        
        write_imagef(cost_volume_target, (int4)(width1, height1, disparity, 0), (float4)((cost + minOptCost) / 2.0, 0.0, 0.0, 0.0));
    }
}

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

    uint disparity = read_imageui(disparity_left, sampler, (int2)(x, y)).x;
    uint right_disparity = read_imageui(disparity_right, sampler, (int2)(x - disparity, y)).x;
    // initialize all outlier_mask values with 0 (i hate OpenCL... no zero initialization for device mem)
    //write_imageui(outlier_mask, (int2)(x, y), (uint4)(0, 0, 0, 0)); 
    if(x - ((int)disparity) < 0 || abs(((int)disparity) - ((int)right_disparity)) > tolerance) {
        bool occlusion = true;
        for(int d = dMin; d < dMax && occlusion; d++) {

            right_disparity = read_imageui(disparity_right, sampler, (int2)(x - d, y)).x;
            if(x - d >= 0 && d == (int)right_disparity) {
                occlusion = false;
            }
        }
        if(occlusion) {
            disparity = 0; // TODO: marker region for later stages
            write_imageui(outlier_mask, (int2)(x, y), (uint4)(OCCLUDED_REGION, 0, 0, 0)); 
        }
        else {
            disparity = 0; // TODO: marker region differently for later stages
            write_imageui(outlier_mask, (int2)(x, y), (uint4)(MISMATCH_REGION, 0, 0, 0)); 
        }
    }

    write_imageui(disparity_image, (int2)(x, y), (uint4)(disparity, 0, 0, 0)); 
}


//this define will be added at the compiling time
//#define NUM_DISPARITIES 60

__kernel void region_voting(__read_only image2d_t disparity_src,
                            __read_only image2d_t outlier_mask_src,
			    __read_only image2d_t limits_image,
                            __write_only image2d_t disparity_target,
                            __write_only image2d_t outlier_mask_target,
                            const int dMin,
                            const int dMax,
			    const int horizontal,
                            const int votingThreshold,
                            const float votingRatioThreshold)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    ushort bins[NUM_DISPARITIES] = {0};

    uint outlier = read_imageui(outlier_mask_src, sampler, (int2)(x, y)).x;
    // copy the outlier mask 1 to 1
    write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(outlier, 0, 0, 0));

    if(outlier == 0) {
    	uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
	write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
	return;
    }

    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));
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
            uint4 inner_limits = read_imageui(limits_image, sampler, (int2)(x, y + outer));
            iLA = -((int)inner_limits.z);
            iLB = (int)inner_limits.w;
	}
        else {
            uint4 inner_limits = read_imageui(limits_image, sampler, (int2)(x + outer, y));
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
            
            outlier = read_imageui(outlier_mask_src, sampler, (int2)(width, height)).x;
            if(outlier == 0) { // we dont count outliers (1 and 2)
                vote += 1;
    	        uint disparity = read_imageui(disparity_src, sampler, (int2)(width, height)).x;
                bins[disparity] += 1;
            }
        }
    }    

    // thresholding
    uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
    if(vote <= votingThreshold) {
	write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
        // delete outlier ?
        if(disparity != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
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
	write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
        // delete outlier
        if(disparity != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }

}

__kernel void region_voting_bitwise(__read_only image2d_t disparity_src,
                            __read_only image2d_t outlier_mask_src,
			                __read_only image2d_t limits_image,
                            __write_only image2d_t disparity_target,
                            __write_only image2d_t outlier_mask_target,
                            const int dMin,
                            const int dMax,
			                const int horizontal,
                            const int votingThreshold,
                            const float votingRatioThreshold)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uchar bins[8] = {0,0,0,0,0,0,0,0};

    uint outlier = read_imageui(outlier_mask_src, sampler, (int2)(x, y)).x;
    // copy the outlier mask 1 to 1
    write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(outlier, 0, 0, 0));

    if(outlier == 0) {
    	uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
	    write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
	    return;
    }

    uint4 limits = read_imageui(limits_image, sampler, (int2)(x, y));
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
            uint4 inner_limits = read_imageui(limits_image, sampler, (int2)(x, y + outer));
            iLA = -((int)inner_limits.z);
            iLB = (int)inner_limits.w;
	    }
        else {
            uint4 inner_limits = read_imageui(limits_image, sampler, (int2)(x + outer, y));
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

            outlier = read_imageui(outlier_mask_src, sampler, (int2)(width, height)).x;
            if(outlier == 0) { // we dont count outliers (1 and 2)
                vote += 1;
    	        uchar disparity = read_imageui(disparity_src, sampler, (int2)(width, height)).x;
                uchar mask = 1;
                int dist = 0;
                for (int i = 0; i < 8; ++i, disparity >>= 1)
                    bins[i] += disparity & mask;
            }
        }
    }

    if(vote <= votingThreshold) {
        uchar disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
	    write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
        // delete outlier ?
        if(disparity != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }
    else{
        uchar best_d = 0;

        float thres = 0.5*vote;
        for (int i = 0; i < 8; ++i)
        {
            if (bins[i] > thres)
                best_d = best_d | 1;
            best_d = best_d << 1;
        }

        write_imageui(disparity_target, (int2)(x, y), (uint4)(best_d, 0, 0, 0));

        // delete outlier
        if(best_d != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }
}


__kernel void proper_interpolation(__read_only image2d_t disparity_src,
                                   __read_only image2d_t outlier_mask_src,
                                   __read_only image2d_t left_image,
                                   __write_only image2d_t disparity_target,
                                   __write_only image2d_t outlier_mask_target,
                                   const int maxSearchDepth)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int directionsW[] = {0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1};
    int directionsH[] = {2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2};

    uint outlier = read_imageui(outlier_mask_src, sampler, (int2)(x, y)).x;
    // copy the outlier mask 1 to 1
    write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(outlier, 0, 0, 0));

    uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;

    if(outlier == 0) {
	write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
	return;
    }
    
    int neighborDisps[16] = {disparity, disparity, disparity, disparity,
                             disparity, disparity, disparity, disparity,
                             disparity, disparity, disparity, disparity};

    int neighborDiffs[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                             -1, -1, -1, -1, -1, -1, -1, -1};    

    // fill directional information
    for(int direction = 0; direction < 16; direction++) {
        int hD = y; 
        int wD = x;
        bool inside = true;
        bool gotDisp = false;
        for(int sD = 0; sD < maxSearchDepth && inside && !gotDisp; sD++) {
            if(sD % 2 == 0) {
                hD += directionsH[direction] / 2;
                wD += directionsW[direction] / 2;
            }
            else {
                hD += directionsH[direction] - directionsH[direction] / 2;
                wD += directionsW[direction] - directionsW[direction] / 2;
            }
            inside = hD >= 0 && hD < get_image_height(left_image) && wD >= 0 && wD <= get_image_width(left_image);

            
            outlier = read_imageui(outlier_mask_src, sampler, (int2)(wD, hD)).x;
            
            if(inside && outlier == 0) {
                disparity = read_imageui(disparity_src, sampler, (int2)(wD, hD)).x;
                neighborDisps[direction] = disparity;
                uint4 c1 = read_imageui(left_image, sampler, (int2)(x, y));
                uint4 c2 = read_imageui(left_image, sampler, (int2)(wD, hD));
                uint diff = colorDiff(c1, c2);
                neighborDiffs[direction] = diff;
                gotDisp = true;
            }

        }
    }

    outlier = read_imageui(outlier_mask_src, sampler, (int2)(x, y)).x;
    if(outlier == OCCLUDED_REGION) {
        int minDisp = neighborDisps[0];
        for(int direction = 1; direction < 16; direction++) {
            if(minDisp > neighborDisps[direction])
                minDisp = neighborDisps[direction];
        }

	write_imageui(disparity_target, (int2)(x, y), (uint4)(minDisp, 0, 0, 0));
        // delete outlier
        if(minDisp != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }
    else {
        int minDisp = neighborDisps[0];
        int minDiff = neighborDiffs[0];
        for(int direction = 1; direction < 16; direction++) {
            if(minDiff < 0 || (minDiff > neighborDiffs[direction] && neighborDiffs[direction] > 0)) {
                minDisp = neighborDisps[direction];
                minDiff = neighborDiffs[direction];
            }
        }
	write_imageui(disparity_target, (int2)(x, y), (uint4)(minDisp, 0, 0, 0));
        if(minDisp != 0)
            write_imageui(outlier_mask_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }

}



__constant float gaus[3][3] = { {0.0625, 0.125, 0.0625},
                                {0.1250, 0.250, 0.1250},
                                {0.0625, 0.125, 0.0625} };

__kernel void gaussian_3x3(__read_only image2d_t image_src,
                       __write_only image2d_t image_target)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float sum = 0;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            sum += gaus[i][j] * (float)read_imageui(image_src, sampler, (int2)(x + i - 1, y + j - 1)).x;
        }
    }
    uint val = (uint)min(255, max(0, (int)sum));
    write_imageui(image_target, (int2)(x, y), (uint4)(val, 0, 0, 0));
}

__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };

__kernel void sobel(__read_only image2d_t image_src,
                    __write_only image2d_t image_target,
                    __write_only image2d_t thetas)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float sumx = 0.0;
    float sumy = 0.0;
    float angle = 0.0;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            sumx += sobx[i][j] * (float)read_imageui(image_src, sampler, (int2)(x + i - 1, y + j - 1)).x;
            sumx += soby[i][j] * (float)read_imageui(image_src, sampler, (int2)(x + i - 1, y + j - 1)).x;
        }
    }

    // write the sqrt of the squares (builtin func. hypot())
    uint val = (uint)min(255, max(0, (int)hypot(sumx, sumy)));
    write_imageui(image_target, (int2)(x, y), (uint4)(val, 0, 0, 0));

    angle = atan2(sumy, sumx);
    if(angle < 0)
        angle = fmod((angle + 2 * M_PI), (2 * M_PI));

    val = (uint)(((int)(degrees(angle * (M_PI / 8) + M_PI / 8 -0.0001) / 45) * 45) % 180);
    write_imageui(thetas, (int2)(x, y), (uint4)(val, 0, 0, 0));
}

__kernel void non_max_suppression(__read_only image2d_t image_src,
                                  __read_only image2d_t thetas,
                                  __write_only image2d_t image_target)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint theta = read_imageui(thetas, sampler, (int2)(x, y)).x;
    uint magnitude = read_imageui(image_src, sampler, (int2)(x, y)).x;

    switch(theta) {
    case 0:
        {
            uint east = read_imageui(image_src, sampler, (int2)(x + 1, y)).x;
            uint west = read_imageui(image_src, sampler, (int2)(x - 1, y)).x;
            if(magnitude <= east || magnitude <= west) // suppress if my neighbour is larger
                write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
            else
                write_imageui(image_target, (int2)(x, y), (uint4)(magnitude, 0, 0, 0));
        }
        break;
    case 45:
        {
            uint northeast = read_imageui(image_src, sampler, (int2)(x + 1, y - 1)).x;
            uint southwest = read_imageui(image_src, sampler, (int2)(x - 1, y + 1)).x;
            if(magnitude <= northeast || magnitude <= southwest)
                write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
            else
                write_imageui(image_target, (int2)(x, y), (uint4)(magnitude, 0, 0, 0));
        }
        break;
    case 90:
        {
            uint north = read_imageui(image_src, sampler, (int2)(x, y - 1)).x;
            uint south = read_imageui(image_src, sampler, (int2)(x, y + 1)).x;
            if(magnitude <= north || magnitude <= south)
                write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
            else
                write_imageui(image_target, (int2)(x, y), (uint4)(magnitude, 0, 0, 0));
        }
        break;
    case 135:
        {
            uint northwest = read_imageui(image_src, sampler, (int2)(x - 1, y - 1)).x;
            uint southeast = read_imageui(image_src, sampler, (int2)(x + 1, y + 1)).x;
            if(magnitude <= northwest || magnitude <= southeast)
                write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
            else
                write_imageui(image_target, (int2)(x, y), (uint4)(magnitude, 0, 0, 0));
        }
        break;
    default:
        write_imageui(image_target, (int2)(x, y), (uint4)(magnitude, 0, 0, 0));
        break;
    }
}


__kernel void hysteresis_thresholding(__read_only image2d_t image_src,
                                      __write_only image2d_t image_target,
                                      const uint low, const uint high)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    uint magnitude = read_imageui(image_src, sampler, (int2)(x, y)).x;

    if(magnitude >= high)
        write_imageui(image_target, (int2)(x, y), (uint4)(255, 0, 0, 0));
    else if(magnitude <= low)
        write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    else {
        uint med = (high + low) / 2;
        if(magnitude >= med)
            write_imageui(image_target, (int2)(x, y), (uint4)(255, 0, 0, 0));
        else 
            write_imageui(image_target, (int2)(x, y), (uint4)(0, 0, 0, 0));
    }   
}



__kernel void discontinuity_adjustment(__read_only image2d_t disparity_src,
                                       __read_only image2d_t outlier_mask,
                                       __read_only image3d_t cost_volume,
                                       __read_only image2d_t edges,
                                       __write_only image2d_t disparity_target,
                                       const int dMin)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int directionsH[] = {-1, 1, -1, 1, -1, 1, 0, 0};
    int directionsW[] = {-1, 1, 0, 0, 1, -1, -1, 1};

    uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
    // per default write old disparity value to target
    write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));

    uint edge1 = read_imageui(edges, sampler, (int2)(x, y)).x;
    uint edge2;
    if(edge1 != 0) {
        int direction = -1;
        if(read_imageui(edges, sampler, (int2)(x - 1, y - 1)).x != 0
           && read_imageui(edges, sampler, (int2)(x + 1, y + 1)).x != 0)
            direction = 0;
        else if(read_imageui(edges, sampler, (int2)(x + 1, y - 1)).x != 0
                && read_imageui(edges, sampler, (int2)(x - 1, y + 1)).x != 0)
            direction = 4;
        else if(read_imageui(edges, sampler, (int2)(x, y - 1)).x != 0
                || read_imageui(edges, sampler, (int2)(x, y + 1)).x != 0 ) {
            if(read_imageui(edges, sampler, (int2)(x - 1, y - 1)).x != 0
               || read_imageui(edges, sampler, (int2)(x, y - 1)).x != 0
               || read_imageui(edges, sampler, (int2)(x + 1, y - 1)).x != 0 ) {
               if(read_imageui(edges, sampler, (int2)(x - 1, y + 1)).x != 0
                  || read_imageui(edges, sampler, (int2)(x, y + 1)).x != 0
                  || read_imageui(edges, sampler, (int2)(x + 1, y + 1)).x != 0) {
                   direction = 2;
               }
            }
        }
        else {
            if(read_imageui(edges, sampler, (int2)(x - 1, y - 1)).x != 0
               || read_imageui(edges, sampler, (int2)(x - 1, y)).x != 0
               || read_imageui(edges, sampler, (int2)(x - 1, y + 1)).x != 0)
                if(read_imageui(edges, sampler, (int2)(x + 1, y - 1)).x != 0
                   || read_imageui(edges, sampler, (int2)(x + 1, y)).x != 0
                   || read_imageui(edges, sampler, (int2)(x + 1, y + 1)).x != 0)
                    direction = 6;
        }

        if(direction != -1) {
            write_imageui(disparity_target, (int2)(x, y), (uint4)(0, 0, 0, 0)); // write a mismatch disparity
            direction = (direction + 4) % 8; // select pixels from two sides of the edge
            uint outlier = read_imageui(outlier_mask, sampler, (int2)(x, y)).x;
            uint d1, d2;
            if(outlier == 0) {
                float cost = read_imagef(cost_volume, sampler, (int4)(x, y, disparity - dMin, 0)).x;
                uint outlier1 = read_imageui(outlier_mask, sampler, (int2)(x + directionsW[direction], y + directionsH[direction])).x;
                uint outlier2 = read_imageui(outlier_mask, sampler, (int2)(x + directionsW[direction + 1], y + directionsH[direction + 1])).x;
                float cost1, cost2;
                if(outlier1 == 0) {
                    d1 = read_imageui(disparity_src, sampler, (int2)(x + directionsW[direction], y + directionsH[direction])).x;
                    cost1 = read_imagef(cost_volume, sampler, (int4)(x, y, d1 - dMin, 0)).x;
                } else
                    cost1 = -1.0;
                if(outlier2 == 0) {
                    d2 = read_imageui(disparity_src, sampler, (int2)(x + directionsW[direction], y + directionsH[direction])).x;
                    cost2 = read_imagef(cost_volume, sampler, (int4)(x, y, d2 - dMin, 0)).x;
                } else
                    cost2 = -1.0;

                if(cost1 != -1.0 && cost1 < cost) {
                    disparity = d1;
                    cost = cost1;
                }
                if(cost2 != -1 && cost2 < cost) {
                    disparity = d2;
                }
            }

            write_imageui(disparity_target, (int2)(x, y), (uint4)(disparity, 0, 0, 0));
        }
        
    }
}


__kernel void subpixel_enhancement(__read_only image2d_t disparity_src,
                                   __read_only image3d_t cost_volume,
                                   __write_only image2d_t disparity_float_target,
                                   const int dMin, const int dMax)
{
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);
    
    uint disparity = read_imageui(disparity_src, sampler, (int2)(x, y)).x;
    float interpolatedDisparity = (float)disparity;
    if(disparity > dMin && disparity < dMax) {
        float cost = read_imagef(cost_volume, sampler, (int4)(x, y, ((int)disparity) - dMin, 0)).x;
        float costP = read_imagef(cost_volume, sampler, (int4)(x, y, ((int)disparity) + 1 - dMin, 0)).x;
        float costM = read_imagef(cost_volume, sampler, (int4)(x, y, ((int)disparity) - 1 - dMin, 0)).x;
        float diff = (costP - costM) / (2 * (costP + costM - 2 * cost));
        if(diff > -1 && diff < 1)
            interpolatedDisparity -= diff;
    }
    write_imagef(disparity_float_target, (int2)(x, y), (float4)(interpolatedDisparity, 0.0, 0.0, 0.0));
}


__kernel void median_3x3(__read_only image2d_t disparity_float_src,
                         __write_only image2d_t disparity_float_target)
{

    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float win[9];
    win[0] = read_imagef(disparity_float_src, sampler, (int2)(x - 1, y - 1)).x;
    win[1] = read_imagef(disparity_float_src, sampler, (int2)(x, y - 1)).x;
    win[2] = read_imagef(disparity_float_src, sampler, (int2)(x + 1, y - 1)).x;
    win[3] = read_imagef(disparity_float_src, sampler, (int2)(x - 1, y)).x;
    win[4] = read_imagef(disparity_float_src, sampler, (int2)(x, y)).x;
    win[5] = read_imagef(disparity_float_src, sampler, (int2)(x + 1, y)).x;
    win[6] = read_imagef(disparity_float_src, sampler, (int2)(x - 1, y + 1)).x;
    win[7] = read_imagef(disparity_float_src, sampler, (int2)(x, y + 1)).x;
    win[8] = read_imagef(disparity_float_src, sampler, (int2)(x + 1, y + 1)).x;


    // perform bitonic sort (taken from intel SDK samples)
    float result;
    float fMin = min(win[0], win[1]);
	float fMax = max(win[0], win[1]);
    win[0] = fMin;
    win[1] = fMax;

    fMin = min(win[3], win[2]);
    fMax = max(win[3], win[2]);
    win[3] = fMin;
    win[2] = fMax;

    fMin = min(win[2], win[0]);
    fMax = max(win[2], win[0]);
    win[2] = fMin;
    win[0] = fMax;

    fMin = min(win[3], win[1]);
    fMax = max(win[3], win[1]);
    win[3] = fMin;
    win[1] = fMax;

    fMin = min(win[1], win[0]);
    fMax = max(win[1], win[0]);
    win[1] = fMin;
    win[0] = fMax;

    fMin = min(win[3], win[2]);
    fMax = max(win[3], win[2]);
    win[3] = fMin;
    win[2] = fMax;

    fMin = min(win[5], win[4]);
    fMax = max(win[5], win[4]);
    win[5] = fMin;
    win[4] = fMax;

    fMin = min(win[7], win[8]);
    fMax = max(win[7], win[8]);
    win[7] = fMin;
    win[8] = fMax;

    fMin = min(win[6], win[8]);
    fMax = max(win[6], win[8]);
    win[6] = fMin;
    win[8] = fMax;

    fMin = min(win[6], win[7]);
    fMax = max(win[6], win[7]);
    win[6] = fMin;
    win[7] = fMax;

    fMin = min(win[4], win[8]);
    fMax = max(win[4], win[8]);
    win[4] = fMin;
    win[8] = fMax;

    fMin = min(win[4], win[6]);
    fMax = max(win[4], win[6]);
    win[4] = fMin;
    win[6] = fMax;

    fMin = min(win[5], win[7]);
    fMax = max(win[5], win[7]);
    win[5] = fMin;
    win[7] = fMax;

    fMin = min(win[4], win[5]);
    fMax = max(win[4], win[5]);
    win[4] = fMin;
    win[5] = fMax;

    fMin = min(win[6], win[7]);
    fMax = max(win[6], win[7]);
    win[6] = fMin;
    win[7] = fMax;

    fMin = min(win[0], win[8]);
    fMax = max(win[0], win[8]);
    win[0] = fMin;
    win[8] = fMax;

    win[4] = max(win[0], win[4]);
    win[5] = max(win[1], win[5]);

    win[6] = max(win[2], win[6]);
    win[7] = max(win[3], win[7]);

    win[4] = min(win[4], win[6]);
    win[5] = min(win[5], win[7]);

    //store found median into result
    result = min(win[4], win[5]);

    write_imagef(disparity_float_target, (int2)(x, y), (float4)(result, 0.0, 0.0, 0.0));

}
