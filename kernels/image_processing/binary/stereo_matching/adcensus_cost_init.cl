int alu_hamdist_64(unsigned long a, unsigned long b)
{
    int c = a ^ b;
    unsigned long mask = 1;
    int dist = 0;
    for (int i = 0; i < 64; ++i, c >>= 1)
        dist += c & mask;
    return dist;
}

__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

__kernel void mux_average_kernel(__read_only image2d_t image_in, __write_only image2d_t image_out)
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    uint4 color = read_imageui(image_in, sampler,  (int2)(gx,  gy));
    //float avg = ((float)color.x  + (float)color.y + (float)color.z) * 0.3333333333333f;
    //float avg = 0.2989*color.z + 0.5870*color.y + 0.1140 * color.x;
    float avg = 0.2989 * color.x + 0.5870*color.y + 0.1140*color.z;
    write_imageui(image_out, (int2)(gx, gy), (unsigned char)avg);
}

#define DISP_PADDING 60
#define GROUP_SIZE_W 16
#define GROUP_SIZE_H 16
#define ci_ad_kernel_5_BLOCK_X GROUP_SIZE_W+2*DISP_PADDING

__kernel void ci_ad_kernel_5(__read_only image2d_t img_l,   __read_only image2d_t img_r,
                             __write_only image3d_t cost_l, __write_only image3d_t cost_r,
                                int zero_disp, int num_disp)
{
  int gx = (int) get_global_id(0);
  int gy = (int) get_global_id(1);

  int tx = (int) get_local_id(0);
  int ty = (int) get_local_id(1);

  int num_cols = get_image_width(img_l);

  int2 blockDim = {get_local_size(0), get_local_size(1)};

  int sm_cols = ci_ad_kernel_5_BLOCK_X;

  __local unsigned char sm_img[2][GROUP_SIZE_H][ci_ad_kernel_5_BLOCK_X][3];

  int gsx_begin = gx - DISP_PADDING;
  for(int gsy = gy, tsy=ty; tsy < GROUP_SIZE_H; tsy+= blockDim.y, gsy+=blockDim.y)
  {
      for (int gsx=gsx_begin, tsx=tx; tsx < sm_cols; gsx+=blockDim.x, tsx+=blockDim.x)
      {
            int gm_x = min(max(gsx, 0), num_cols - 1);
            uint4 color_l = read_imageui(img_l, sampler,  (int2)(gm_x,  gsy));
            uint4 color_r = read_imageui(img_r, sampler,  (int2)(gm_x,  gsy));

            sm_img[0][tsy][tsx][0] = color_l.x;
            sm_img[0][tsy][tsx][1] = color_l.y;
            sm_img[0][tsy][tsx][2] = color_l.z;

            sm_img[1][tsy][tsx][0] = color_r.x;
            sm_img[1][tsy][tsx][1] = color_r.y;
            sm_img[1][tsy][tsx][2] = color_r.z;
      }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int l_x = tx + DISP_PADDING;
  unsigned char l1_0 = sm_img[0][ty][l_x][0];
  unsigned char l2_0 = sm_img[0][ty][l_x][1];
  unsigned char l3_0 = sm_img[0][ty][l_x][2];

  unsigned char r1_0 = sm_img[1][ty][l_x][0];
  unsigned char r2_0 = sm_img[1][ty][l_x][1];
  unsigned char r3_0 = sm_img[1][ty][l_x][2];

  for (int d = 0; d < num_disp; d++)
  {
        int r_off_x = l_x - (d - zero_disp);
        int l_off_x = l_x + (d - zero_disp);
        unsigned char l1_1 = sm_img[0][ty][l_off_x][0];
        unsigned char l2_1 = sm_img[0][ty][l_off_x][1];
        unsigned char l3_1 = sm_img[0][ty][l_off_x][2];

        unsigned char r1_1 = sm_img[1][ty][r_off_x][0];
        unsigned char r2_1 = sm_img[1][ty][r_off_x][1];
        unsigned char r3_1 = sm_img[1][ty][r_off_x][2];

        float cost_1_l = (float) abs(l1_0 - r1_1);
        float cost_2_l = (float) abs(l2_0 - r2_1);
        float cost_3_l = (float) abs(l3_0 - r3_1);
        float cost_1_r = (float) abs(r1_0 - l1_1);
        float cost_2_r = (float) abs(r2_0 - l2_1);
        float cost_3_r = (float) abs(r3_0 - l3_1);

        float cost_average = (cost_1_l + cost_2_l + cost_3_l) * 0.33333333333f;
        write_imagef(cost_l, (int4)(gx, gy, d, 0), (float4)(cost_average, 0.0, 0.0, 0.0));

        cost_average = (cost_1_r + cost_2_r + cost_3_r) * 0.33333333333f;
        write_imagef(cost_r, (int4)(gx, gy, d, 0), (float4)(cost_average, 0.0, 0.0, 0.0));
  }
}

__kernel void ci_ad_kernel_2(__read_only image2d_t left_image, __read_only image2d_t right_image,
                             __write_only image3d_t left_cost, __write_only image3d_t right_cost,
                             const int zero_disp, const int num_disp)
{
  int gx = (int) get_global_id(0);
  int gy = (int) get_global_id(1);
  int num_cols = get_image_width(left_image);
  float dist = 0.0;
  int other_x = 0;
  for (int d = 0; d < num_disp; ++d)
  {
        other_x  = min(max(gx - (d-zero_disp), 0), num_cols -1);
        uint4 colorL = read_imageui(left_image, sampler,  (int2)(gx,  gy));
        uint4 colorR = read_imageui(right_image, sampler, (int2)(other_x, gy));
        dist  =   abs((int)colorL.x - (int)colorR.x) +
                  abs((int)colorL.y - (int)colorR.y) +
                  abs((int)colorL.z - (int)colorR.z);
        dist  *= 0.33333333;
        //dist /= 255.0; //for visualization
        write_imagef(left_cost, (int4)(gx, gy, d, 0), (float4)(dist, 0.0, 0.0, 0.0));

        other_x = min(max(gx + (d-zero_disp), 0), num_cols - 1);
        colorR = read_imageui(right_image, sampler, (int2)(gx,  gy));
        colorL = read_imageui(left_image, sampler,  (int2)(other_x, gy));
        dist  =   abs((int)colorL.x - (int)colorR.x) +
                  abs((int)colorL.y - (int)colorR.y) +
                  abs((int)colorL.z - (int)colorR.z);
        dist  *= 0.33333333;
        //dist /= 255.0; //for visualization
        write_imagef(right_cost, (int4)(gx, gy, d, 0), (float4)(dist, 0.0, 0.0, 0.0));
  }
}

__kernel void tx_census_9x7_kernel_3(__read_only image2d_t img, __write_only image2d_t census)
{
  int gx = (int) get_global_id(0);
  int gy = (int) get_global_id(1);

  int win_h2 = 3; // Half of 7 + anchor
  int win_w2 = 4; // Half of 9 + anchor

  int num_cols = get_image_width(img);
  int num_rows = get_image_height(img);

  unsigned long c = 0;

  unsigned char compare = read_imageui(img, sampler, (int2)(gx, gy)).x;
  unsigned char neighbor = 0;

  for (int y = -win_h2; y <= win_h2; ++y)
  {
    for (int x = -win_w2; x <= win_w2; ++x)
    {
        int cx = min(max(gx+x, 0), num_cols - 1);
        int cy = min(max(gy+y, 0), num_rows - 1);
        if (x != 0 && y != 0)
        {
            c =  c << 1;
            neighbor = read_imageui(img, sampler, (int2)(cx, cy)).x;
            if (neighbor < compare)
                c = c + 1;
        }
    }
  }

  uint2 rg  = as_uint2(c);
  write_imageui(census, (int2)(gx, gy), (uint4)(rg.x, rg.y, 0, 0));
  //write_imageui(census, (int2)(gx, gy), (uint4)((int)c, 0, 0, 0));
}

__kernel void ci_census_kernel_2(__read_only image2d_t census_l, __read_only image2d_t census_r,
                                  __write_only image3d_t cost_l, __write_only image3d_t cost_r,
                                  const int zero_disp, const int num_disp)
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);
    int num_cols = get_image_width(census_l);
    unsigned long c0, c1;
    float dst;
    int other_x;
    for (int d = 0; d < num_disp; ++d)
    {
        other_x = min(max(gx - (d - zero_disp), 0), num_cols - 1);
        c0 = as_ulong(read_imageui(census_l, sampler, (int2)(gx,      gy)).xy);
        c1 = as_ulong(read_imageui(census_r, sampler, (int2)(other_x, gy)).xy);
        dst = alu_hamdist_64(c0, c1);
        write_imagef(cost_l, (int4)(gx, gy, d, 0), (float4)(dst, 0.0, 0.0, 0.0));

        other_x = min(max(gx + (d - zero_disp), 0), num_cols - 1);
        c0 = as_ulong(read_imageui(census_r, sampler, (int2)(gx,      gy)).xy);
        c1 = as_ulong(read_imageui(census_l, sampler, (int2)(other_x, gy)).xy);
        dst = alu_hamdist_64(c0, c1);
        write_imagef(cost_r, (int4)(gx, gy, d, 0), (float4)(dst, 0.0, 0.0, 0.0));
    }
}

__kernel void ci_adcensus_kernel(__read_only image3d_t ad_cost_l, __read_only image3d_t ad_cost_r,
                                 __read_only image3d_t census_cost_l, __read_only image3d_t census_cost_r,
                                 __write_only image3d_t adcensus_cost_l, __write_only image3d_t adcensus_cost_r,
                                 float inv_ad_coeff, float inv_census_coeff, int zero_disp, int num_disp)
{
    int gx = (int) get_global_id(0);
    int gy = (int) get_global_id(1);

    for (int d =0; d < num_disp; ++d)
    {
        float ad_comp_l     = 1.0 - exp(-inv_ad_coeff * read_imagef(ad_cost_l, sampler, (int4)(gx, gy, d, 0)).x);
        float cencus_comp_l = 1.0 - exp(-inv_census_coeff * read_imagef(census_cost_l, sampler, (int4)(gx, gy, d, 0)).x);

        float ad_comp_r     = 1.0 - exp(-inv_ad_coeff * read_imagef(ad_cost_r, sampler, (int4)(gx, gy, d, 0)).x);
        float cencus_comp_r = 1.0 - exp(-inv_census_coeff * read_imagef(census_cost_r, sampler, (int4)(gx, gy, d, 0)).x);

        write_imagef(adcensus_cost_l, (int4)(gx, gy, d, 0), (float4)(0.5*(ad_comp_l+cencus_comp_l), 0.0, 0.0, 0.0));
        write_imagef(adcensus_cost_r, (int4)(gx, gy, d, 0), (float4)(0.5*(ad_comp_r+cencus_comp_r), 0.0, 0.0, 0.0));
    }
}