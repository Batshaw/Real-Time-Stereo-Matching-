__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

__kernel void filter (__read_only image2d_t input_image_1,
                      __write_only image2d_t output_image) {

	int x = (int) get_global_id(0);
	int y = (int) get_global_id(1);
	if (x >= get_image_width(input_image_1) || y >= get_image_height(input_image_1)) {
		return;
	}

	float4 p00 = read_imagef(input_image_1, sampler, (int2)(x - 1, y - 1));
	float4 p10 = read_imagef(input_image_1, sampler, (int2)(x, y - 1));
	float4 p20 = read_imagef(input_image_1, sampler, (int2)(x + 1, y - 1));

	float4 p01 = read_imagef(input_image_1, sampler, (int2)(x - 1, y));
	float4 p21 = read_imagef(input_image_1, sampler, (int2)(x + 1, y));

	float4 p02 = read_imagef(input_image_1, sampler, (int2)(x - 1, y + 1));
	float4 p12 = read_imagef(input_image_1, sampler, (int2)(x, y + 1));
	float4 p22 = read_imagef(input_image_1, sampler, (int2)(x + 1, y + 1));

	float3 gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;
	float3 gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;
	float3 g = native_sqrt(gx * gx + gy * gy);
	float gs_x = 0.3333f * (gx.x + gx.y + gx.z);
	float gs_y = 0.3333f * (gy.x + gy.y + gy.z);
	//float g = native_sqrt(gs_x * gs_x + gs_y * gs_y);
	
	write_imagef(output_image, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0f));
	//write_imagef(output_image, (int2)(x, y), (float4)(g, g, g, 1.0f));
	//float4 o = read_imagef(input_image_1, sampler, (int2)(x, y));
	//write_imagef(output_image, (int2)(x, y), (float4)(o.z, o.y, o.x, 1.0f));
}
