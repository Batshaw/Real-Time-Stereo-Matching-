__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;


__kernel void filter (__read_only image2d_t input_image_1,
                      __write_only image2d_t output_image,
		      __constant float * mask,
		      __private int maskSize) {

	int x = (int) get_global_id(0);
	int y = (int) get_global_id(1);

	float3 p;
	for(int a = -maskSize; a < maskSize+1; a++) {
		for(int b = -maskSize; b < maskSize+1; b++) {
		    p.x += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] * read_imagef(input_image_1, sampler, (int2)(x,y) + (int2)(a,b)).x;
		    p.y += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] * read_imagef(input_image_1, sampler, (int2)(x,y) + (int2)(a,b)).y;
		    p.z += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] * read_imagef(input_image_1, sampler, (int2)(x,y) + (int2)(a,b)).z;
		}
	}
	write_imagef(output_image, (int2)(x, y), (float4)(p.z, p.y, p.x, 1.0f));
}
