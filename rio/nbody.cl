__kernel void nbody(__global float4 *sourceGlob, __global float *targetGlob) {
  int N = 32;
  float EPS2 = 1e-4;
  float4 d;
  __local float4 sourceLocl[8];
  float4 target = sourceGlob[get_global_id(0)];
  target.w *= -rsqrt(EPS2);
  for( int iblok=0; iblok<N/get_local_size(0); iblok++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    sourceLocl[get_local_id(0)] = sourceGlob[iblok * get_local_size(0) + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for( int i=0; i<get_local_size(0); i++ ) {
      d.x = target.x - sourceLocl[i].x;
      d.y = target.y - sourceLocl[i].y;
      d.z = target.z - sourceLocl[i].z;
      target.w += sourceLocl[i].w * rsqrt(d.x * d.x + d.y * d.y + d.z * d.z + EPS2);
    }
  }
  targetGlob[get_global_id(0)] = target.w;
}
