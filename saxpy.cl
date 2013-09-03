
__kernel void SAXPY (__global float* x, __global float* y, __global float* z)
{
  const int i = get_global_id (0);

  z [i] = x[i] + y[i];
  for (int j=0; j<100; j++) {
	  z[i] = sqrt(z[i]);
	  z[i] = z[i] + y[i];
	  z[i] = z[i] / x[i];
	}
}