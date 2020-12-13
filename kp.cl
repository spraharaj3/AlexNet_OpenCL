__kernel void convolve(const __global  float * const input,	__constant float * const mask,	__global  float * const output,    const int inputWidth,    const int maskWidth,	const int opwidth, const int STRIDE)
{
    const int ci = get_global_id(0);
    const int cj = get_global_id(1);
    int x=0;
    float sum = 0;
    int op = opwidth;
    int jj = cj*STRIDE;
    int ii = ci*STRIDE;

	
    for (int r = jj; r < jj+maskWidth; r++)
    {
        for (int c = ii; c < ii+maskWidth; c++)
        {
			sum += input[r*inputWidth+c]*mask[(r-jj)*maskWidth+(c-ii)];
        }
    } 
    
	output[cj*opwidth + ci] = sum;
}
