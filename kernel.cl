__kernel void convolve(const __global  uint * const input,	__constant uint * const mask,	__global  uint * const output,    const int inputWidth,    const int maskWidth,	const int opwidth)
{
    const int ci = get_global_id(0);
    const int cj = get_global_id(1);
    int x=0;
    int sum = 0;
    int op = opwidth;

	
    for (int r = cj; r < cj+maskWidth; r++)
    {
        for (int c = ci; c < ci+maskWidth; c++)
        {
			sum += input[r*inputWidth+c]*mask[(r-cj)*maskWidth+(c-ci)];
        }
    } 
    
	output[cj*opwidth + ci] = sum;
}
