__kernel void convolve(__global float * const input, __global float * const weights,__global  float * const output, const int inputWidth, const int maskWidth, const int opwidth, const int STRIDE, __global float * const bias, const int channels)
{
    int ci = get_global_id(0);
    int cj = get_global_id(1);
    int x = get_global_id(2);

    float sum = 0;
    int op = opwidth;
    int jj = cj*STRIDE;
    int ii = ci*STRIDE;

	
    for (int r = jj; r < jj+maskWidth; r++)
    {
        for (int c = ii; c < ii+maskWidth; c++)
        {
		sum += input[0+r*inputWidth*channels+c*channels]*weights[(r-jj)*96*3*11+(c-ii)*96*3+0*96+x];
		sum += input[1+r*inputWidth*channels+c*channels]*weights[(r-jj)*96*3*11+(c-ii)*96*3+1*96+x];
		sum += input[2+r*inputWidth*channels+c*channels]*weights[(r-jj)*96*3*11+(c-ii)*96*3+2*96+x];
        }
    } 
    
	
	sum+=bias[x];
	



	output[x+cj*opwidth*96+ci*96] = (sum>0)?sum:0; 

}
