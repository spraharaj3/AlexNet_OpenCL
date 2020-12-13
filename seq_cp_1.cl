__kernel void convolve(const __global  uint * const input,	__constant uint * const mask,	__global  uint * const output,    const int inputWidth,    const int maskWidth,	const int opwidth)
{
	int STRIDE = 1;
	int IMG = inputWidth;
	int ll = opwidth;
	int ci=0;
	int cj=0;
	int i,j,ii,jj,FILTER=maskWidth,sum=0;
	int r,c;
	

	for(jj=0;jj+FILTER<=IMG;jj=jj+STRIDE)
        {
                for(ii=0;ii+FILTER<=IMG;ii+=STRIDE)
                {
                        sum=0;
                        for(r=jj;r<jj+FILTER;r++)
                        {
                                for(c=ii;c<ii+FILTER;c++)
                                {
                                        sum+=input[r*IMG+c]*mask[(r-jj)*FILTER+(c-ii)];
                                }
                        }

         
         
                        output[ci*ll+cj] = sum;
        
                        cj++;
                }
                cj=0;
                ci++;
        }
	
}
