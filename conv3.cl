__kernel void convolve3(__global float * mp2, __global float * const weights3,__global  float * const op3, const int IMG3, const int FILTER3, const int OP3, const int STRIDE, __global float * const bias3,int L3)
{
    int ci = get_global_id(0);
    int cj = get_global_id(1);
    int x = get_global_id(2);
    int r,c;

    float sum = 0;
    int op = OP3;
    int ch;
    int L22 = 256;
    L3 = 384;
	int jj = cj*STRIDE;
	int ii = ci*STRIDE;


	sum=0;
        for(r=jj;r<jj+FILTER3;r++)
        {
        	for(c=ii;c<ii+FILTER3;c++)
	       	{
                       	for(ch=0;ch<256;ch++)
                        {               
                        	sum+= mp2[ch+c*L22+r*L22*IMG3]*weights3[x+ch*L3+(c-ii)*L3*L22+(r-jj)*FILTER3*L22*L3];
                        }
                }
       }


       sum+=bias3[x];

       op3[x+(cj+1)*(OP3)*L3+(ci+1)*L3] = (sum>0)?sum:0;


}
