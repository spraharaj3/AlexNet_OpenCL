__kernel void convolve4(__global float * const op3, __global float * const weights4_1, __global float * const weights4_2, __global  float * const op4, const int IMG4, const int FILTER4, const int OP4, const int STRIDE4, __global float * const bias4_1,__global float * const bias4_2)
{
    


    int ci = get_global_id(0);
    int cj = get_global_id(1);
    int x = get_global_id(2);

    float sum = 0,sum2=0;
    int op = OP4;
    int jj = cj*STRIDE4;
    int ii = ci*STRIDE4;
    int ch,r,c,L44 = 384;

    sum=0;sum2=0;
    for(r=jj;r<jj+FILTER4;r++)
    {
        for(c=ii;c<ii+FILTER4;c++)
        {
            for(ch=0;ch<192;ch++)
            {
                    sum+= op3[ch+c*384+r*384*IMG4]*weights4_1[x+ch*192+(c-ii)*192*192+(r-jj)*3*192*192];
                    sum2+=op3[(ch+192)+c*384+r*384*IMG4]*weights4_2[x+ch*192+(c-ii)*192*192+(r-jj)*3*192*192];
            }
        }
    }



    sum +=bias4_1[x];
    sum2+=bias4_2[x];



    op4[x+(cj+1)*OP4*L44+(ci+1)*L44] = (sum>0)?sum:0;
    op4[(192+x)+(ci+1)*L44+(cj+1)*L44*OP4] = (sum2>0)?sum2:0;    


}
