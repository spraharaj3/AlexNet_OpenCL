__kernel void convolve5(__global float * const op4, __global float * const weights5_1, __global float * const weights5_2, __global  float * const op5, const int IMG5, const int FILTER5, const int OP5, const int STRIDE5, __global float * const bias5_1,__global float * const bias5_2)
{
    


    int ci = get_global_id(0);
    int cj = get_global_id(1);
    int x = get_global_id(2);

    float sum = 0,sum2=0;
    int op = OP5;
    int jj = cj*STRIDE5;
    int ii = ci*STRIDE5;
    int ch,r,c,L55 = 256;

    sum=0;sum2=0;
    for(r=jj;r<jj+FILTER5;r++)
    {
        for(c=ii;c<ii+FILTER5;c++)
        {
            for(ch=0;ch<192;ch++)
            {
                    sum+= op4[ch+c*384+r*384*IMG5]*weights5_1[x+ch*128+(c-ii)*192*128+(r-jj)*3*192*128];
                    sum2+=op4[(ch+192)+c*384+r*384*IMG5]*weights5_2[x+ch*128+(c-ii)*192*128+(r-jj)*3*192*128];
            }
        }
    }



    sum +=bias5_1[x];
    sum2+=bias5_2[x];



    op5[x+(cj)*OP5*L55+(ci)*L55] = (sum>0)?sum:0;
    op5[(128+x)+(ci)*L55+(cj)*L55*OP5] = (sum2>0)?sum2:0;    


}
