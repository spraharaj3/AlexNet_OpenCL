__kernel void convolve2(__global float * const mp, __global float * const weights2_1, __global float * const weights2_2, __global  float * const op2, const int IMG2, const int FILTER2, const int OP2, const int STRIDE2, __global float * const bias2_1,__global float * const bias2_2)
{
    


    int ci = get_global_id(0);
    int cj = get_global_id(1);
    int x = get_global_id(2);

    float sum = 0,sum2=0;
    int op = OP2;
    int jj = cj*STRIDE2;
    int ii = ci*STRIDE2;
    int ch,r,c,L22=256;

    sum=0;sum2=0;
    for(r=jj;r<jj+FILTER2;r++)
    {
        for(c=ii;c<ii+FILTER2;c++)
        {
            for(ch=0;ch<48;ch++)
            {
                    sum+= mp[ch+c*96+r*96*IMG2]*weights2_1[x+ch*128+(c-ii)*128*48+(r-jj)*5*48*128];
                    sum2+=mp[(ch+48)+c*96+r*96*IMG2]*weights2_2[x+ch*128+(c-ii)*128*48+(r-jj)*5*48*128];
            }
        }
    }



    sum +=bias2_1[x];
    sum2+=bias2_2[x];



    op2[x+cj*OP2*L22+ci*L22] = (sum>0)?sum:0;
    op2[(128+x)+ci*L22+cj*L22*OP2] = (sum2>0)?sum2:0;    


}
