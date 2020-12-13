__kernel void mpool1(__global float * const output, __global float * const mp,const int OP, const int ca, const int MPSTRIDE, const int PADDING, const int l)
{
	

	//ca=MP+4

	int cj = get_global_id(0);
	int ci = get_global_id(1);
	int x = get_global_id(2);
	int L1 = l;
	float max = 0;
	int op = OP;
	int jj = cj*MPSTRIDE;
	int ii = ci*MPSTRIDE;


/*	if(cj==0 || cj==1 || ci==0 || ci==1 || cj==29 || cj==30 || ci==29 || ci==30)
	{
		max = 0;
	}


	else
	{	

*/		for (int r = jj; r <= jj+MPSTRIDE; r++)
    		{
        		for (int c = ii; c <= ii+MPSTRIDE; c++)
		        {
        			if(output[x+r*OP*L1+c*L1]>max)
                		        max = output[x+r*OP*L1+c*L1];  
                      	}
		}
	
		
		mp[x+(cj+PADDING)*L1*(ca)+(ci+PADDING)*L1] = max;
//	}
}
