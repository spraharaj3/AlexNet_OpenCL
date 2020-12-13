__kernel void nn6(__global float * const input, __global float * const weights,__global  float * const output, __global float * const bias, const int dimin, const int dimout) 
{
    int j = get_global_id(0);

    float sum = 0;

    int i;




                sum=0;
                for(i=0;i<dimin;i++)
                {
                        sum+= input[i]*weights[i*dimout+j];
                }

                sum+=bias[j];


                output[j]= (sum>0)?sum:0;
                

}
