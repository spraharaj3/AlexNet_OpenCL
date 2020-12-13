/*Rectangular Matrix Convolution host code
Author: Sushanto Praharaj
Date: 19/2/2019*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>
#include "defs.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif











#include "common.c"

#define MAX_SOURCE_SIZE (0x100000)

void checkErr(cl_int err, const char *name);
void checkKernelEnqueue(cl_int err);
void showDeviceInfo(cl_device_id device_id);




float w_mat[11][11][3][96];
/*int inputSignal[IMG][IMG] =
{
		{3, 1, 1, 4, 8, 2, 1, 3},
                {4, 2, 1, 1, 2, 1, 2, 3},
                {4, 4, 4, 4, 3, 2, 2, 2},
                {9, 8, 3, 8, 9, 0, 0, 0},
                {9, 3, 3, 9, 0, 0, 0, 0},
                {0, 9, 0, 8, 0, 0, 0, 0},
                {3, 0, 8, 8, 9, 4, 4, 4},
                {5, 9, 8, 1, 8, 1, 1, 1}
};*/


//int mask[FILTER][FILTER] = {{1,1,1},{1,0,1},{1,1,1}};
float inputSignal[IMG][IMG];
float outputSignal[OP][OP]={0};
float mask[FILTER][FILTER];




int stride = STRIDE;
int inputSignalWidth = IMG;
int inputSignalHeight = IMG;
int outputSignalWidth = OP;
int outputSignalHeight = OP;
int maskWidth = FILTER;
int maskHeight = FILTER;





// Constants

void CL_CALLBACK contextCallback(const char * errInfo,const void * private_info,size_t cb,void * user_data)
{
        printf("Error occured during context use: %s\n\n",errInfo);
        exit(1);
}


int main()
{

	float bufferw[NUM_WEIGHTS];
	float bufferi[NUM_PIXELS];	
	FILE *fo,*fimg,*fbias;
	float image[228][228][3];
	int l,k,c=0;


	










	cl_device_id device_id = NULL;
	cl_int errNum;
        cl_uint ret_num_platforms;
        cl_uint ret_num_devices;
        cl_context context = NULL;
        cl_command_queue command_queue;
        cl_program program;
        cl_kernel kernel;
        cl_mem inputSignalBuffer;
        cl_mem outputSignalBuffer;
        cl_mem maskBuffer;
	cl_platform_id platform_id[2];
        cl_int ret;		


	FILE *fp;
        char fileName[] = "./kp.cl";
        char *source_str;
        size_t source_size;
	int x,y,i,j;

	fp = fopen(fileName, "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

	//printf("\n\nKERNEL IS \n\n%s\n\n",source_str);

        /* Get Platform and Device Info */
        ret = clGetPlatformIDs(2, platform_id, &ret_num_platforms);
        checkErr(ret,"platform_id");
        printf("No. of platforms detected: %d\n",ret_num_platforms);
        for(int i=0; i<ret_num_platforms; i++){
                //Experience the portability of OpenCL, choose either CPU or GPU as the device.
                //ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
                ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
                if(ret == CL_SUCCESS){
                        break;
                }
        }
        checkErr(ret,"device_id");
        showDeviceInfo(device_id);


	//OPEN WEIGHTS FILE. WEIGHTS STORED IN ORDER 11x11x3x96

        fp = fopen("temp.bin","rb");

        fread(bufferw,4,NUM_WEIGHTS,fp);

        printf("\n\n%10.30f\t%10.30f\n\n",bufferw[0],bufferw[1]);



        printf("\nSIZE OF DOUBLE IS %ld\n",sizeof(double));


        ///READ WEIGHTS INTO W_MAT 11X11X3X96
        for(i=0;i<11;i++)
        {
                for(j=0;j<11;j++)
                {
                        for(k=0;k<3;k++)
                        {
                                for(l=0;l<96;l++)
                                {
                                        w_mat[i][j][k][l] = bufferw[c++];
                                }
                        }
                }
        }


        printf("\nVALUE:\t%10.30f\n\n",w_mat[3][4][2][31]);

        fclose(fp);


        /////READ IMAGE

        c=0;

        fimg = fopen("imgfloat.bin","rb");

        fread(bufferi,4,NUM_PIXELS,fimg);

        printf("\n\n%10.30f\n",bufferi[0]);


        printf("\nSIZEOF INT :\t%ld\n",sizeof(int));

        for(i=0;i<228;i++)
        {
                for(j=0;j<228;j++)
                {
                        for(k=0;k<3;k++)
                        {
                                image[i][j][k] = bufferi[c++];
                        }
                }
        }


	printf("\n\n%10.30f\n\n",image[26][121][0]);

        fclose(fimg);


        ///IMAGE READ COMPLETED

	//INIT Zero Matrices
	
		
	for(i=0;i<FILTER;i++)
	{	
		for(j=0;j<FILTER;j++)
		{
			mask[i][j] = w_mat[i][j][0][0];
		}
	}	
	
	for(i=0;i<IMG;i++)
		for(j=0;j<IMG;j++)
			inputSignal[i][j] = image[i][j][0];

	


        /* Create OpenCL context */
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context");
        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue");

	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
        checkErr(ret,"Program");

        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve", &ret); checkErr(ret,"Kernel_PRobleM\n");

	/*Write Device Buffers*/
	cl_mem inputbuffer; //memory object for A matrix      
        inputbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*inputSignalWidth*inputSignalHeight, NULL, NULL);
        cl_mem maskbuffer; //memory object for B matrix              
        maskbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*maskHeight*maskWidth, NULL, NULL);
        cl_mem outputbuffer; //memory object for C matrix              
        outputbuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*outputSignalWidth*outputSignalHeight, NULL, NULL);


	errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputbuffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskbuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputbuffer);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &outputSignalWidth);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &stride);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth,outputSignalHeight};
	const size_t localWorkSize[1]  = { 1 };

	ret = clEnqueueWriteBuffer(command_queue, inputbuffer, CL_TRUE, 0, sizeof(float)*inputSignalWidth*inputSignalHeight, inputSignal, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, maskbuffer, CL_TRUE, 0, sizeof(float)*maskWidth*maskHeight, mask, 0, NULL, NULL); checkErr(ret,"Write Buffer mask");
	
	cl_event event;

	errNum = clEnqueueNDRangeKernel(command_queue, kernel,2,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkErr(errNum, "clEnqueueNDRangeKernel");


	/* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

        cl_ulong time_start;
        cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        double nanoSeconds = time_end-time_start;
        printf("OpenCL Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, outputbuffer, CL_TRUE, 0, outputSignalWidth*outputSignalHeight*sizeof(float), outputSignal, 0, NULL, NULL);

	//PRINT MATRICES

/*	for(i=0;i<IMG;i++)
	{
		for(j=0;j<IMG;j++)
		{
			printf("%d ",inputSignal[i][j]);
		}
		printf("\n");
	}
	
	printf("\n\nMASK\n\n");
	for(i=0;i<FILTER;i++)
        {
                for(j=0;j<FILTER;j++)
                {
                        printf("%10.30f ",mask[i][j]);
                }
                printf("\n");
        }
*/	
	printf("\n\nOUTPUT\n\n");
	
	for(i=0;i<11;i++)
        {
                for(j=0;j<11;j++)
                {
                        printf("%10.10f ",outputSignal[i][j]);
                }
                printf("\n");
        }


	return(0);
}










