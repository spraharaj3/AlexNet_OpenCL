/*ALEXNET PARALLELIZED (BIG TIME) USING OPENCL

Author: Sushanto Praharaj
Date: 19/2/2019

*/


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

#define MAX_SOURCE_SIZE (0x10000000)

void checkErr(cl_int err, const char *name);
void checkKernelEnqueue(cl_int err);
void showDeviceInfo(cl_device_id device_id);
void checkKernelErr(cl_int err);



float w_mat[11][11][3][96];

float inputSignal[IMG][IMG];
float outputSignal[OP][OP][L1]={0};
float mask[FILTER][FILTER];
float bias[L1];
float mp[MP+4][MP+4][L1] = {0};

float op2[OP2][OP2][L22];


float weights2_1[NUM_WEIGHTS2];
float weights2_2[NUM_WEIGHTS2];
float weights3[NUM_WEIGHTS3];
float weights4_1[NUM_WEIGHTS4];
float weights4_2[NUM_WEIGHTS4];
float weights5_1[NUM_WEIGHTS5];
float weights5_2[NUM_WEIGHTS5];
float weights6[NUM_WEIGHTS6];
float weights7[NUM_WEIGHTS7];
float weights8[NUM_WEIGHTS8];

float bias2_1[NUM_BIAS2];
float bias2_2[NUM_BIAS2];
float bias3[NUM_BIAS3];
float bias4_1[NUM_BIAS4];
float bias4_2[NUM_BIAS4];
float bias5_1[NUM_BIAS5];
float bias5_2[NUM_BIAS5];
float bias6[NUM_BIAS6];
float bias7[NUM_BIAS7];
float bias8[NUM_BIAS8];


int stride = STRIDE;
int inputSignalWidth = IMG;
int inputSignalHeight = IMG;
int outputSignalWidth = OP;
int outputSignalHeight = OP;
int maskWidth = FILTER;
int maskHeight = FILTER;
int ca = MP+4;
int padding1 = 2;
int channels = 3;
int padding2 = 1,op2width = OP2;
int layers=L1;

//float op2[OP2][OP2][L22];
float mp2[(MP2+2)][MP2+2][L22] = {0};
float op3[OP3+2][OP3+2][L3] = {0};
float op4[OP4+2][OP4+2][L44] = {0};
float op5[OP5][OP5][L55];
float mp5[L55*MP5*MP5];
float op6[OP6];
float op7[OP7];
float op8[OP8];


// Constants

void CL_CALLBACK contextCallback(const char * errInfo,const void * private_info,size_t cb,void * user_data)
{
        printf("Error occured during context use: %s\n\n",errInfo);
        exit(1);
}


int main()
{


	char classes[1000][500],ch;
	float bufferw[NUM_WEIGHTS];
	float bufferi[NUM_PIXELS];	
	FILE *fo,*fimg,*fbias,*fclasses;
	float image[228][228][3];
	int l,k,c=0,dummy,max,pos,classno=0;
	

	cl_device_id device_id = NULL;
	cl_int errNum;
	cl_int errNum1;
        cl_uint ret_num_platforms;
        cl_uint ret_num_devices;
        cl_context context = NULL, context_mpool1 = NULL;
        cl_command_queue command_queue;
        cl_program program, program_mpool1;
        cl_kernel kernel, kernel_mpool1;
        cl_mem inputSignalBuffer;
        cl_mem outputSignalBuffer;
        cl_mem maskBuffer;
	cl_platform_id platform_id[2];
        cl_int ret;		


	FILE *fp,*fp1;
        char fileName[] = "./trial.cl";
	char mpoolname[] = "./mpool1.cl";
        char *source_str,*source_str_mpool1,*source_str_conv2,*source_str_conv3,*source_str_conv4,*source_str_conv5,*source_str_nn;
        size_t source_size, source_size_mpool1,source_size_conv2,source_size_conv3,source_size_conv4,source_size_conv5,source_size_nn;
	int x,y,i,j,d=2;
	double ttime=0;
	
	//READ CLASSES FOR CLASSIFICATION

        fclasses = fopen("classes.txt","r");
        c=0;
      
	if(fclasses == NULL)
        {
                printf("\nClasses Open Error\n");
                exit(0);
        }

        ch = fgetc(fclasses);

        while(ch!=EOF)
        {
                if(ch == '\n')
                {
                        classno++;
                        c=0;
                        ch = fgetc(fclasses);
                }

                else
                {
                        classes[classno][c++] = ch;
                        ch = fgetc(fclasses);
                }
        }


        printf("\n\nClass %d is %s\n\n",4,classes[4]);



	//READ CONV1 KERNEL
	fp = fopen(fileName, "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);


	printf("\nKernel 1 is \n\n%s\n\n",source_str);
	

	//READ MPOOL1 KERNEL
	
	fp1 = fopen(mpoolname, "r");
        if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_mpool1 = (char*)malloc(MAX_SOURCE_SIZE);
        source_size_mpool1 = fread(source_str_mpool1, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);
	
	

	//READ CONV2 Kernel
	
	fp1 = fopen("./conv2.cl","r");
	if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_conv2 = (char*)malloc(MAX_SOURCE_SIZE);
        source_size_conv2 = fread(source_str_conv2, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);
	
	//READ CONV3 KERNEL

	fp1 = fopen("./conv3.cl","r");
        if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_conv3 = (char*)malloc(MAX_SOURCE_SIZE*3);
        source_size_conv3 = fread(source_str_conv3, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);


	//READ CONV4 KERNEL
        fp1 = fopen("./conv4.cl","r");
        if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_conv4 = (char*)malloc(MAX_SOURCE_SIZE*3);
        source_size_conv4 = fread(source_str_conv4, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);

	//READ CONV5 KERNEL
        fp1 = fopen("./conv5.cl","r");
        if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_conv5 = (char*)malloc(MAX_SOURCE_SIZE*3);
        source_size_conv5 = fread(source_str_conv5, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);
	
	//READ NN6 KERNEL
        fp1 = fopen("./nn6.cl","r");
        if (!fp1) {
                fprintf(stderr, "Failed to load kernel file.\n");
                exit(1);
        }
        source_str_nn = (char*)malloc(MAX_SOURCE_SIZE*3);
        source_size_nn = fread(source_str_nn, 1, MAX_SOURCE_SIZE, fp1);
        fclose(fp1);


	printf("\n\nKERNEL IS \n\n%s\n\n",source_str_nn);






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

	//READ BIAS

	fbias = fopen("bias.bin","rb");
        fread(bias,4,NUM_BIAS,fbias);
        fclose(fbias);
        printf("\n\nBIAS1 DONE\n\n");

        ///IMAGE READ COcaLETED

	//READ WEIGHTS 2_1
        fp = fopen("weights2_1.bin","rb");
        dummy = fread(weights2_1,4,NUM_WEIGHTS2,fp);
        printf("\n\n2_1 %10.30f\t%10.30f\n\n",weights2_1[0],weights2_1[128]);
        fclose(fp);

        //READ WEIGHTS 2_2
        fp = fopen("weights2_2.bin","rb");
        dummy = fread(weights2_2,4,NUM_WEIGHTS2,fp);
        printf("\n\n2_2 %10.30f\t%10.30f\n\n",weights2_2[0],weights2_2[128]);
        fclose(fp);

        //READ WEIGHTS 3

        fp = fopen("weights3.bin","rb");
        dummy = fread(weights3,4,NUM_WEIGHTS3,fp);
        printf("\n\n3  %10.30f\t%10.30f\n\n",weights3[0],weights3[256]);
        fclose(fp);

        //READ WEIGHTS 4_1
        fp = fopen("weights4_1.bin","rb");
        dummy = fread(weights4_1,4,NUM_WEIGHTS4,fp);
        printf("\n\n4_1 %10.30f\t%10.30f\n\n",weights4_1[0],weights4_1[128]);
        fclose(fp);

        //READ WEIGHTS 4_2
        fp = fopen("weights4_2.bin","rb");
        dummy = fread(weights4_2,4,NUM_WEIGHTS4,fp);
        printf("\n\n4_2 %10.30f\t%10.30f\n\n",weights4_2[0],weights4_2[128]);
        fclose(fp);

        //READ WEIGHTS 5_1
        fp = fopen("weights5_1.bin","rb");
        dummy = fread(weights5_1,4,NUM_WEIGHTS5,fp);
        printf("\n\n5_1 %10.30f\t%10.30f\n\n",weights5_1[0],weights5_1[128]);
        fclose(fp);

        //READ WEIGHTS 5_2
        fp = fopen("weights5_2.bin","rb");
        dummy = fread(weights5_2,4,NUM_WEIGHTS5,fp);
        printf("\n\n5_2 %10.30f\t%10.30f\n\n",weights5_2[0],weights5_2[128]);
        fclose(fp);

        //READ WEIGHTS 6
        fp = fopen("wts6.bin","rb");
        dummy = fread(weights6,4,NUM_WEIGHTS6,fp);
        printf("\n\n6 %10.30f\t%10.30f\n\n",weights6[0],weights6[128]);
        fclose(fp);

        //READ WEIGHTS 7
        fp = fopen("wts7.bin","rb");
        dummy = fread(weights7,4,NUM_WEIGHTS7,fp);
        printf("\n\n7 %10.30f\t%10.30f\n\n",weights7[0],weights7[128]);
        fclose(fp);

        //READ WEIGHTS 8
        fp = fopen("weights8.bin","rb");
        dummy = fread(weights8,4,NUM_WEIGHTS8,fp);
        printf("\n\n8 %10.30f\t%10.30f\n\n",weights8[0],weights8[128]);
	fclose(fp);


	fbias = fopen("bias2_1.bin","rb");
        dummy = fread(bias2_1,4,NUM_BIAS2,fbias);
        fclose(fbias);

        printf("\n\nBIAS2 DONE\n\n");

        fbias = fopen("bias2_2.bin","rb");
        dummy = fread(bias2_2,4,NUM_BIAS2,fbias);
        fclose(fbias);
        printf("\n\nBIAS 2_2 done\n\n");

        //READ Bias 3
        fbias = fopen("bias3.bin","rb");
        dummy = fread(bias3,4,NUM_BIAS3,fbias);
        fclose(fbias);
        printf("BIAS3\n");
        //READ BIAS 4_1
        fbias = fopen("bias4_1.bin","rb");
        if(fbias==NULL) printf("ERROR\n\n");
        dummy = fread(bias4_1,4,NUM_BIAS4,fbias);
        fclose(fbias);
        printf("BIAS4\n");

        //READ BIAS 4_2
        fbias = fopen("bias4_2.bin","rb");
        if(fbias == NULL) printf("\n\nError\n\n");
        dummy = fread(bias4_2,4,NUM_BIAS4,fbias);

        fclose(fbias);
        printf("BIAS4_2\n");

        //READ BIAS 5_1
        fbias = fopen("bias5_1.bin","rb");
        if(fbias==NULL) printf("ERROR\n\n");
        dummy = fread(bias5_1,4,NUM_BIAS5,fbias);
        fclose(fbias);
        printf("BIAS5\n");

        //READ BIAS 5_2
        fbias = fopen("bias5_2.bin","rb");
        if(fbias == NULL) printf("\n\nError\n\n");
        dummy = fread(bias5_2,4,NUM_BIAS5,fbias);

        //READ BIAS 6
        fbias = fopen("bias6.bin","rb");
        if(fbias == NULL) printf("\n\nError\n\n");
        dummy = fread(bias6,4,NUM_BIAS6,fbias);
        printf("BIASES DONE\n\n");

        //READ BIAS 7
        fbias = fopen("biass7.bin","rb");
        if(fbias == NULL) printf("\n\nError\n\n");
        dummy = fread(bias7,4,NUM_BIAS7,fbias);
        printf("BIASES DONE\n\n");

        //READ BIAS 8

        fbias = fopen("bias8.bin","rb");
	if(fbias == NULL) printf("\n\nError\n\n");
        dummy = fread(bias8,4,NUM_BIAS8,fbias);
        printf("BIASES DONE\n\n");
	







	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CONV1




	int ret1,err1;


        /* Create OpenCL context */
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context");
       
       

	/* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue");
	

	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
        checkErr(ret,"Convolve1");
	
	
        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable111!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }

	
        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve", &ret); checkErr(ret,"Kernel_PRobleM\n");

	/*Wrss#includeite Device Buffers*/
	cl_mem imagebuffer; //ImageBUFFER
        imagebuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_PIXELS, NULL, NULL);
        cl_mem weightsbuffer; //WEIGHTS BUFFER
        weightsbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS, NULL, NULL);
	cl_mem biasbuffer; //BIAS BUFFER
	biasbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS,NULL,NULL);
	cl_mem outputbuffer; //OUTPUT BUFFER
        outputbuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*OP*OP*L1, NULL, NULL);


	errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imagebuffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightsbuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputbuffer);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &outputSignalWidth);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &stride);
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &biasbuffer);
	errNum |= clSetKernelArg(kernel, 8, sizeof(cl_uint), &channels);
	

	checkErr(errNum, "clSetKernelArg");


	size_t globalWorkSize[3] = { outputSignalWidth,outputSignalHeight,L1};
	size_t localWorkSize[1]  = { 1 };

	
	ret = clEnqueueWriteBuffer(command_queue, imagebuffer, CL_TRUE, 0, sizeof(float)*inputSignalWidth*inputSignalHeight*channels, bufferi, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS, bufferw, 0, NULL, NULL); checkErr(ret,"Write Buffer mask");
	ret = clEnqueueWriteBuffer(command_queue, biasbuffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS, bias, 0,NULL,NULL); checkErr(ret, "BIAS WRITE ERROR\n\n");

	
	cl_event event;

	
	errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


	/* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

        cl_ulong time_start;
        cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        double nanoSeconds = time_end-time_start;
        printf("Convolution1 Execution time: %f s\n",nanoSeconds/1000000000.0);
	ttime+=nanoSeconds;
        /* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, outputbuffer, CL_TRUE, 0, L1*OP*OP*sizeof(float), outputSignal, 0, NULL, NULL);

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
                        printf("%10.10f\t",outputSignal[i][j][95]);
                }
                printf("\n");
        }


	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(imagebuffer);
        ret = clReleaseMemObject(weightsbuffer);
        ret = clReleaseMemObject(biasbuffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);




	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context");








	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//MPOOL1









        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_mpool1, (const size_t *)&source_size_mpool1, &ret);
        checkErr(ret,"mpool1");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable22222!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "mpool1", &ret); checkErr(ret,"Kernel_PRobleM\n");

        /*Wrss#includeite Device Buffers*/
        cl_mem conv1buffer; //ImageBUFFER
        conv1buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*L1*OP*OP, NULL, NULL);       
 
        cl_mem mp1buffer; //OUTPUT BUFFER
        mp1buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*(ca)*(ca)*L1, NULL, NULL);


	

        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &conv1buffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mp1buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &outputSignalWidth);
        
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &ca);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &d);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &padding1); 
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layers);


	checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = MP;
	globalWorkSize[1] = MP;
	globalWorkSize[2] = L1;

        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, conv1buffer, CL_TRUE, 0, sizeof(float)*OP*OP*L1, outputSignal, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
      //  ret = clEnqueueWriteBuffer(command_queue, mpbuffer, CL_TRUE, 0, sizeof(float)*(ca)*(MP)*L1, mp, 0, NULL, NULL); checkErr(ret,"Write Buffer mask");
      //  ret = clEnqueueWriteBuffer(command_queue, biasbuffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS, bias, 0,NULL,NULL); checkErr(ret, "BIAS WRITE ERROR\n\n");


       


        errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


        /* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Mpool1 Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, mp1buffer, CL_TRUE, 0, L1*(ca)*(ca)*sizeof(float), mp, 0, NULL, NULL);


	printf("\n\nOutput of mpool1 is\n\n\n");

	for(i=0;i<31;i++)
	{
		for(j=0;j<31;j++)
		{
			printf("%.1f ",mp[i][j][95]);
		}
		
		printf("\n");
	}	
	
	ttime+=nanoSeconds;
	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


	ret = clReleaseMemObject(mp1buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);







	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CONV2

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context Conv2");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue COnv2");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_conv2, (const size_t *)&source_size_conv2, &ret);
        checkErr(ret,"Convolve2 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable conv2!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }
	printf("\n\nGOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve2", &ret); checkErr(ret,"Kernel_PRobleM\n");

        /*Wrss#includeite Device Buffers*/
        cl_mem mpbuffer; //ImageBUFFER
        mpbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(ca)*(ca)*L1, NULL, NULL); checkErr(errNum,"Error\n\n\n");
        cl_mem weightsbuffer2_1; //WEIGHTS BUFFER
        weightsbuffer2_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS2, NULL, NULL);
        cl_mem weightsbuffer2_2; //WEIGHTS BUFFER
        weightsbuffer2_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS2, NULL, NULL);
	cl_mem biasbuffer2_1; //BIAS BUFFER
        biasbuffer2_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS2,NULL,NULL);
	cl_mem biasbuffer2_2; //BIAS BUFFER
        biasbuffer2_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS2,NULL,NULL);
        cl_mem conv2buffer; //OUTPUT BUFFER
        conv2buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP2*OP2*L22, NULL, NULL);


	int img2 = IMG2;
	int filter2 = FILTER2;
	int opp2 = OP2;
	int stride2 = STRIDE2;



        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mpbuffer);    checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightsbuffer2_1);  checkKernelErr(errNum);      
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weightsbuffer2_2);      checkKernelErr(errNum);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &conv2buffer);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &img2);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &filter2);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &opp2);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &stride2);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &biasbuffer2_1);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &biasbuffer2_2);              




        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = OP2;
        globalWorkSize[1] = OP2;
        globalWorkSize[2] = L2;

//        size_t localWorkSize[1]  = { 1 };

	ret = clEnqueueWriteBuffer(command_queue, mpbuffer, CL_TRUE, 0, sizeof(float)*IMG2*IMG2*L1, mp, 0, NULL, NULL); checkErr(ret,"CV 2Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer2_1, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS2, weights2_1, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer2_2, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS2, weights2_2, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");

        ret = clEnqueueWriteBuffer(command_queue, biasbuffer2_1, CL_TRUE, 0, sizeof(float)*NUM_BIAS2, bias2_1, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");
        ret = clEnqueueWriteBuffer(command_queue, biasbuffer2_2, CL_TRUE, 0, sizeof(float)*NUM_BIAS2, bias2_2, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");

        //cl_event event;


        errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


        /* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
  //      clWaitForEvents(1, &event);

        //cl_ulong time_start;
        //cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Convolution1 Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, conv2buffer, CL_TRUE, 0, L22*OP2*OP2*sizeof(float), op2, 0, NULL, NULL);


	for(i=0;i<11;i++)
	{
		for(j=0;j<11;j++)
		{
			printf("%10.10f\t",op2[i][j][0]);
		}
		printf("\n");
	}


	ttime+=nanoSeconds;
	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


      
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);











	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//MPOOL2




	ca = MP2+2;

	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue mpoolw");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_mpool1, (const size_t *)&source_size_mpool1, &ret);
        checkErr(ret,"mpool1");



        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable22222!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }


        kernel = clCreateKernel(program, "mpool1", &ret); checkErr(ret,"Kernel_PRobleM\n");



       

        cl_mem mp2buffer; //OUTPUT BUFFER
        mp2buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(ca)*(ca)*L22, NULL, NULL);

	
	layers = 256;


        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &conv2buffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mp2buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &op2width);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &ca);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &d);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &padding2);
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layers);



        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = MP2;
        globalWorkSize[1] = MP2;
        globalWorkSize[2] = L22;

        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, conv2buffer, CL_TRUE, 0, sizeof(float)*OP2*OP2*L22, op2, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
	
	errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");



        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Mpool1 Execution time: %f s\n",nanoSeconds/1000000000.0);


        ret = clEnqueueReadBuffer(command_queue, mp2buffer, CL_TRUE, 0, L22*(ca)*(ca)*sizeof(float), mp2, 0, NULL, NULL);


        printf("\n\nOutput of mpool1 is\n\n\n");

        for(i=0;i<11;i++)
        {
                for(j=0;j<11;j++)
                {
                        printf("%10.10f\t",mp2[i][j][34]);
                }

                printf("\n");
        }

	ttime+=nanoSeconds;
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);




	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CONV3

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context3");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_conv3, (const size_t *)&source_size_conv3, &ret);
        checkErr(ret,"Convolve3");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL); 
/*        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable333!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }*/
	if (ret == CL_BUILD_PROGRAM_FAILURE) {
	    // Determine the size of the log
   	 size_t log_size;
	    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	    // Allocate memory for the log
	    char *log = (char *) malloc(log_size);

	    // Get the log
	    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	    // Print the log
	    printf("%s\n", log);
	}

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve3", &ret); checkErr(ret,"Kernel_PRobleM\n");

        /*Wrss#includeite Device Buffers*/
	cl_mem mpxbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(ca)*(ca)*L22, NULL,NULL);
        cl_mem weights3buffer; //WEIGHTS BUFFER
        weights3buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS3, NULL, NULL);
        cl_mem bias3buffer; //BIAS BUFFER
        bias3buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS3,NULL,NULL);
        cl_mem conv3buffer; //OUTPUT BUFFER
        conv3buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(OP3+2)*(OP3+2)*L3, NULL, NULL);

	int img3 = IMG3;
	int opp3 = OP3+2;
	int filter3 = FILTER3;
	int stride3 = STRIDE3;
	channels = L3;

        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mpxbuffer);       //checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights3buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &conv3buffer);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &img3);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &filter3);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &opp3);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &stride3);
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &bias3buffer);
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_uint), &channels);


        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = OP3;
	globalWorkSize[1] = OP3;
	globalWorkSize[2] = L3;


	ret = clEnqueueWriteBuffer(command_queue, mpxbuffer, CL_TRUE, 0, sizeof(float)*(ca)*(ca)*L22, mp2, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, weights3buffer, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS3, weights3, 0, NULL, NULL); checkErr(ret,"Write Buffer mask");
        ret = clEnqueueWriteBuffer(command_queue, bias3buffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS3, bias3, 0,NULL,NULL); checkErr(ret, "BIAS WRITE ERROR\n\n");


     


        errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


        /* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

      
      
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	nanoSeconds = time_end-time_start;
        printf("Convolution1 Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, conv3buffer, CL_TRUE, 0, L3*(OP3+2)*(OP3+2)*sizeof(float), op3, 0, NULL, NULL);

	
	printf("\n\nCONV3\n\n");
		
	for(i=0;i<11;i++)
	{
		for(j=0;j<11;j++)
		{
			printf("%10.10f\t",op3[i][j][1]);
		}
		printf("\n");
	}
	
	ttime+=nanoSeconds;
	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CONV4

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context Conv4");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue COnv4");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_conv4, (const size_t *)&source_size_conv4, &ret);
        checkErr(ret,"Convolve4 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable conv4!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }
        printf("\n\nGOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve4", &ret); checkErr(ret,"Kernel_PRobleM\n");

        /*Wrss#includeite Device Buffers*/
/*        cl_mem mpbuffer; //ImageBUFFER
        mpbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(ca)*(ca)*L1, NULL, NULL); checkErr(errNum,"Error\n\n\n");*/

	cl_mem conv3ipbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(IMG4)*(IMG4)*L3, NULL,NULL);	
        cl_mem weightsbuffer4_1; //WEIGHTS BUFFER
        weightsbuffer4_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS4, NULL, NULL);
        cl_mem weightsbuffer4_2; //WEIGHTS BUFFER
        weightsbuffer4_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS4, NULL, NULL);
        cl_mem biasbuffer4_1; //BIAS BUFFER
        biasbuffer4_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS4,NULL,NULL);
        cl_mem biasbuffer4_2; //BIAS BUFFER
        biasbuffer4_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS4,NULL,NULL);
        cl_mem conv4buffer; //OUTPUT BUFFER
        conv4buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(OP4+2)*(OP4+2)*L44, NULL, NULL);


        int img4 = IMG4;
        int filter4 = FILTER4;
        int opp4 = OP4+2;
        int stride4 = STRIDE4;



        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &conv3ipbuffer);    checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightsbuffer4_1);  checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weightsbuffer4_2);      checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &conv4buffer);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &img4);       checkKernelErr(errNum);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &filter4);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &opp4);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &stride4);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &biasbuffer4_1);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &biasbuffer4_2);




        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = OP4;
        globalWorkSize[1] = OP4;
        globalWorkSize[2] = L4;

//        size_t localWorkSize[1]  = { 1 };

        ret = clEnqueueWriteBuffer(command_queue, conv3ipbuffer, CL_TRUE, 0, sizeof(float)*(IMG4)*(IMG4)*L3, op3, 0, NULL, NULL); checkErr(ret,"CV 2Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer4_1, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS4, weights4_1, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer4_2, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS4, weights4_2, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");

        ret = clEnqueueWriteBuffer(command_queue, biasbuffer4_1, CL_TRUE, 0, sizeof(float)*NUM_BIAS4, bias4_1, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");
        ret = clEnqueueWriteBuffer(command_queue, biasbuffer4_2, CL_TRUE, 0, sizeof(float)*NUM_BIAS4, bias4_2, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");

        //cl_event event;


        errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


        /* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
  //      clWaitForEvents(1, &event);

        //cl_ulong time_start;
        //cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Convolution4 Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, conv4buffer, CL_TRUE, 0, L44*(OP4+2)*(OP4+2)*sizeof(float), op4, 0, NULL, NULL);



	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
	ttime+=nanoSeconds;
	printf("\n\nCONV4\n\n");
	for(i=0;i<11;i++)
	{
		for(j=0;j<11;j++)
		{
			printf("%10.10f\t",op4[i][j][201]);
		}
		printf("\n");
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//CONV5

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context Conv5");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue COnv5");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_conv5, (const size_t *)&source_size_conv5, &ret);
        checkErr(ret,"Convolve5 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable conv4!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }
        printf("\n\nGOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "convolve5", &ret); checkErr(ret,"Kernel_PRobleM\n");

        /*Wrss#includeite Device Buffers*/
/*        cl_mem mpbuffer; //ImageBUFFER
        mpbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(ca)*(ca)*L1, NULL, NULL); checkErr(errNum,"Error\n\n\n");*/

        cl_mem conv4ipbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(IMG5)*(IMG5)*L44, NULL,NULL);
        cl_mem weightsbuffer5_1; //WEIGHTS BUFFER
        weightsbuffer5_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS5, NULL, NULL);
        cl_mem weightsbuffer5_2; //WEIGHTS BUFFER
        weightsbuffer5_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS5, NULL, NULL);
        cl_mem biasbuffer5_1; //BIAS BUFFER
        biasbuffer5_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS5,NULL,NULL);
        cl_mem biasbuffer5_2; //BIAS BUFFER
        biasbuffer5_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS5,NULL,NULL);
        cl_mem conv5buffer; //OUTPUT BUFFER
        conv5buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(OP5)*(OP5)*L55, NULL, NULL);


        int img5 = IMG5;
        int filter5 = FILTER5;
        int opp5 = OP5;
        int stride5 = STRIDE5;



        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &conv4ipbuffer);    checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightsbuffer5_1);  checkKernelErr(errNum);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weightsbuffer5_2);      checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &conv5buffer);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &img5);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &filter5);       checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &opp5);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &stride5);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &biasbuffer5_1);          checkKernelErr(errNum);
        errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &biasbuffer5_2);




        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = OP5;
        globalWorkSize[1] = OP5;
        globalWorkSize[2] = L5;

//        size_t localWorkSize[1]  = { 1 };

        ret = clEnqueueWriteBuffer(command_queue, conv4ipbuffer, CL_TRUE, 0, sizeof(float)*(IMG5)*(IMG5)*L44, op4, 0, NULL, NULL); checkErr(ret,"CV 2Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer5_1, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS5, weights5_1, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");
        ret = clEnqueueWriteBuffer(command_queue, weightsbuffer5_2, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS5, weights5_2, 0, NULL, NULL); checkErr(ret,"CV2 Write Buffer mask");

        ret = clEnqueueWriteBuffer(command_queue, biasbuffer5_1, CL_TRUE, 0, sizeof(float)*NUM_BIAS5, bias5_1, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");
        ret = clEnqueueWriteBuffer(command_queue, biasbuffer5_2, CL_TRUE, 0, sizeof(float)*NUM_BIAS5, bias5_2, 0,NULL,NULL); checkErr(ret, "CV2 BIAS WRITE ERROR\n\n");

        //cl_event event;


        errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");


        /* Wait for the event object to complete */
        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
  //      clWaitForEvents(1, &event);

        //cl_ulong time_start;
        //cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Convolution5 Execution time: %f s\n",nanoSeconds/1000000000.0);

        /* Copy results from the memory buffer */
        ret = clEnqueueReadBuffer(command_queue, conv5buffer, CL_TRUE, 0, L55*(OP5)*(OP5)*sizeof(float), op5, 0, NULL, NULL);



        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);



//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);
	ttime+=nanoSeconds;
        printf("\n\nCONV5\n\n");
        for(i=0;i<11;i++)
        {
                for(j=0;j<11;j++)
                {
                        printf("%10.10f\t",op5[i][j][0]);
                }
                printf("\n");
        }

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Mpool5

	
	ca = MP5;

        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue mpool5");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_mpool1, (const size_t *)&source_size_mpool1, &ret);
        checkErr(ret,"mpool5");



        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable22222!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }


        kernel = clCreateKernel(program, "mpool1", &ret); checkErr(ret,"Kernel_PRobleM\n");





        cl_mem mp5buffer; //OUTPUT BUFFER
        mp5buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(ca)*(ca)*L55, NULL, NULL);


        layers = L55;
	int padding = 0;

        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &conv5buffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mp5buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &opp5);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &ca);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &d);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &padding);
        errNum |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layers);



        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = MP5;
        globalWorkSize[1] = MP5;
        globalWorkSize[2] = L55;

        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, conv5buffer, CL_TRUE, 0, sizeof(float)*OP5*OP5*L55, op5, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");

	errNum = clEnqueueNDRangeKernel(command_queue, kernel,3,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");



        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("Mpool1 Execution time: %f s\n",nanoSeconds/1000000000.0);


        ret = clEnqueueReadBuffer(command_queue, mp5buffer, CL_TRUE, 0, L55*(ca)*(ca)*sizeof(float), mp5, 0, NULL, NULL);


        printf("\n\nOutput of mpool5 is\n\n\n");

        for(i=0;i<6;i++)
        {
                for(j=0;j<6;j++)
                {
                        printf("%10.10f\t",mp5[225+j*256+i*256*6]);
                }

                printf("\n");
        }

	ttime+=nanoSeconds;
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// NN6 // 

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context nn6");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue nn6");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_nn, (const size_t *)&source_size_nn, &ret);
        checkErr(ret,"nn6 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable nn6!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }



        printf("\n\nnn6 GOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "nn6", &ret); checkErr(ret,"Kernel_PRobleM\n");

		
	cl_mem nn6inbuffer;
        nn6inbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(ca)*(ca)*(L55), NULL, NULL);
	
	cl_mem wt6buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS6, NULL,NULL);
	cl_mem bias6buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS6, NULL, NULL);
	cl_mem op6buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP6, NULL, NULL);

	int dimin = 9216;
	int dimout = OP6;


        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn6inbuffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &wt6buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &op6buffer);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias6buffer);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &dimin);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &dimout);




        checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = dimout;



        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, nn6inbuffer, CL_TRUE, 0, sizeof(float)*ca*ca*L55, mp5, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
	ret = clEnqueueWriteBuffer(command_queue, wt6buffer, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS6, weights6, 0, NULL, NULL); checkErr(ret, "WEIGHTS6 buffer write\n");
	ret = clEnqueueWriteBuffer(command_queue, bias6buffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS6, bias6, 0, NULL, NULL); checkErr(ret, "BIAS6BUFFER\n");

        errNum = clEnqueueNDRangeKernel(command_queue, kernel,1,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");



        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("NN Execution time: %f s\n",nanoSeconds/1000000000.0);

	ttime+=nanoSeconds;
        ret = clEnqueueReadBuffer(command_queue, op6buffer, CL_TRUE, 0, OP6*sizeof(float), op6, 0, NULL, NULL);


	ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

	printf("\n\nFC6\n\n");

	for(i=0;i<30;i++)
		printf("%10.10f\n",op6[i]);
	

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// NN7 ///

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context nn6");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue nn7");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_nn, (const size_t *)&source_size_nn, &ret);
        checkErr(ret,"nn7 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable nn7!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }



        printf("\n\nnn7 GOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "nn6", &ret); checkErr(ret,"Kernel_PRobleM\n");


        cl_mem nn7inbuffer;
        nn7inbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP6, NULL, NULL);

        cl_mem wt7buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS7, NULL,NULL);
        cl_mem bias7buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS7, NULL, NULL);
        cl_mem op7buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP7, NULL, NULL);

	dimin = 4096;
	dimout = OP7;


        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn7inbuffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &wt7buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &op7buffer);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias7buffer);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &dimin);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &dimout);


	checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = dimout;



        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, nn7inbuffer, CL_TRUE, 0, sizeof(float)*OP6, op6, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, wt7buffer, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS7, weights7, 0, NULL, NULL); checkErr(ret, "WEIGHTS6 buffer write\n");
        ret = clEnqueueWriteBuffer(command_queue, bias7buffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS7, bias7, 0, NULL, NULL); checkErr(ret, "BIAS6BUFFER\n");

        errNum = clEnqueueNDRangeKernel(command_queue, kernel,1,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");



        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("NN Execution time: %f s\n",nanoSeconds/1000000000.0);

	ttime+=nanoSeconds;
        ret = clEnqueueReadBuffer(command_queue, op7buffer, CL_TRUE, 0, OP7*sizeof(float), op7, 0, NULL, NULL);


        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

        printf("\n\nFC7\n\n");

        for(i=0;i<30;i++)
                printf("%10.10f\n",op7[i]);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// NN 8 ////////////////


	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        checkErr(ret,"Context nn8");



        /* Create Command Queue */
        //cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES#include<stdlib.h>,CL_QUEUE_PROFILING_ENABLE, 0};
        //command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        checkErr(ret,"Command Queue nn8");


        program = clCreateProgramWithSource(context, 1, (const char **)&source_str_nn, (const size_t *)&source_size_nn, &ret);
        checkErr(ret,"nn8 Program");


        /* Build Kernel Program */
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(ret != CL_SUCCESS){
                size_t len;
                char   buffer[2048];
                printf("Error: Failed to build program executable nn8!\n");
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n",buffer);
                exit(EXIT_FAILURE);
        }



        printf("\n\nnn8 GOOD\n\n");

        /* Create OpenCL Kernel */
        kernel = clCreateKernel(program, "nn6", &ret); checkErr(ret,"Kernel_PRobleM\n");


        cl_mem nn8inbuffer;
        nn8inbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP7, NULL, NULL);

        cl_mem wt8buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_WEIGHTS8, NULL,NULL);
        cl_mem bias8buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUM_BIAS8, NULL, NULL);
        cl_mem op8buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*OP8, NULL, NULL);

        dimin = 4096;
        dimout = OP8;


        errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &nn8inbuffer);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &wt8buffer);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &op8buffer);

        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias8buffer);
        errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &dimin);
        errNum |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &dimout);


	checkErr(errNum, "clSetKernelArg");


        globalWorkSize[0] = dimout;



        //localWorkSize[]  = { 1 };


        ret = clEnqueueWriteBuffer(command_queue, nn8inbuffer, CL_TRUE, 0, sizeof(float)*OP7, op7, 0, NULL, NULL); checkErr(ret,"Write Buffer ip");
        ret = clEnqueueWriteBuffer(command_queue, wt8buffer, CL_TRUE, 0, sizeof(float)*NUM_WEIGHTS8, weights8, 0, NULL, NULL); checkErr(ret, "WEIGHTS6 buffer write\n");
        ret = clEnqueueWriteBuffer(command_queue, bias8buffer, CL_TRUE, 0, sizeof(float)*NUM_BIAS8, bias8, 0, NULL, NULL); checkErr(ret, "BIAS6BUFFER\n");

        errNum = clEnqueueNDRangeKernel(command_queue, kernel,1,NULL,globalWorkSize, NULL,0, NULL,&event);
        checkKernelEnqueue(errNum);//checkErr(errNum, "clEnqueueNDRangeKernel");



        //cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
        clWaitForEvents(1, &event);

       // cl_ulong time_start;
       // cl_ulong time_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        nanoSeconds = time_end-time_start;
        printf("NN Execution time: %f s\n",nanoSeconds/1000000000.0);

	ttime+=nanoSeconds;
        ret = clEnqueueReadBuffer(command_queue, op8buffer, CL_TRUE, 0, OP8*sizeof(float), op8, 0, NULL, NULL);


        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);


//        ret = clReleaseMemObject(mp2buffer);
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

        printf("\n\nFC8\n\n");

        for(i=0;i<30;i++)
                printf("%10.10f\n",op8[i]);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////// 	PREDICTING CLASS OUTPUTS!!!!!! 	//////////////////////////////

	printf("\n\nCLASSES ARE\n\n");

	for(i=0;i<10;i++)
        {
                max = 0;
                pos = 0;
                for(j=0;j<1000;j++)
                {
                        if(op8[j]>max)
                        {
                                max = op8[j];
                                pos = j;
                        }
                }
                printf("%d\t%s\t%d\%\n\n",pos,classes[pos],max);
                op8[pos] = -999;
        }


	printf("\n\n\nTOTAL TIME TAKEN\t%10.10f s\n\n\n",ttime/1000000000);

	return(0);

}
