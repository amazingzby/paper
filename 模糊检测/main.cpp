/*
 * main.cpp
 *
 *  Created on: 2019���2���23���
 *      Author: swj
 */

#include <opencv.hpp>
#include <stdio.h>
#include <string>
#include <iostream>

int min(int a,int b)
{
	return ((a<b)?a:b);
}

int max(int a,int b)
{
	return ((a>b)?a:b);
}

void calculateGradMagDist(uchar *data,int width,int height, float *magVec, int length)
{
	for(int i=0;i<height;++i)
	{
		for(int j=0;j<width-1;++j)
		{
			int Gy=(int)*(data+i*width+j+1)-(int)*(data+i*width+j);
			Gy=min(Gy,255);//clip
			magVec[Gy+255]+=1.0;
		}
	}

	//normalization
	int pixCnt=(width-1)*(height-1);
	for(int i=0;i<length;++i)
	{
		magVec[i]/=pixCnt;
	}
}

float calculateStdVariance(float *prob, float *value, int length)
{
	//EX
	float ex=0;
	for(int i=0;i<length;++i)
	{
		ex+=prob[i]*value[i];
	}

	//variance
	float variance=0;
	for(int i=0;i<length;++i)
	{
		variance+=prob[i]*(value[i]-ex)*(value[i]-ex);
	}
	variance*=length/(length-1);

	return sqrt(variance);
}

float calculateGradientStdVariance(uchar *data,int width,int height)
{
	int length=255*2+1;
	float magVec[length]={0};
	calculateGradMagDist(data,width,height,magVec,length);

	float value[length];
	for(int i=0;i<length;++i)
	{
		value[i]=i-255.0;
	}

	return calculateStdVariance(magVec,value,length);
}

 float calculateContrast(uchar *data,int width,int height)
{
	int maxVal=0;
	int minVal=255;

	for(int i=0;i<height;++i)
	{
		for(int j=0;j<width;++j)
		{
			int val=(int)*(data+i*width+j);
			maxVal=max(val,maxVal);
			minVal=min(val,minVal);
		}
	}

	return (1.0*(maxVal-minVal)/(maxVal+minVal+1e-5));
}

float calculateMeanContrast(uchar *data,int width,int height,int blockSz=10)
{
	//1.segment into blockSzxblockSz regions
	int blockDimX=(height-1)/blockSz+1;
	int blockDimY=(width-1)/blockSz+1;

	float sum=0;
	for(int blk_x=0;blk_x<blockDimX;++blk_x)
	{
		for(int blk_y=0;blk_y<blockDimY;++blk_y)
		{
			//2.cpy block data
			int start_x=blockSz*blk_x;
			int end_x=start_x+blockSz-1;
			end_x=min(end_x,(height-1));
			int blkH=end_x-start_x+1;

			int start_y=blockSz*blk_y;
			int end_y=start_y+blockSz-1;
			end_y=min(end_y,(width-1));
			int blkW=end_y-start_y+1;

			uchar *blkData=new uchar[blkW*blkH];
			for(int i=start_x;i<=end_x;++i)
			{
				memcpy(blkData+(i-start_x)*blkW,data+i*width+start_y,sizeof(uchar)*blkW);
			}

			//3.calculate contrast and accumulate
			float contrast=calculateContrast(blkData,blkW,blkH);
			sum+=contrast;

			delete[] blkData;
		}
	}

	return sum/(blockDimX*blockDimY);
}


FILE *fp;
int imgCnt=0;
bool isBlur(cv::Mat oriImg,cv::Size resizeSz,float threshold)
{
	//1.calculate contrast
	float contrast=calculateMeanContrast(oriImg.data,oriImg.cols,oriImg.rows,min(oriImg.cols,oriImg.rows)/10);

	//2.calculate gradient std with resized img
	cv::Mat img(resizeSz,CV_8UC1);
	cv::resize(oriImg,img,resizeSz,cv::INTER_CUBIC);
	float std=calculateGradientStdVariance(img.data,resizeSz.width,resizeSz.height);

	//3.judge blur
	bool blur=(contrast*std<threshold);
	fprintf(fp,"%d,%f,%f,%f,%d\n",imgCnt,contrast,std,contrast*std,(int)blur);

	return blur;
}

int main()
{
	std::string filePath="./face/";
	std::string blurPath="./blur/";
	std::string clarityPath="./clarity/";

	fp=fopen("./face_results.csv","w");
	if(fp!=NULL)
	{
		fprintf(fp,"file name,contrast,gradient std,product,blur\n");
		for(int i=0;i<61776;++i)
		//for(int i=0;i<999;++i)
		{
			//import
			char fileName[20];
			sprintf(fileName,"%d.jpg",i);
			//sprintf(fileName,"%05d.jpg",i);
			cv::Mat oriImg=cv::imread(filePath+fileName, cv::IMREAD_GRAYSCALE);

			bool blur=isBlur(oriImg,cv::Size(128,128),6.0);
			if(blur)
			{
				rename((filePath+fileName).c_str(),(blurPath+fileName).c_str());
			}
			else
			{
				rename((filePath+fileName).c_str(),(clarityPath+fileName).c_str());
			}

			if(i%100==0)
			{
				std::cout<<"img "<<i<<"\n";
			}

			++imgCnt;
		}
		fclose(fp);
	}

	std::cout<<"done\n";
	return 0;
}
