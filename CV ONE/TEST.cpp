#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

Mat src,src_gray,image,src_gray_prev,src1,src_gray1,copy,copy1,frames,copy2;
int maxCorners = 23;
RNG rng(12345);
CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;
void detect(IplImage *img);
void detect_first(IplImage *img1);
int number = 0;
int corners_in_this_part = 0;
int corners_in_temp = 0;
vector<Point2f> corners,corners_prev,corners_temp;
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
bool useHarrisDetector = false;
double k = 0.04;
vector<uchar> status;
vector<float> err;
int frame_number = 0;
int rx[10000];
int ry[10000];
int rh[10000];
int rw[10000];
 
int main(int argc, char** argv)
{
  CvCapture *capture;
  IplImage  *frame;
	IplImage  *test;
  int input_resize_percent = 100;
	
	  
	
  

  if(argc == 4)
  {
    input_resize_percent = atoi(argv[3]);
    std::cout << "Resizing to: " << input_resize_percent << "%" << std::endl;
  }

	cascade = (CvHaarClassifierCascade*) cvLoad("D:/cars3.xml", 0, 0, 0);
  storage = cvCreateMemStorage(0);
  capture = cvCaptureFromAVI("D:/video2.avi");

  assert(cascade && storage && capture);
cvNamedWindow("video", 1);
//cvNamedWindow("video1",1);
	
  IplImage* frame1 = cvQueryFrame(capture);
  frame = cvCreateImage(cvSize((int)((frame1->width*input_resize_percent)/100) , (int)((frame1->height*input_resize_percent)/100)), frame1->depth, frame1->nChannels);
	
	cvResize(frame1, frame);
	const int KEY_SPACE  = 32;
  const int KEY_ESC    = 27;

	detect_first(frame1);	
//	cvShowImage("sdaad",frame1);	
  int key = 0;
  do
  {
		src_gray_prev = src_gray.clone();
	corners_prev = corners;
	 
		frame1 = cvQueryFrame(capture);
		

		if(!frame1)
      break;
	frame_number = frame_number + 1;
		if(frame_number % 25 == 0)
		{
			Mat framess(frame1);
			src = framess.clone();
			cvtColor( src, src_gray, CV_BGR2GRAY ); 	
			Mat copy;
  		copy = src.clone();
		
		goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               Mat(),
               blockSize,
               useHarrisDetector,
               k );

			calcOpticalFlowPyrLK(src_gray_prev, src_gray, corners_prev, corners, status, err);
			//cout<<"** Number of corners detected: "<<corners.size()<<endl;
  		int r = 10;
  		for( int i = 0; i < corners.size(); i++ )
    	 { circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255),
              rng.uniform(0,255)), -1, 8, 0 ); 

				//cout << corners_prev[i]<<endl;
				//cout << status << endl;
			
		}
		//cvResize(frame1, frame);

		IplImage test3 = copy;
		IplImage* test4 = &test3;
	
//		cvShowImage("video",frame1);
		detect(test4);
		key = cvWaitKey(10);

    if(key == KEY_SPACE)
      key = cvWaitKey(0);

    if(key == KEY_ESC)
      break;
}
  }while(1);

	cvWaitKey(10);
  cvDestroyAllWindows();
  cvReleaseImage(&frame);
  cvReleaseCapture(&capture);
  cvReleaseHaarClassifierCascade(&cascade);
  cvReleaseMemStorage(&storage);
	
  return 0;
}


void detect(IplImage *img)
{
	Mat framesss(img);
	src = framesss.clone();
	cvtColor( src, src_gray, CV_BGR2GRAY );
	cv::Mat mask2 = cv::Mat::zeros(src.size(), CV_8UC1);  //NOTE: using the type explicitly 
		
	
	CvSize img_size = cvGetSize(img);
  CvSeq *object = cvHaarDetectObjects(
    img,
    cascade,
    storage,
    1.1, //1.1,//1.5, //-------------------SCALE FACTOR
    1, //2        //------------------MIN NEIGHBOURS
    0, //CV_HAAR_DO_CANNY_PRUNING
    cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
    img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
    );
	if (frame_number%35==0)
	number = number + object->total;
  std::cout << "Cars in this frame: " << object->total << "Total cars detected=" << "  " << number <<"  "<<frame_number<< std::endl;

	char a[100] = "images/frame";
  char c[100] = ".png";
	char l[100] = "_";
  	char d[100];
  	
  for(int i = 0 ; i < ( object ? object->total : 0 ) ; i++)
  {
    CvRect *r = (CvRect*)cvGetSeqElem(object, i);
    cvRectangle(img,
      cvPoint(r->x, r->y),
      cvPoint(r->x + r->width, r->y + r->height),
      CV_RGB(255, 0, 0), 2, 8, 0);
			rx[number-i-1] = r->x;
			ry[number-i-1] = r->y;
			rw[number-i-1] = r->width;
			rh[number-i-1] = r->height;
  }

	for(int i = 0 ; i < number ; i++)
		{		
			cv::Mat roi(mask2, cv::Rect(rx[i]+10,ry[i]+10,rw[i]-10,rh[i]-10));
			roi = cv::Scalar(255, 255, 255);
		}		
			//Mat copy1;
  	copy2 = src.clone();		
		goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               mask2,
               blockSize,
               useHarrisDetector,
               k );

		int rad = 10;
  	for( int i = 0; i < corners.size(); i++ )
  	   { circle( copy2, corners[i], rad, Scalar(rng.uniform(0,255), rng.uniform(0,255),
  	            rng.uniform(0,255)), -1, 8, 0 );
				}

		IplImage test5 = copy2;
  	IplImage* test6 = &test5;

    cvShowImage("video", test6);

//    cvShowImage("video1", img);
}


void detect_first(IplImage *img1)
{
  CvSize img_size = cvGetSize(img1);
  CvSeq *object = cvHaarDetectObjects(
    img1,
    cascade,
    storage,
    1.1, //1.1,//1.5, //-------------------SCALE FACTOR
    1, //2        //------------------MIN NEIGHBOURS
    0, //CV_HAAR_DO_CANNY_PRUNING
    cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
    img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
    );

	number = number + object->total;
  std::cout << "Cars in this frame: " << object->total << "Total cars detected=" << "  " << number << std::endl;
	Mat frames(img1);
//	Mat copy1;	
  for(int i = 0 ; i < ( object ? object->total : 0 ) ; i++)
  {
    CvRect *r = (CvRect*)cvGetSeqElem(object, i);
    cvRectangle(img1,
      cvPoint(r->x, r->y),
      cvPoint(r->x + r->width, r->y + r->height),
      CV_RGB(255, 0, 0), 2, 8, 0);
			rx[i] = r->x;
			ry[i] = r->y;
			rw[i] = r->width;
			rh[i] = r->height;
//		Mat frames(img);
/*	
		src = frames.clone();
		cvtColor( src, src_gray, CV_BGR2GRAY );
		cv::Mat mask1 = cv::Mat::zeros(src.size(), CV_8UC1);  //NOTE: using the type explicitly
		cv::Mat roi(mask1, cv::Rect(rx,ry,rw,rh));
		roi = cv::Scalar(255, 255, 255);
		//Mat copy1;
  	copy1 = src.clone();		
		goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               mask1,
               blockSize,
               useHarrisDetector,
               k );
		
				
		cout<<"** Number of corners detected: "<<corners.size()<<endl;
/*		
		corners_in_this_part = corners.size();
		corners_in_temp = corners_temp.size();
//		cout << " fasdafs " << corners_in_this_part << endl;
	//	cout << " fasdafssssssssssss " << corners_in_temp << endl;
		if(corners_in_temp == 0)
		{
//			for(int i = 0;i<corners_in_this_part;i++)
//			{
//				cout << "entered for loop " << endl;
//				cout << corners[i] << endl;
//				cout << corners_temp[i] << endl;
//			cout << "kill" << endl;
//				corners_temp[i] = corners[i];
//				cout << "exitted for loop " << endl;
//			}
				cout << "for first" <<endl;
				corners_temp = corners; 
		}
		else
		{
			for(int i = corners_in_temp-1;i<corners_in_temp + corners_in_this_part;i++)
			{
				cout << "entered for loop for second " << endl;
				cout << "i = " << i << endl;
				corners_temp[i] = corners[i-corners_in_temp+1];
				cout << corners_temp[i] << endl;
				cout << "sfdfdsasfd = " << corners_temp.size() << endl;
			}
		}
		
*/
		
	
				//cout << corners[i]<<endl;
			} 

			src = frames.clone();
		cvtColor( src, src_gray, CV_BGR2GRAY );
		cv::Mat mask1 = cv::Mat::zeros(src.size(), CV_8UC1);  //NOTE: using the type explicitly
		for(int i = 0 ; i < ( object ? object->total : 0 ) ; i++)
		{		
			cv::Mat roi(mask1, cv::Rect(rx[i]+10,ry[i]+10,rw[i]-10,rh[i]-10));
			roi = cv::Scalar(255, 255, 255);
		}		
			//Mat copy1;
  	copy1 = src.clone();		
		goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               mask1,
               blockSize,
               useHarrisDetector,
               k );

		int rad = 10;
  	for( int i = 0; i < corners.size(); i++ )
  	   { circle( copy1, corners[i], rad, Scalar(rng.uniform(0,255), rng.uniform(0,255),
  	            rng.uniform(0,255)), -1, 8, 0 );  
}

		
		
		
  	IplImage test1 = copy1;
  	IplImage* test2 = &test1;

    cvShowImage("video", test2);
//  imshow("dfasadf",frames);
//		cvShowImage("video1",img1);
}


