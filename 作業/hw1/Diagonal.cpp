#include<cstdio>
#include<cstdlib>
#include<cmath>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main()
{
	// read image 
	Mat img1 = imread("lena.bmp");
	Mat img2 = imread("lena.bmp");

	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
		{
			img2.at<Vec3b>(i, j) = img1.at<Vec3b>(j,i);
		}

	// show image
	imshow("lena.bmp", img2);

	// write image 
	imwrite("diagonal.jpg", img2);

	waitKey(0);
	return 0;
}