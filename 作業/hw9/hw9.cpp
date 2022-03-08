#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat padding(Mat input)
{
	Mat padded_image(input.rows + 2, input.cols + 2, CV_8UC1);

	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			padded_image.at<uint8_t>(i + 1, j + 1) = input.at<uint8_t>(i, j);

	for (int j = 1; j < padded_image.cols - 1; j++) {
		padded_image.at<uint8_t>(0, j) = input.at<uint8_t>(0, j - 1);
		padded_image.at<uint8_t>(padded_image.rows - 1, j) = input.at<uint8_t>(input.rows - 1, j - 1);
	}

	for (int i = 1; i < padded_image.rows - 1; i++) {
		padded_image.at<uint8_t>(i, 0) = input.at<uint8_t>(i - 1, 0);
		padded_image.at<uint8_t>(i, padded_image.cols - 1) = input.at<uint8_t>(i - 1, input.cols - 1);
	}

	padded_image.at<uint8_t>(0, 0) = input.at<uint8_t>(0, 0);
	padded_image.at<uint8_t>(padded_image.rows - 1, 0) = input.at<uint8_t>(input.rows - 1, 0);
	padded_image.at<uint8_t>(0, padded_image.cols - 1) = input.at<uint8_t>(0, input.cols - 1);
	padded_image.at<uint8_t>(padded_image.rows - 1, padded_image.cols - 1) = input.at<uint8_t>(input.rows - 1, input.cols - 1);

	return padded_image;
}

double matrixSum(Mat kernel, Mat mask)
{
	double total = 0;
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			total += kernel.at<uint8_t>(i, j) * mask.at<double>(i, j);
		}
	}
	return total;
}

Mat Roberts_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);
	
	Mat mask[2];
	mask[0] = (Mat_<double>(2, 2) << //r1
		-1, 0,
		0, 1);
	mask[1] = (Mat_<double>(2, 2) << //r2
		0, -1,
		1, 0);
	
	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j, i, 2, 2));
			double value = 0;
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				double temp = matrixSum(kernel, mask[n]);
				value += pow(temp, 2);
			}
			if (sqrt(value) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

Mat Prewitt_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[2];
	mask[0] = (Mat_<double>(3, 3) << //p1
		-1, -1, -1,
		 0,  0,  0,
		 1,  1,  1);
	mask[1] = (Mat_<double>(3, 3) << //p2
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = 0;
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				double temp = matrixSum(kernel, mask[n]);
				value += pow(temp, 2);
			}
			if (sqrt(value) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

Mat Sobel_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[2];
	mask[0] = (Mat_<double>(3, 3) << //s1
		-1, -2, -1,
		 0,  0,  0,
		 1,  2,  1);
	mask[1] = (Mat_<double>(3, 3) << //s2
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = 0;
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				double temp = matrixSum(kernel, mask[n]);
				value += pow(temp, 2);
			}
			if (sqrt(value) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

Mat Frei_and_Chen_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[2];
	mask[0] = (Mat_<double>(3, 3) << //f1
		-1, -sqrt(2), -1,
		0,         0,  0,
		1,   sqrt(2),  1);
	mask[1] = (Mat_<double>(3, 3) << //f2
		      -1, 0,       1,
		-sqrt(2), 0, sqrt(2),
		      -1, 0,      1);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = 0;
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				double temp = matrixSum(kernel, mask[n]);
				value += pow(temp, 2);
			}
			if (sqrt(value) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

double findMax(double array[], int arraysize)
{
	for (int i = arraysize - 2; i >= 0; i--) {
		for (int j = 0; j <= i; j++) {
			if (array[j] > array[j + 1]) {
				double temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
	return array[arraysize - 1];
}

Mat Kirsch_compass_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[8];
	mask[0] = (Mat_<double>(3, 3) << //k0
		-3, -3, 5,
		-3,  0, 5,
		-3, -3, 5);
	mask[1] = (Mat_<double>(3, 3) << //k1
		-3,  5,  5,
		-3,  0,  5,
		-3, -3, -3);
	mask[2] = (Mat_<double>(3, 3) << //k2
		 5,  5,  5,
		-3,  0, -3,
		-3, -3, -3);
	mask[3] = (Mat_<double>(3, 3) << //k3
		 5,  5, -3,
		 5,  0, -3,
		-3, -3, -3);
	mask[4] = (Mat_<double>(3, 3) << //k4
		 5, -3, -3,
		 5,  0, -3,
		 5, -3, -3);
	mask[5] = (Mat_<double>(3, 3) << //k5
		-3, -3, -3,
		 5,  0, -3,
		 5,  5, -3);
	mask[6] = (Mat_<double>(3, 3) << //k6
		-3, -3, -3,
		-3,  0, -3,
		 5,  5,  5);
	mask[7] = (Mat_<double>(3, 3) << //k7
		-3, -3, -3,
		-3,  0,  5,
		-3,  5,  5);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value[8] = { 0 };
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				value[n] = matrixSum(kernel, mask[n]);
			}
			if (findMax(value, 8) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

Mat Robinson_compass_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[8];
	mask[0] = (Mat_<double>(3, 3) << //r0
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	mask[1] = (Mat_<double>(3, 3) << //r1
		 0,  1, 2,
		-1,  0, 1,
		-2, -1, 0);
	mask[2] = (Mat_<double>(3, 3) << //r2
		 1,  2,  1,
		 0,  0,  0,
		-1, -2, -1);
	mask[3] = (Mat_<double>(3, 3) << //r3
		 2,  1,  0,
		 1,  0, -1,
		 0, -1, -2);
	mask[4] = -mask[0]; //r4
	mask[5] = -mask[1]; //r5
	mask[6] = -mask[2]; //r6
	mask[7] = -mask[3]; //r7

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value[8] = { 0 };
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				value[n] = matrixSum(kernel, mask[n]);
			}
			if (findMax(value, 8) >= threshold)
				result.at<uint8_t>(i - 1, j - 1) = 0;
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}

Mat Nevatia_Babu_operator(Mat padded, Mat orig, int threshold)
{
	Mat result;
	orig.copyTo(result);

	Mat mask[6];
	mask[0] = (Mat_<double>(5, 5) << //0度
		 100,  100,  100,  100,  100,
		 100,  100,  100,  100,  100,
		   0,    0,    0,    0,    0,
		-100, -100, -100, -100, -100,
		-100, -100, -100, -100, -100);
	mask[1] = (Mat_<double>(5, 5) << //30度
		 100,  100,  100,  100,  100,
		 100,  100,  100,   78,  -32,
		 100,   92,    0,  -92, -100,
		  32,  -78, -100, -100, -100,
		-100, -100, -100, -100, -100);
	mask[2] = (Mat_<double>(5, 5) << //60度
		100, 100,  100,   32, -100,
		100, 100,   92,  -78, -100,
		100, 100,    0, -100, -100,
		100,  78,  -92, -100, -100,
		100, -32, -100, -100, -100);
	mask[3] = (Mat_<double>(5, 5) << //-90度
		-100, -100, 0, 100, 100,
		-100, -100, 0, 100, 100,
		-100, -100, 0, 100, 100,
		-100, -100, 0, 100, 100,
		-100, -100, 0, 100, 100);
	mask[4] = (Mat_<double>(5, 5) << //-60度
		-100,   32,  100, 100, 100,
		-100,  -78,   92, 100, 100,
		-100, -100,    0, 100, 100,
		-100, -100,  -92,  78, 100,
		-100, -100, -100, -32, 100);
	mask[5] = (Mat_<double>(5, 5) << //-30度
		 100,  100,  100,  100,  100,
		 -32,   78,  100,  100,  100,
		-100,  -92,    0,   92,  100,
		-100, -100, -100,  -78,   32,
		-100, -100, -100, -100, -100);

	for (int i = 2; i < padded.rows - 2; i++) {
		for (int j = 2; j < padded.cols - 2; j++) {
			Mat kernel = padded(cv::Rect(j - 2, i - 2, 5, 5));
			double value[6] = { 0 };
			for (int n = 0; n < sizeof(mask) / sizeof(*mask); n++) {
				value[n] = matrixSum(kernel, mask[n]);
			}
			if (findMax(value, 6) >= threshold)
				result.at<uint8_t>(i - 2, j - 2) = 0;
			else
				result.at<uint8_t>(i - 2, j - 2) = 255;
		}
	}
	return result;
}
int main()
{
	//read image
	Mat image = imread("lena.bmp", 1);
	cvtColor(image, image, COLOR_RGB2GRAY);

	//Roberts operator
	Mat Roberts_image = Roberts_operator(padding(image), image, 30);
	imshow("Roberts_image", Roberts_image);
	waitKey(0);
	
	//Prewitt operator
	Mat Prewitt_image = Prewitt_operator(padding(image), image, 24);
	imshow("Prewitt_image", Prewitt_image);
	waitKey(0);

	//Sobel operator
	Mat Sobel_image = Sobel_operator(padding(image), image, 38);
	imshow("Sobel_image", Sobel_image);
	waitKey(0);

	//Frei and Chen gradient operator
	Mat Frei_and_Chen_image = Frei_and_Chen_operator(padding(image), image, 30);
	imshow("Frei_and_Chen_image", Frei_and_Chen_image);
	waitKey(0);

	//Kirsch compass operator
	Mat Kirsch_image = Kirsch_compass_operator(padding(image), image, 135);
	imshow("Kirsch_image", Kirsch_image);
	waitKey(0);

	//Robinson compass operator
	Mat Robinson_image = Robinson_compass_operator(padding(image), image, 43);
	imshow("Robinson_image", Robinson_image);
	waitKey(0);

	//Nevatia-Babu 5x5 operator
	Mat Nevatia_Babu_image = Nevatia_Babu_operator(padding(padding(image)), image, 12500);
	imshow("Nevatia_Babu_image", Nevatia_Babu_image);
	waitKey(0);

	imwrite("Roberts_image.jpg", Roberts_image);
	imwrite("Prewitt_image.jpg", Prewitt_image);
	imwrite("Sobel_image.jpg", Sobel_image);
	imwrite("Frei_and_Chen_image.jpg", Frei_and_Chen_image);
	imwrite("Kirsch_image.jpg", Kirsch_image);
	imwrite("Robinson_image.jpg", Robinson_image);
	imwrite("Nevatia_Babu_image.jpg", Nevatia_Babu_image);
	return 0;
}