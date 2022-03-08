#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

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

Mat signed_padding(Mat input)
{
	Mat signed_padded_image(input.rows + 2, input.cols + 2, CV_8SC1);

	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			signed_padded_image.at<int8_t>(i + 1, j + 1) = input.at<int8_t>(i, j);

	for (int j = 1; j < signed_padded_image.cols - 1; j++) {
		signed_padded_image.at<int8_t>(0, j) = input.at<int8_t>(0, j - 1);
		signed_padded_image.at<int8_t>(signed_padded_image.rows - 1, j) = input.at<int8_t>(input.rows - 1, j - 1);
	}

	for (int i = 1; i < signed_padded_image.rows - 1; i++) {
		signed_padded_image.at<int8_t>(i, 0) = input.at<int8_t>(i - 1, 0);
		signed_padded_image.at<int8_t>(i, signed_padded_image.cols - 1) = input.at<int8_t>(i - 1, input.cols - 1);
	}

	signed_padded_image.at<int8_t>(0, 0) = input.at<int8_t>(0, 0);
	signed_padded_image.at<int8_t>(signed_padded_image.rows - 1, 0) = input.at<int8_t>(input.rows - 1, 0);
	signed_padded_image.at<int8_t>(0, signed_padded_image.cols - 1) = input.at<int8_t>(0, input.cols - 1);
	signed_padded_image.at<int8_t>(signed_padded_image.rows - 1, signed_padded_image.cols - 1) = input.at<int8_t>(input.rows - 1, input.cols - 1);

	return signed_padded_image;
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

Mat Laplacin_mask_1(Mat padded, int threshold)
{
	Mat output = Mat::zeros(512, 512, CV_8SC1);
	
	Mat mask = (Mat_<double>(3, 3) <<
		0,  1, 0,
		1, -4, 1,
		0,  1, 0);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = matrixSum(kernel, mask);
			if (value >= threshold)
				output.at<int8_t>(i - 1, j - 1) = 1;
			else if (value <= -threshold)
				output.at<int8_t>(i - 1, j - 1) = -1;
			else
				output.at<int8_t>(i - 1, j - 1) = 0;
		}
	}
	return output;
}

Mat Laplacin_mask_2(Mat padded, int threshold)
{
	Mat output = Mat::zeros(512, 512, CV_8SC1);

	Mat mask = ((double)1.0/3) * (Mat_<double>(3, 3) <<
		1,  1, 1,
		1, -8, 1,
		1,  1, 1);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = matrixSum(kernel, mask);
			if (value >= threshold)
				output.at<int8_t>(i - 1, j - 1) = 1;
			else if (value <= -threshold)
				output.at<int8_t>(i - 1, j - 1) = -1;
			else
				output.at<int8_t>(i - 1, j - 1) = 0;
		}
	}
	return output;
}

Mat minimum_variance_Laplacin_mask(Mat padded, int threshold)
{
	Mat output = Mat::zeros(512, 512, CV_8SC1);

	Mat mask = 1.0 / 3 * (Mat_<double>(3, 3) <<
		 2, -1,  2,
		-1, -4, -1,
		 2, -1,  2);

	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			Mat kernel = padded(cv::Rect(j - 1, i - 1, 3, 3));
			double value = matrixSum(kernel, mask);
			if (value >= threshold)
				output.at<int8_t>(i - 1, j - 1) = 1;
			else if (value <= -threshold)
				output.at<int8_t>(i - 1, j - 1) = -1;
			else
				output.at<int8_t>(i - 1, j - 1) = 0;
		}
	}
	return output;
}

Mat Laplacin_of_Gaussian_mask(Mat padded, int threshold)
{
	Mat output = Mat::zeros(512, 512, CV_8SC1);

	Mat mask = (Mat_<double>(11, 11) <<
		 0,  0,   0,  -1,   -1,  -2,  -1,  -1,   0,  0,  0,
		 0,  0,  -2,  -4,   -8,  -9,  -8,  -4,  -2,  0,  0,
		 0, -2,  -7, -15,  -22, -23, -22, -15,  -7, -2,  0,
		-1, -4, -15, -24,  -14,  -1, -14, -24, -15, -4, -1,
		-1, -8, -22, -14,   52, 103,  52, -14, -22, -8, -1,
		-2, -9, -23,  -1,  103, 178, 103,  -1, -23, -9, -2,
		-1, -8, -22, -14,   52, 103,  52, -14, -22, -8, -1,
		-1, -4, -15, -24,  -14,  -1, -14, -24, -15, -4, -1,
		 0, -2,  -7, -15,  -22, -23, -22, -15,  -7, -2,  0,
		 0,  0,  -2,  -4,   -8,  -9,  -8,  -4,  -2,  0,  0,
		 0,  0,   0,  -1,   -1,  -2,  -1,  -1,   0,  0,  0);

	for (int i = 5; i < padded.rows - 5; i++) {
		for (int j = 5; j < padded.cols - 5; j++) {
			Mat kernel = padded(cv::Rect(j - 5, i - 5, 11, 11));
			double value = matrixSum(kernel, mask);
			if (value >= threshold)
				output.at<int8_t>(i - 5, j - 5) = 1;
			else if (value <= -threshold)
				output.at<int8_t>(i - 5, j - 5) = -1;
			else
				output.at<int8_t>(i - 5, j - 5) = 0;
		}
	}
	return output;
}

Mat difference_of_Gaussian_mask(Mat padded, int threshold)
{
	Mat output = Mat::zeros(512, 512, CV_8SC1);

	Mat mask = (Mat_<double>(11, 11) <<
		-1,  -3,  -4, -6,  -7,  -8,  -7, -6,  -4,  -3, -1,
		-3,  -5,  -8,-11, -13, -13, -13,-11,  -8,  -5, -3,
		-4,  -8, -12,-16, -17, -17, -17,-16, -12,  -8, -4,
		-6, -11, -16,-16,   0,  15,   0,-16, -16, -11, -6,
		-7, -13, -17,  0,  85, 160,  85,  0, -17, -13, -7,
		-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8,
		-7, -13, -17,  0,  85, 160,  85,  0, -17, -13, -7,
		-6, -11, -16,-16,   0,  15,   0,-16, -16, -11, -6,
		-4,  -8, -12,-16, -17, -17, -17,-16, -12,  -8, -4,
		-3,  -5,  -8,-11, -13, -13, -13,-11,  -8,  -5, -3,
		-1,  -3,  -4, -6,  -7,  -8,  -7, -6,  -4,  -3, -1);

	for (int i = 5; i < padded.rows - 5; i++) {
		for (int j = 5; j < padded.cols - 5; j++) {
			Mat kernel = padded(cv::Rect(j - 5, i - 5, 11, 11));
			double value = matrixSum(kernel, mask);
			if (value >= threshold)
				output.at<int8_t>(i - 5, j - 5) = 1;
			else if (value <= -threshold)
				output.at<int8_t>(i - 5, j - 5) = -1;
			else
				output.at<int8_t>(i - 5, j - 5) = 0;
		}
	}
	return output;
}
Mat zerodetector(Mat input)
{
	Mat padded = signed_padding(input);
	
	Mat result = Mat(512, 512, CV_8UC1, Scalar(255));
	for (int i = 1; i < padded.rows - 1; i++) {
		for (int j = 1; j < padded.cols - 1; j++) {
			if (padded.at<int8_t>(i, j) == 1) {
				for (int dr = -1; dr <= 1; dr++) {
					for (int dc = -1; dc <= 1; dc++) {
						if (dr != 0 || dc != 0) {
							if (padded.at<int8_t>(i + dr, j + dc) == -1) {
								result.at<uint8_t>(i - 1, j - 1) = 0;
								break;
							}
							else
								result.at<uint8_t>(i - 1, j - 1) = 255;
						}
					}
					if (result.at<uint8_t>(i - 1, j - 1) == 0)
						break;
				}
			}
			else
				result.at<uint8_t>(i - 1, j - 1) = 255;
		}
	}
	return result;
}
int main()
{
	//read image
	Mat image = imread("lena.bmp", 1);
	cvtColor(image, image, COLOR_RGB2GRAY);

	//Laplacin mask 1
	Mat Laplacin_1 = Laplacin_mask_1(padding(image), 15);
	Mat output1 = zerodetector(Laplacin_1);
	imshow("output1", output1);
	waitKey(0);
	imwrite("Laplacin_1.jpg", output1);
	
	Mat Laplacin_2 = Laplacin_mask_2(padding(image), 15);
	Mat output2 = zerodetector(Laplacin_2);
	imshow("output2", output2);
	waitKey(0);
	imwrite("Laplacin_2.jpg", output2);

	Mat minimum_variance_Laplacin = minimum_variance_Laplacin_mask(padding(image), 20);
	Mat output3 = zerodetector(minimum_variance_Laplacin);
	imshow("output3", output3);
	waitKey(0);
	imwrite("minimum_variance_Laplacin.jpg", output3);

	Mat Laplacin_of_Gaussian = Laplacin_of_Gaussian_mask(padding(padding(padding(padding(padding(image))))), 3000);
	Mat output4 = zerodetector(Laplacin_of_Gaussian);
	imshow("output4", output4);
	waitKey(0);
	imwrite("Laplacin_of_Gaussian.jpg", output4);

	Mat difference_of_Gaussian = difference_of_Gaussian_mask(padding(padding(padding(padding(padding(image))))), 1);
	Mat output5 = zerodetector(difference_of_Gaussian);
	imshow("output5", output5);
	waitKey(0);
	imwrite("difference_of_Gaussian.jpg", output5);

	return 0;
}