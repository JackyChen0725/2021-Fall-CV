#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>



using namespace cv;
using namespace std;

int findMax(Mat input)
{
	int max = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uint8_t>(i, j) > max)
				max = input.at<uint8_t>(i, j);
		}
	}
	return max;
}

int findMin(Mat input)
{
	int min = 255;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uint8_t>(i, j) < min)
				min = input.at<uint8_t>(i, j);
		}
	}
	return min;
}
Mat grayscaleDilation(Mat input, Mat kernel)
{
	Mat result;
	result = Mat::zeros(input.rows, input.cols, CV_8UC1);
	int kernelSize = kernel.rows;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			Mat temp;
			if ((i + kernelSize) < input.rows && (j + kernelSize) < input.cols) {
				temp = input(cv::Rect(j, i, kernelSize, kernelSize));
			}
			else if ((i + kernelSize) >= input.rows && (j + kernelSize) < input.cols) {
				temp = input(cv::Rect(j, i, kernelSize, input.rows - i));
			}
			else if ((i + kernelSize) < input.rows && (j + kernelSize) >= input.cols) {
				temp = input(cv::Rect(j, i, input.cols - j, kernelSize));
			}
			else
				temp = input(cv::Rect(j, i, input.cols - j, input.rows - i));
			int max = findMax(temp);
			result.at<uint8_t>(i, j) = max;
		}
	}
	return result;
}

Mat grayscaleErosion(Mat input, Mat kernel)
{
	Mat result;
	result = Mat::zeros(input.rows, input.cols, CV_8UC1);
	int kernelSize = kernel.rows;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			Mat temp;
			if ((i + kernelSize) < input.rows && (j + kernelSize) < input.cols) {
				temp = input(cv::Rect(j, i, kernelSize, kernelSize));
			}
			else if ((i + kernelSize) >= input.rows && (j + kernelSize) < input.cols) {
				temp = input(cv::Rect(j, i, kernelSize, input.rows - i));
			}
			else if ((i + kernelSize) < input.rows && (j + kernelSize) >= input.cols) {
				temp = input(cv::Rect(j, i, input.cols - j, kernelSize));
			}
			else
				temp = input(cv::Rect(j, i, input.cols - j, input.rows - i));
			int min = findMin(temp);
			result.at<uint8_t>(i, j) = min;
		}
	}
	return result;
}
int main()
{
	/*read image*/
	Mat image = imread("lena.bmp", 0);
	Mat img1;
	image.copyTo(img1);

	/*kernel initialize*/
	Mat kernel = (Mat_<uint8_t>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0);

	/*grayscale_dilation*/
	Mat dilation_image = grayscaleDilation(img1, kernel);
	imshow("dilation", dilation_image);
	waitKey(0);

	/*grayscale_erosion*/
	Mat erosion_image = grayscaleErosion(img1, kernel);
	imshow("erosion", erosion_image);
	waitKey(0);

	/*grayscale_opening*/
	Mat opening_image = grayscaleDilation(erosion_image, kernel);
	imshow("opening", opening_image);
	waitKey(0);

	/*grayscale_closing*/
	Mat closing_image = grayscaleErosion(dilation_image, kernel);
	imshow("closing", closing_image);
	waitKey(0);

	/*write image*/
	imwrite("grayscale_dilation.jpg", dilation_image);
	imwrite("grayscale_erosion.jpg", erosion_image);
	imwrite("grayscale_opening.jpg", opening_image);
	imwrite("grayscale_closing.jpg", closing_image);
	return 0;

}


