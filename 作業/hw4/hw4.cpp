#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#define kernelSize kernel.rows

using namespace cv;
using namespace std;


Mat binary(Mat input)
{
	Mat result;
	input.copyTo(result);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uint8_t>(i, j) < 128)
				result.at<uint8_t>(i, j) = 0;
			else
				result.at<uint8_t>(i, j) = 1;
		}
	}
	return result;
}

Mat dilation(Mat input, Mat kernel)
{
	Mat dilation_image = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = (kernelSize - 1) / 2; i < input.rows - (kernelSize - 1) / 2; i++) {
		for (int j = (kernelSize - 1) / 2; j < input.cols - (kernelSize - 1) / 2; j++) {
			if (input.at<uint8_t>(i, j) == 1) {
				Mat dilation_temp = dilation_image(cv::Rect(j - (kernelSize - 1) / 2, i - (kernelSize - 1) / 2, kernelSize, kernelSize));
				add(dilation_temp, kernel, dilation_temp);
			}
		}
	}
	for (int i = 0; i < dilation_image.rows; i++) {
		for (int j = 0; j < dilation_image.cols; j++) {
			if (dilation_image.at<uint8_t>(i, j) != 0)
				dilation_image.at<uint8_t>(i, j) = 255;
		}
	}
	return dilation_image;
}

Mat erosion(Mat input, Mat kernel)
{
	Mat erosion_image = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = (kernelSize - 1) / 2; i < input.rows - (kernelSize - 1) / 2; i++) {
		for (int j = (kernelSize - 1) / 2; j < input.cols - (kernelSize - 1) / 2; j++) {
			Mat erosion_temp = input(cv::Rect(j - (kernelSize - 1) / 2, i - (kernelSize - 1) / 2, kernelSize, kernelSize));
			Mat erosion_result = erosion_temp.mul(kernel);
			Mat diff = erosion_result != kernel;
			bool equal = cv::countNonZero(diff) == 0;
			if (equal) {
				for (int a = i - (kernelSize - 1) / 2; a < kernelSize; a++) {
					for (int b = j - (kernelSize - 1) / 2; b < kernelSize; b++) {
						erosion_image.at<uint8_t>(a, b) = 0;
					}
				}
				erosion_image.at<uint8_t>(i, j) = 255;
			}
		}
	}
	return erosion_image;
}
int main()
{
	/*read image*/
	Mat img = imread("lena.bmp", IMREAD_UNCHANGED);

	/*binarize*/
	Mat thresh_image;
	img.copyTo(thresh_image);
	thresh_image = binary(thresh_image);
	
	/*initialize kernel*/
	Mat kernel = (Mat_<uint8_t>(5, 5) <<
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0);

	Mat J_kernel = (Mat_<uint8_t>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		1, 1, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 0, 0, 0);

	Mat K_kernel = (Mat_<uint8_t>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 1, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0);

	/*dilation*/
	Mat dilation_image = dilation(thresh_image, kernel);
	imshow("dilation", dilation_image);
	waitKey(0);

	/*erosion*/
	Mat erosion_image = erosion(thresh_image, kernel);
	imshow("erosion", erosion_image);
	waitKey(0);

	/*opening*/
	Mat opening_image = dilation(binary(erosion_image), kernel);
	imshow("opening", opening_image);
	waitKey(0);

	/*closing*/
	Mat closing_image = erosion(binary(dilation_image), kernel);
	imshow("closing", closing_image);
	waitKey(0);

	/*hit-and-miss*/
	Mat complement = Mat::zeros(thresh_image.rows, thresh_image.cols, CV_8UC1);
	for (int i = 0; i < thresh_image.rows; i++) {
		for (int j = 0; j < thresh_image.cols; j++) {
			if (thresh_image.at<uint8_t>(i, j) == 1)
				complement.at<uint8_t>(i, j) = 0;
			else
				complement.at<uint8_t>(i, j) = 1;
		}
	}

	Mat erosion_by_J = erosion(thresh_image, J_kernel);
	Mat erosion_by_K = erosion(complement, K_kernel);
	Mat hit_and_miss = Mat::zeros(thresh_image.rows, thresh_image.cols, CV_8UC1);
	for (int i = 0; i < hit_and_miss.rows; i++) {
		for (int j = 0; j < hit_and_miss.cols; j++) {
			if (erosion_by_J.at<uint8_t>(i, j) && erosion_by_K.at<uint8_t>(i, j))
				hit_and_miss.at<uint8_t>(i, j) = 255;
			else
				hit_and_miss.at<uint8_t>(i, j) = 0;
		}
	}
	imshow("hit and miss", hit_and_miss);
	waitKey(0);

	/*output image*/
	imwrite("dilation.jpg", dilation_image);
	imwrite("erosion.jpg", erosion_image);
	imwrite("opening.jpg", opening_image);
	imwrite("closing.jpg", closing_image);
	imwrite("hit-and-miss.jpg", hit_and_miss);
	return 0;
}

