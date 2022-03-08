#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;

Mat gaussianNoise(Mat input, int amplitude, int mean, int sigma)
{
	Mat noiseImage;
	input.copyTo(noiseImage);

	default_random_engine e;
	normal_distribution<double> distribution(mean, sigma);
	for (int i = 0; i < noiseImage.rows; i++) {
		for (int j = 0; j < noiseImage.cols; j++) {
			int noisePixel = noiseImage.at<uint8_t>(i, j) + distribution(e) * amplitude;
			if (noisePixel < 0)
				noisePixel = 0;
			if (noisePixel > 255)
				noisePixel = 255;
			noiseImage.at<uint8_t>(i, j) = noisePixel;
		}
	}
	return noiseImage;
}

Mat salt_and_pepper(Mat input, double threshold)
{
	Mat noiseImage;
	input.copyTo(noiseImage);

	default_random_engine e;
	uniform_real_distribution<double> distribution(0.0, 1.0);
	for (int i = 0; i < noiseImage.rows; i++) {
		for (int j = 0; j < noiseImage.cols; j++) {
			if (distribution(e) <= threshold)
				noiseImage.at<uint8_t>(i, j) = 0;
			else if (distribution(e) >= 1 - threshold)
				noiseImage.at<uint8_t>(i, j) = 255;
		}
	}

	return noiseImage;
}

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

Mat box_filter(Mat padded, Mat input, int boxSize)
{
	Mat result;
	input.copyTo(result);
	if (boxSize == 3) {
		for (int i = 0; i < result.rows; i++) {
			for (int j = 0; j < result.cols; j++) {
				int value = 0;
				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						value += padded.at<uint8_t>(i + 1 + dy, j + 1 + dx);
					}
				}
				result.at<uint8_t>(i, j) = value / (boxSize * boxSize);
			}
		}
	}

	if (boxSize == 5) {
		
		for (int i = 0; i < result.rows; i++) {
			for (int j = 0; j < result.cols; j++) {
				int value = 0;
				for (int dy = -2; dy <= 2; dy++) {
					for (int dx = -2; dx <= 2; dx++) {
						value += padded.at<uint8_t>(i + 2 + dy, j + 2 + dx);
					}
				}
				result.at<uint8_t>(i, j) = value / (boxSize * boxSize);
			}
		}
	}

	return result;
}


int findMed(Mat temp, int medSize)
{
	if (medSize == 3) {
		int array[9] = { 0 };
		int* ptr = array;
		for (int i = 0; i < medSize; i++) {
			for (int j = 0; j < medSize; j++) {
				*(ptr++) = temp.at<uint8_t>(i, j);
			}
		}

		for (int i = 9 - 2; i >= 0; i--) {
			for (int j = 0; j <= i; j++) {
				if (array[j] > array[j + 1]) {
					int temp = array[j];
					array[j] = array[j + 1];
					array[j + 1] = temp;
				}
			}
		}

		return array[4];
	}

	
	if (medSize == 5) {
		int array[25] = { 0 };
		int* ptr = array;
		
		for (int i = 0; i < medSize; i++) {
			for (int j = 0; j < medSize; j++) {
				*(ptr++) = temp.at<uint8_t>(i, j);
			}
		}

		for (int i = 25 - 2; i >= 0; i--) {
			for (int j = 0; j <= i; j++) {
				if (array[j] > array[j + 1]) {
					int temp = array[j];
					array[j] = array[j + 1];
					array[j + 1] = temp;
				}
			}
		}

		return array[12];
	}
}
Mat median_filter(Mat padded, Mat input, int medSize)
{
	Mat result;
	input.copyTo(result);
	if (medSize == 3) {
		for (int i = 0; i < padded.rows - 2; i++) {
			for (int j = 0; j < padded.cols - 2; j++) {
				Mat temp = padded(cv::Rect(j, i, 3, 3));
				result.at<uint8_t>(i, j) = findMed(temp, medSize);
			}
		}
	}

	if (medSize == 5) {
		for (int i = 0; i < padded.rows - 4; i++) {
			for (int j = 0; j < padded.cols - 4; j++) {
				Mat temp = padded(cv::Rect(j, i, 5, 5));
				result.at<uint8_t>(i, j) = findMed(temp, medSize);
			}
		}
	}

	return result;
}

int findMax(Mat input)
{
	int max = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (!(i == 0 && j == 0) && !(i == 4 && j == 0) && !(i == 0 && j == 4) && !(i == 4 && j == 4)) {
				if (input.at<uint8_t>(i, j) > max)
					max = input.at<uint8_t>(i, j);
			}
		}
	}
	return max;
}

int findMin(Mat input)
{
	int min = 255;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (!(i == 0 && j == 0) && !(i == 4 && j == 0) && !(i == 0 && j == 4) && !(i == 4 && j == 4)) {
				if (input.at<uint8_t>(i, j) < min)
					min = input.at<uint8_t>(i, j);
			}
		}
	}
	return min;
}

Mat opening_closing(Mat input, Mat kernel)
{
	int kernelSize = kernel.rows;
	Mat padded_input = padding(padding(input));
	Mat erosion1 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_input.rows - 4; i++) {
		for (int j = 0; j < padded_input.cols - 4; j++) {
			Mat temp = padded_input(cv::Rect(j, i, kernelSize, kernelSize));
			erosion1.at<uint8_t>(i, j) = findMin(temp);
		}
	}

	Mat padded_erosion1 = padding(padding(erosion1));
	Mat dilation1 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_erosion1.rows - 4; i++) {
		for (int j = 0; j < padded_erosion1.cols - 4; j++) {
			Mat temp = padded_erosion1(cv::Rect(j, i, kernelSize, kernelSize));
			dilation1.at<uint8_t>(i, j) = findMax(temp);
		}
	}

	Mat padded_dilation1 = padding(padding(dilation1));
	Mat dilation2 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_dilation1.rows - 4; i++) {
		for (int j = 0; j < padded_dilation1.cols - 4; j++) {
			Mat temp = padded_dilation1(cv::Rect(j, i, kernelSize, kernelSize));
			dilation2.at<uint8_t>(i, j) = findMax(temp);
		}
	}

	Mat padded_dilation2 = padding(padding(dilation2));
	Mat erosion2 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_dilation2.rows - 4; i++) {
		for (int j = 0; j < padded_dilation2.cols - 4; j++) {
			Mat temp = padded_dilation2(cv::Rect(j, i, kernelSize, kernelSize));
			erosion2.at<uint8_t>(i, j) = findMin(temp);
		}
	}
	return erosion2;
}

Mat closing_opening(Mat input, Mat kernel)
{
	int kernelSize = kernel.rows;
	Mat padded_input = padding(padding(input));
	Mat dilation1 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_input.rows - 4; i++) {
		for (int j = 0; j < padded_input.cols - 4; j++) {
			Mat temp = padded_input(cv::Rect(j, i, kernelSize, kernelSize));
			dilation1.at<uint8_t>(i, j) = findMax(temp);
		}
	}

	Mat padded_dilation1 = padding(padding(dilation1));
	Mat erosion1 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_dilation1.rows - 4; i++) {
		for (int j = 0; j < padded_dilation1.cols - 4; j++) {
			Mat temp = padded_dilation1(cv::Rect(j, i, kernelSize, kernelSize));
			erosion1.at<uint8_t>(i, j) = findMin(temp);
		}
	}

	Mat padded_erosion1 = padding(padding(erosion1));
	Mat erosion2 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_erosion1.rows - 4; i++) {
		for (int j = 0; j < padded_erosion1.cols - 4; j++) {
			Mat temp = padded_erosion1(cv::Rect(j, i, kernelSize, kernelSize));
			erosion2.at<uint8_t>(i, j) = findMin(temp);
		}
	}

	Mat padded_erosion2 = padding(padding(erosion2));
	Mat dilation2 = Mat::zeros(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < padded_erosion2.rows - 4; i++) {
		for (int j = 0; j < padded_erosion2.cols - 4; j++) {
			Mat temp = padded_erosion2(cv::Rect(j, i, kernelSize, kernelSize));
			dilation2.at<uint8_t>(i, j) = findMax(temp);
		}
	}
	return dilation2;
}

double signal_to_noise_ratio(Mat Signal, Mat Noise)
{
	double meanSignal = 0;
	for (int i = 0; i < Signal.rows; i++)
		for (int j = 0; j < Signal.cols; j++)
			meanSignal += Signal.at<uint8_t>(i, j) / 255.0;
	meanSignal /= (Signal.rows * Signal.cols);

	double varianceSignal = 0;
	for (int i = 0; i < Signal.rows; i++)
		for (int j = 0; j < Signal.cols; j++)
			varianceSignal += pow(Signal.at<uint8_t>(i, j) / 255.0 - meanSignal, 2);
	varianceSignal /= (Signal.rows * Signal.cols);

	double meanNoise = 0;
	for (int i = 0; i < Noise.rows; i++)
		for (int j = 0; j < Noise.cols; j++)
			meanNoise += (Noise.at<uint8_t>(i, j) - Signal.at<uint8_t>(i, j)) / 255.0;
	meanNoise /= (Noise.rows * Noise.cols);

	double varianceNoise = 0;
	for (int i = 0; i < Noise.rows; i++)
		for (int j = 0; j < Noise.cols; j++)
			varianceNoise += pow((Noise.at<uint8_t>(i, j) - Signal.at<uint8_t>(i, j)) / 255.0 - meanNoise, 2);
	varianceNoise /= (Noise.rows * Noise.cols);

	double SNR = 20 * log10(sqrt(varianceSignal / varianceNoise));
	return SNR;
}
int main()
{
	//read image
	Mat image = imread("lena.bmp", 1);
	cvtColor(image, image, COLOR_RGB2GRAY);

	fstream file("SNR.txt", ios::out);

	//gaussian noise image 
	Mat gaussian10 = gaussianNoise(image, 10, 0, 1);
	imwrite("gaussian10.jpg", gaussian10);
	file << "gaussian10 SNR: " << signal_to_noise_ratio(image, gaussian10) << endl;
	

	Mat gaussian30 = gaussianNoise(image, 30, 0, 1);
	imwrite("gaussian30.jpg", gaussian30);
	file << "gaussian30 SNR: " << signal_to_noise_ratio(image, gaussian30) << endl;
	

	//salt_and_pepper noise image
	Mat salt_pepper5 = salt_and_pepper(image, 0.05);
	imwrite("salt_pepper5.jpg", salt_pepper5);
	file << "salt_pepper5 SNR: " << signal_to_noise_ratio(image, salt_pepper5) << endl;

	Mat salt_pepper10 = salt_and_pepper(image, 0.10);
	imwrite("salt_pepper10.jpg", salt_pepper10);
	file << "salt_pepper10 SNR: " << signal_to_noise_ratio(image, salt_pepper10) << endl;

	//box filter 過濾 gaussian noise image
	Mat gaussian10_box3 = box_filter(padding(gaussian10), gaussian10, 3);
	imwrite("gaussian10_box3.jpg", gaussian10_box3);
	file << "gaussian10_box3 SNR: " << signal_to_noise_ratio(image, gaussian10_box3) << endl;

	Mat gaussian10_box5 = box_filter(padding(padding(gaussian10)), gaussian10, 5);
	imwrite("gaussian10_box5.jpg", gaussian10_box5);
	file << "gaussian10_box5 SNR: " << signal_to_noise_ratio(image, gaussian10_box5) << endl;

	Mat gaussian30_box3 = box_filter(padding(gaussian30), gaussian30, 3);
	imwrite("gaussian30_box3.jpg", gaussian30_box3);
	file << "gaussian30_box3 SNR: " << signal_to_noise_ratio(image, gaussian30_box3) << endl;

	Mat gaussian30_box5 = box_filter(padding(padding(gaussian30)), gaussian30, 5);
	imwrite("gaussian30_box5.jpg", gaussian30_box5);
	file << "gaussian30_box5 SNR: " << signal_to_noise_ratio(image, gaussian30_box5) << endl;

	//box filter 過濾 salt_and_pepper image
	Mat salt_pepper5_box3 = box_filter(padding(salt_pepper5), salt_pepper5, 3);
	imwrite("salt_pepper5_box3.jpg", salt_pepper5_box3);
	file << "salt_pepper5_box3 SNR: " << signal_to_noise_ratio(image, salt_pepper5_box3) << endl;

	Mat salt_pepper5_box5 = box_filter(padding(padding(salt_pepper5)), salt_pepper5, 5);
	imwrite("salt_pepper5_box5.jpg", salt_pepper5_box5);
	file << "salt_pepper5_box5 SNR: " << signal_to_noise_ratio(image, salt_pepper5_box5) << endl;

	Mat salt_pepper10_box3 = box_filter(padding(salt_pepper10), salt_pepper10, 3);
	imwrite("salt_pepper10_box3.jpg", salt_pepper10_box3);
	file << "salt_pepper10_box3 SNR: " << signal_to_noise_ratio(image, salt_pepper10_box3) << endl;

	Mat salt_pepper10_box5 = box_filter(padding(padding(salt_pepper10)), salt_pepper10, 5);
	imwrite("salt_pepper10_box5.jpg", salt_pepper10_box5);
	file << "salt_pepper10_box5 SNR: " << signal_to_noise_ratio(image, salt_pepper10_box5) << endl;

	//median filter 過濾 gaussian noise image
	Mat gaussian10_med3 = median_filter(padding(gaussian10), gaussian10, 3);
	imwrite("gaussian10_med3.jpg", gaussian10_med3);
	file << "gaussian10_med3 SNR: " << signal_to_noise_ratio(image, gaussian10_med3) << endl;

	Mat gaussian10_med5 = median_filter(padding(padding(gaussian10)), gaussian10, 5);
	imwrite("gaussian10_med5.jpg", gaussian10_med5);
	file << "gaussian10_med5 SNR: " << signal_to_noise_ratio(image, gaussian10_med5) << endl;

	Mat gaussian30_med3 = median_filter(padding(gaussian30), gaussian30, 3);
	imwrite("gaussian30_med3.jpg", gaussian30_med3);
	file << "gaussian30_med3 SNR: " << signal_to_noise_ratio(image, gaussian30_med3) << endl;

	Mat gaussian30_med5 = median_filter(padding(padding(gaussian30)), gaussian30, 5);
	imwrite("gaussian30_med5.jpg", gaussian30_med5);
	file << "gaussian30_med5 SNR: " << signal_to_noise_ratio(image, gaussian30_med5) << endl;

	//median filter 過濾 salt_and_pepper image
	Mat salt_pepper5_med3 = median_filter(padding(salt_pepper5), salt_pepper5, 3);
	imwrite("salt_pepper5_med3.jpg", salt_pepper5_med3);
	file << "salt_pepper5_med3 SNR: " << signal_to_noise_ratio(image, salt_pepper5_med3) << endl;

	Mat salt_pepper5_med5 = median_filter(padding(padding(salt_pepper5)), salt_pepper5, 5);
	imwrite("salt_pepper5_med5.jpg", salt_pepper5_med5);
	file << "salt_pepper5_med5 SNR: " << signal_to_noise_ratio(image, salt_pepper5_med5) << endl;

	Mat salt_pepper10_med3 = median_filter(padding(salt_pepper10), salt_pepper10, 3);
	imwrite("salt_pepper10_med3.jpg", salt_pepper10_med3);
	file << "salt_pepper10_med3 SNR: " << signal_to_noise_ratio(image, salt_pepper10_med3) << endl;

	Mat salt_pepper10_med5 = median_filter(padding(padding(salt_pepper10)), salt_pepper10, 5);
	imwrite("salt_pepper10_med5.jpg", salt_pepper10_med5);
	file << "salt_pepper10_med5 SNR: " << signal_to_noise_ratio(image, salt_pepper10_med5) << endl;

	/*kernel initialize*/
	Mat kernel = (Mat_<uint8_t>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0);

	//opening_closing
	Mat gaussian10_opening_closing = opening_closing(gaussian10, kernel);
	imwrite("gaussian10_opening_closing.jpg", gaussian10_opening_closing);
	file << "gaussian10_opening_closing SNR: " << signal_to_noise_ratio(image, gaussian10_opening_closing) << endl;

	Mat gaussian30_opening_closing = opening_closing(gaussian30, kernel);
	imwrite("gaussian30_opening_closing.jpg", gaussian30_opening_closing);
	file << "gaussian30_opening_closing SNR: " << signal_to_noise_ratio(image, gaussian30_opening_closing) << endl;

	Mat salt_pepper5_opening_closing = opening_closing(salt_pepper5, kernel);
	imwrite("salt_pepper5_opening_closing.jpg", salt_pepper5_opening_closing);
	file << "salt_pepper5_opening_closing SNR: " << signal_to_noise_ratio(image, salt_pepper5_opening_closing) << endl;

	Mat salt_pepper10_opening_closing = opening_closing(salt_pepper10, kernel);
	imwrite("salt_pepper10_opening_closing.jpg", salt_pepper10_opening_closing);
	file << "salt_pepper10_opening_closing SNR: " << signal_to_noise_ratio(image, salt_pepper10_opening_closing) << endl;

	//closing_opening
	Mat gaussian10_closing_opening = closing_opening(gaussian10, kernel);
	imwrite("gaussian10_closing_opening.jpg", gaussian10_closing_opening);
	file << "gaussian10_closing_opening SNR: " << signal_to_noise_ratio(image, gaussian10_closing_opening) << endl;

	Mat gaussian30_closing_opening = closing_opening(gaussian30, kernel);
	imwrite("gaussian30_closing_opening.jpg", gaussian30_closing_opening);
	file << "gaussian30_closing_opening SNR: " << signal_to_noise_ratio(image, gaussian30_closing_opening) << endl;

	Mat salt_pepper5_closing_opening = closing_opening(salt_pepper5, kernel);
	imwrite("salt_pepper5_closing_opening.jpg", salt_pepper5_closing_opening);
	file << "salt_pepper5_closing_opening SNR: " << signal_to_noise_ratio(image, salt_pepper5_closing_opening) << endl;

	Mat salt_pepper10_closing_opening = closing_opening(salt_pepper10, kernel);
	imwrite("salt_pepper10_closing_opening.jpg", salt_pepper10_closing_opening);
	file << "salt_pepper10_closing_opening SNR: " << signal_to_noise_ratio(image, salt_pepper10_closing_opening) << endl;

	Mat median5 = median_filter(padding(padding(image)), image, 5);
	file << "median5 SNR: " << signal_to_noise_ratio(image, median5) << endl;
	file.close();
	return 0;
}