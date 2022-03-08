#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <tuple>
#include <fstream>



using namespace cv;
using namespace std;


std::tuple<Mat, int>topdown_labeling(Mat image, int i, int j, int label) {
	int min = label;
	if (i < image.rows - 1 && j < image.cols - 1) {
		if (image.at<uchar>(i + 1, j) > 1 && image.at<uchar>(i + 1, j) < min)
			min = image.at<uchar>(i + 1, j);
		if (image.at<uchar>(i - 1, j) > 1 && image.at<uchar>(i - 1, j) < min)
			min = image.at<uchar>(i - 1, j);
		if (image.at<uchar>(i, j + 1) > 1 && image.at<uchar>(i, j + 1) < min)
			min = image.at<uchar>(i, j + 1);
		if (image.at<uchar>(i, j - 1) > 1 && image.at<uchar>(i, j - 1) < min)
			min = image.at<uchar>(i, j - 1);
		image.at<uchar>(i, j) = min;
	}
	if (min == label)
		label = label % 256 + 1;

	return { image, label };
}

Mat bottomup_labeling(Mat image, int i, int j) {
	int min = 1000;
	if (i < image.rows - 1 && j < image.cols - 1) {
		if (image.at<uchar>(i + 1, j) > 1 && image.at<uchar>(i + 1, j) < min)
			min = image.at<uchar>(i + 1, j);
		if (image.at<uchar>(i - 1, j) > 1 && image.at<uchar>(i - 1, j) < min)
			min = image.at<uchar>(i - 1, j);
		if (image.at<uchar>(i, j + 1) > 1 && image.at<uchar>(i, j + 1) < min)
			min = image.at<uchar>(i, j + 1);
		if (image.at<uchar>(i, j - 1) > 1 && image.at<uchar>(i, j - 1) < min)
			min = image.at<uchar>(i, j - 1);
		image.at<uchar>(i, j) = min;
	}
	return image;
}
int main()
{
	/*binarize image*/
	// read image 
	Mat img1 = imread("lena.bmp");
	Mat thresh_img;
	cvtColor(img1, thresh_img, COLOR_RGB2GRAY);

	//binarize
	for (int i = 0; i < thresh_img.rows; i++)
		for (int j = 0; j < thresh_img.cols; j++)
		{
			if (thresh_img.at<uchar>(i, j) < 128)
				thresh_img.at<uchar>(i, j) = 0;
			else
				thresh_img.at<uchar>(i, j) = 255;
		}

	// show image
	imshow("binary", thresh_img);
	waitKey(0);

	/*histogram*/
	//read image
	Mat img2 = imread("lena.bmp", 1);
	cvtColor(img2, img2, COLOR_RGB2GRAY);
	//use histogram array to calculate pixels
	int histogram[256] = { 0 };
	for (int i = 0; i < img2.rows; i++) {
		for (int j = 0; j < img2.cols; j++) {
			histogram[img2.at<uchar>(i, j)] += 1;
		}
	}

	//畫圖
	Mat histogram_graph(256, 256, CV_8UC1, Scalar(255));
	int max = 0;
	for (int i = 0; i < sizeof(histogram) / sizeof(*histogram); i++) {
		if (histogram[i] > max)max = histogram[i];
	}

	for (int i = 0; i < sizeof(histogram) / sizeof(*histogram); i++) {
		cv::line(histogram_graph, Point(i, 255), Point(i, 255 - histogram[i] * 255 / max), Scalar(0), 1);
	}

	imshow("histogram", histogram_graph);
	waitKey(0);

	//轉成CSV檔
	fstream myfile("histogram.csv", ios::out);
	for (int y = 0; y < 256; y++)
		myfile << histogram[y] << endl;
	myfile.close();

	/*connected component*/
	//read image
	Mat img3, img4;
	thresh_img.copyTo(img3);
	thresh_img.copyTo(img4);


	int label = 1;
	int count[700] = { 0 };
	int threshold = 400;
	int epoch = 10;

	for (int c = 0; c < epoch; c++) {
		for (int x = 0; x < 50; x++) {
			label = 1;
			//top down
			for (int i = 0; i < img3.rows - 1; i++) {
				for (int j = 0; j < img3.cols - 1; j++) {
					if (img3.at<uchar>(i, j) != 0) {
						tie(img3, label) = topdown_labeling(img3, i, j, label);
					}
				}
			}
			//bottomup
			for (int i = img3.rows - 1; i > 0; i--) {
				for (int j = img3.cols - 1; j > 0; j--) {
					if (img3.at<uchar>(i, j) != 0) {
						img3 = bottomup_labeling(img3, i, j);
					}
				}
			}
		}
		for (int i = 0; i < sizeof(count) / sizeof(*count); i++) {
			count[i] = 0;
		}
		for (int i = 0; i < img3.rows; i++) {
			for (int j = 0; j < img3.cols; j++) {
				count[img3.at<uchar>(i, j)] += 1;
			}
		}

		if (c != epoch - 1) {
			for (int r = 1; r < sizeof(count) / sizeof(*count); r++) {
				if (count[r] < threshold) {
					for (int i = 0; i < img3.rows; i++) {
						for (int j = 0; j < img3.cols; j++) {
							if (img3.at<uchar>(i, j) == r) {
								img3.at<uchar>(i, j) = 0;
							}
						}
					}
				}
				else {
					for (int i = 0; i < img3.rows; i++) {
						for (int j = 0; j < img3.cols; j++) {
							if (img3.at<uchar>(i, j) == r) {
								img3.at<uchar>(i, j) = 255;
							}
						}
					}
				}
			}
		}

	}
	for (int i = 0; i < img3.rows; i++) {
		for (int j = 0; j < img3.cols; j++) {
			if (img3.at<uchar>(i, j) == 255) {
				img3.at<uchar>(i, j) = 0;
			}
		}
	}

	Mat test_image;
	img3.copyTo(test_image);

	for (int r = 1; r < sizeof(count) / sizeof(*count); r++) {
		if (count[r] < threshold) {
			for (int i = 0; i < img3.rows; i++) {
				for (int j = 0; j < img3.cols; j++) {
					if (img3.at<uchar>(i, j) == r) {
						img3.at<uchar>(i, j) = 0;
					}
				}
			}
		}
		else {
			for (int i = 0; i < img3.rows; i++) {
				for (int j = 0; j < img3.cols; j++) {
					if (img3.at<uchar>(i, j) == r) {
						img3.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}

	cvtColor(img4, img4, COLOR_GRAY2BGR);
	cvtColor(img3, img3, COLOR_GRAY2BGR);
	for (int r = 1; r < sizeof(count) / sizeof(*count); r++) {

		if (count[r] >= 500 && r != 255) {
			int left = 512, top = 512, right = 0, bottom = 0;
			for (int i = 0; i < test_image.rows; i++) {
				for (int j = 0; j < test_image.cols; j++) {
					if (test_image.at<uchar>(i, j) == r) {
						if (i > bottom)
							bottom = i + 1;
						if (i < top)
							top = i;
						if (j > right)
							right = j + 1;
						if (j < left)
							left = j;

					}
				}
			}
			cv::Rect rect(left, top, (right - left), bottom - top);
			cv::rectangle(img4, rect, cv::Scalar(255, 0, 0), 2);
			cv::line(img4, cv::Point(((right + left) / 2 - 10), (top + bottom) / 2), cv::Point(((right + left) / 2 + 10), (top + bottom) / 2), CV_RGB(255, 0, 0), 1, LINE_AA, 0);
			cv::line(img4, cv::Point(((right + left) / 2), (top + bottom) / 2 - 10), cv::Point(((right + left) / 2), (top + bottom) / 2 + 10), CV_RGB(255, 0, 0), 1, LINE_AA, 0);

			imshow("connected_component", img4);
			waitKey(0);
		}
	}
	
	// write image 
	imwrite("binary_image.jpg", thresh_img);
	imwrite("histogram.jpg", histogram_graph);
	imwrite("connected_component.jpg", img4);
	
	return 0;
}

