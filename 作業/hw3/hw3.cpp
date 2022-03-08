#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;

int main() 
{
	/*read image*/
	Mat img = imread("lena.bmp", 0);
    
	/*part1*/
	/*read image*/
	Mat img1;
	img.copyTo(img1);

	/*count pixels*/
	int histogram1[256] = { 0 };
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			histogram1[img1.at<uchar>(i, j)] += 1;
		}
	}

	/*以CSV檔輸出*/
	fstream file1("part1.csv", ios::out);
	for (int i = 0; i < 256; i++) {
		file1 << histogram1[i] << endl;
	}
	file1.close();

	/*part2*/
	/*read image*/
	Mat img2;
	img.copyTo(img2);

	/*灰階值除3並輸出影像*/
	for (int i = 0; i < img2.rows; i++) {
		for (int j = 0; j < img2.cols; j++) {
			img2.at<uchar>(i, j) /= 3;
		}
	}
    imshow("img2", img2);
	waitKey(0);
	imwrite("part2.jpg", img2);

	/*count pixels*/
	int histogram2[256] = { 0 };
	for (int i = 0; i < img2.rows; i++) {
		for (int j = 0; j < img2.cols; j++) {
			histogram2[img2.at<uchar>(i, j)] += 1;
		}
	}

	/*以CSV檔輸出*/
	fstream file2("part2.csv", ios::out);
	for (int i = 0; i < 256; i++) {
		file2 << histogram2[i] << endl;
	}
	file2.close();

	/*part3*/
	/*read image*/
	Mat img3;
	img2.copyTo(img3);

    /*count pixels*/
	int grayscale[256] = { 0 };
	for (int i = 0; i < img3.rows; i++) {
		for (int j = 0; j < img3.cols; j++) {
			grayscale[img3.at<uchar>(i, j)] += 1;
		}
	}

	/*equalization*/
	/*count cdf*/
	int cdf[256] = { 0 };
	int count = 0;
	for (int i = 0; i < 256; i++) {
		if (grayscale[i] != 0) {
			count += grayscale[i];
			cdf[i] = count;
		}
	}


	/*find max_cdf and min_cdf*/
	int max = 0, min = 0;
	for (int i = 0; i < 256; i++) {
		if (cdf[i] != 0) {
			min = cdf[i];
			break;
		}
	}
    
	for (int i = 255; i >= 0; i--) {
		if (cdf[i] != 0) {
			max = cdf[i];
			break;
		}
	}

	/*計算新的灰值*/
	double h[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		if (cdf[i] != 0) {
			h[i] = round((double)((cdf[i] - min) * 255 / (max - min)));
		}
	}

	/*out put lena*/
	for (int i = 0; i < img3.rows; i++) {
		for (int j = 0; j < img3.cols; j++) {
			for (int k = 0; k < 256; k++) {
				if (img3.at<uchar>(i, j) == k) {
					img3.at<uchar>(i, j) = h[k];
					break;
				}
			}
		}
	}

	imshow("img3", img3);
	waitKey(0);
	imwrite("part3.jpg", img3);

	/*count pixels and 直方圖*/
	int histogram3[256] = { 0 };
	for (int i = 0; i < img3.rows; i++) {
		for (int j = 0; j < img3.cols; j++) {
			histogram3[img3.at<uchar>(i, j)] += 1;
		}
	}
	/*以CSV檔輸出*/
	fstream file3("part3.csv", ios::out);
	for (int i = 0; i < 256; i++) {
		file3 << histogram3[i] << endl;
	}
	file3.close();
	return 0;
}

