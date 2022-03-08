#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

Mat downsampling(Mat input)
{
	Mat resized_image = Mat::zeros(64, 64, CV_8UC1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (i % 8 == 0 && j % 8 == 0) {
				resized_image.at<uint8_t>(i / 8, j / 8) = input.at<uint8_t>(i, j);
			}
		}
	}
	return resized_image;
}

int Sum(Mat input)
{
	int total = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uint8_t>(i, j) == 255)
				total += 1;
		}
	}
	return total;
}

int Yokoi(Mat input,int row, int col)
{
	int di[] = {0, -1, 0, 1, 1, -1, -1, 1};
	int dj[] = {1, 0, -1, 0, 1, 1, -1, -1};
	char a[4] = { 0 };
	int n;
	if (row == 0 && col == 0) {
		int i = 0, j = 0;
		for (n = 2; n < 3; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[0], j + dj[0]))
			a[0] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (row == 0 && col == 63) {
		int i = 0, j = 1;
		for (n = 1; n < 2; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[3], j + dj[3]))
			a[3] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (row == 63 && col == 0) {
		int i = 1, j = 0;
		for (n = -1; n < 0; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[1], j + dj[1]))
			a[1] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if(row == 63 && col == 63) {
		int i = 1, j = 1;
		for (n = 0; n < 1; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[2], j + dj[2]))
			a[2] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (row == 0) {
		int i = 0, j = 1;
		for (n = 1; n < 3; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[0], j + dj[0]))
			a[0] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (row == 63) {
		int i = 1, j = 1;
		for (n = -1; n < 1; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[2], j + dj[2]))
			a[2] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (col == 0) {
		int i = 1, j = 0;
		for (n = -1; n < 3; n+=3) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[1], j + dj[1]))
			a[1] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else if (col == 63) {
		int i = 1, j = 1;
		for (n = 0; n < 2; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) && input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'r';
				else
					a[n + 1] = 'q';
			}
			else
				a[n + 1] = 's';
		}
		if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[3], j + dj[3]))
			a[3] = 'q';
		int count = 0;
		for (n = 0; n < 4; n++)
			if (a[n] == 'q')
				count++;
		return count;
	}
	else {
		int i = 1, j = 1;
		for (n = -1; n < 3; n++) {
			if (input.at<uint8_t>(i, j) == input.at<uint8_t>(i + di[n + 1], j + dj[n + 1])) {
				if (input.at<uint8_t>(i, j) != input.at<uint8_t>(i + di[n + 2], j + dj[n + 2]) || input.at<uint8_t>(i, j) != input.at<uint8_t>(i + di[(n + 6) % 8], j + dj[(n + 6) % 8]))
					a[n + 1] = 'q';
				else
					a[n + 1] = 'r';
			}
			else
				a[n + 1] = 's';
		}
		if (a[0] == 'r' && a[1] == 'r' && a[2] == 'r' && a[3] == 'r')
			return 5;
		else {
			int count = 0;
			for (n = 0; n < 4; n++)
				if (a[n] == 'q')
					count++;
			return count;
		}
	}
}

char Pair(Mat input, int i, int j, int value)
{
	
	if (value != 1)
		return 'q';
	else {
		int di[] = { 0, -1, 0, 1 };
		int dj[] = { 1, 0, -1, 0 };
		int count = 0;
		for (int n = 0; n < 4; n++) {
			if (i + di[n] < 0) continue;
			else if (i + di[n] > 63) continue;
			else if (j + dj[n] < 0) continue;
			else if (j + dj[n] > 63) continue;
			else {
				if (input.at<uint8_t>(i + di[n], j + dj[n]) == 1)
					count++;
			}
				
		}
		if (count == 0)
			return 'q';
		else
			return 'p';
	}
}
int main()
{
	/*read image*/
	Mat image = imread("lena.bmp", 0);
	Mat thresh_image;
	image.copyTo(thresh_image);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uint8_t>(i, j) < 128)
				thresh_image.at<uint8_t>(i, j) = 0;
			else
				thresh_image.at<uint8_t>(i, j) = 255;
		}
	}
	/*downsampling*/
	Mat downsampled_image = downsampling(thresh_image);
	
	Mat result;
	while (1) {
		int value = Sum(downsampled_image);
		/*Yokoi operator*/
		Mat Yokoi_image = Mat::zeros(64, 64, CV_8UC1);
		for (int i = 0; i < downsampled_image.rows; i++) {
			for (int j = 0; j < downsampled_image.cols; j++) {
				if (downsampled_image.at<uint8_t>(i, j) == 255) {
					int startRow = i > 0 ? 1 : 0;
					int startCol = j > 0 ? 1 : 0;
					int sizeRow = startRow == 1 ? 3 : 2;
					int sizeCol = startCol == 1 ? 3 : 2;
					sizeRow = i == 63 ? 2 : sizeRow;
					sizeCol = j == 63 ? 2 : sizeCol;
					Mat temp = downsampled_image(cv::Rect(j - startCol, i - startRow, sizeCol, sizeRow));
					Yokoi_image.at<uint8_t>(i, j) = Yokoi(temp, i, j);
				}
			}
		}

		/*Pair relationship operator*/
		Mat Pair_image;
		Yokoi_image.copyTo(Pair_image);

		for (int i = 0; i < Yokoi_image.rows; i++) {
			for (int j = 0; j < Yokoi_image.cols; j++) {
				Pair_image.at<uint8_t>(i, j) = Pair(Yokoi_image, i, j, Yokoi_image.at<uint8_t>(i, j));
			}
		}

		/*Connected shriking operator*/
		Mat Connected_shrinking_image;
		downsampled_image.copyTo(Connected_shrinking_image);

		for (int i = 0; i < Connected_shrinking_image.rows; i++) {
			for (int j = 0; j < Connected_shrinking_image.cols; j++) {
				if (Pair_image.at<uint8_t>(i, j) == 'p' && Connected_shrinking_image.at<uint8_t>(i, j) == 255) {
					int startRow = i > 0 ? 1 : 0;
					int startCol = j > 0 ? 1 : 0;
					int sizeRow = startRow == 1 ? 3 : 2;
					int sizeCol = startCol == 1 ? 3 : 2;
					sizeRow = i == 63 ? 2 : sizeRow;
					sizeCol = j == 63 ? 2 : sizeCol;
					Mat temp = Connected_shrinking_image(cv::Rect(j - startCol, i - startRow, sizeCol, sizeRow));
					if (Yokoi(temp, i, j) == 1) {
						Connected_shrinking_image.at<uint8_t>(i, j) = 0;
					}
				}
			}
		}

		if (Sum(Connected_shrinking_image) == value) {
			Connected_shrinking_image.copyTo(result);
			break;
		}
		Connected_shrinking_image.copyTo(downsampled_image);
	}
	
	imshow("Connected_shrinking", result);
	waitKey(0);
	imwrite("Connected_shrinking.jpg", result);
	return 0;
}