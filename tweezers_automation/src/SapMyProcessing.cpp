#include "stdafx.h"
#include "SapMyProcessing.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <SapClassBasic.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstring>
//#define COUT(string) cout<< string << endl
using namespace std;
using namespace std::chrono;


// Constructor/Destructor
//
SapMyProcessing::SapMyProcessing(SapBuffer* pBuffers) : SapProcessing(pBuffers)
{
	SapBuffer *CurBuffer = pBuffers;
}

void SapMyProcessing::PassMatrix(cv::Mat pass)
{
	Image = pass.clone();
}

// Processing Control

cv::Mat SapMyProcessing::ReadMatFromTxt(std::string filename, int rows, int cols)
{
	double m;
	cv::Mat out = cv::Mat::zeros(rows, cols, CV_8UC1);//Matrix to store values

	std::ifstream fileStream(filename);
	int cnt = 0;//index starts from 0
	while (fileStream >> m)
	{
		int temprow = cnt / cols;
		int tempcol = cnt % cols;
		out.at<double>(temprow, tempcol) = m;
		cnt++;
	}
	return out;
}
void SapMyProcessing::autoCanny(cv::Mat image, cv::Mat &output, float sigma) {
	std::vector<uchar> vecFromMat;
	if (image.isContinuous()) {
		vecFromMat.assign(image.datastart, image.dataend);
	}
	else {
		for (int i = 0; i < image.rows; ++i) {
			vecFromMat.insert(vecFromMat.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) +image.cols);
		}
	}
	nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
	double median = vecFromMat[int(vecFromMat.size() / 2)];
	//cout << median << endl;
	//double median = 131;
	double lower = 0.66*median;
	double upper = 1.33*median;
	cv::Canny(image, output, int(lower), int(upper));
}


BOOL SapMyProcessing::Run()
{
	//image processing code goes here
	
	
	//int rows = 480;
	//int cols = 640;
	//cv::Mat Image(cols,rows,CV_8UC1);
	//for (int y = 0; y < 480;y++)
	//{
		//for (int x = 0; x < 640; x++)
		//{	
		//	CurBuffer->ReadElement(x, y, &(Image.at<uchar>(x, y)));
	//	}
//	}

	int Debug = TRUE;
	int repeat = 100;
	if (Debug) cv::imwrite("acquiredimage.jpg", Image);
	__debugbreak();

	high_resolution_clock::time_point t0 = high_resolution_clock::now();


	duration<double> time_span = duration_cast<duration<double>>(t0 - t0);
	std::cout << "Loading Matrices";
	std::cout << "1 of 2";
	cv::Mat a = ReadMatFromTxt("a.txt", 480, 640);
	std::cout << "2 of 2";
	cv::Mat b = ReadMatFromTxt("b.txt", 480, 640);
	std::cout << "Done";
	for (int i = 0; i < repeat; i++) {
		cv::Mat oriimg;
		oriimg = Image;
		if (oriimg.empty()) // Check for invalid input
		{
			cout << "Could not open or find the image" << endl;
			return -1;
		}

		//cv:: namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		cv::Mat img;
		cv::Mat img1;
		cv::Mat img2;
		cv::Mat edges;
		int h = oriimg.cols; int w = oriimg.rows;
		//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
		cv::subtract(oriimg, b, img1, cv::noArray(), CV_8UC1);
		cv::divide(img1, a, img2);
		//cv::imshow("Display window", img2);  // Show our image inside it.
		if (Debug) cv::imwrite("1removeSmudges.jpg", img2);
		__debugbreak();
		//cv::waitKey(0);
		//Step 1: detecting beads
		img2.convertTo(img, CV_8UC1);
		cv::Mat Step1;
		int offset = 2;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		cv::equalizeHist(img, Step1);
		if (Debug) cv::imwrite("2 equalizeHist.jpg", Step1);
		cv::medianBlur(Step1, Step1, 3);
		cv::copyMakeBorder(Step1, Step1, offset, offset, offset, offset, cv::BORDER_CONSTANT, cv::Scalar(255));
		if (Debug) cv::imwrite("3 addBorder.jpg", Step1);

		//try
		//{
		//	cv::Ptr<cv::MSER> asd = cv::MSER::create(5, 3000, 5000,0.15);
		//	vector<vector<cv::Point>> mservec;
		//	vector <cv::Rect> mserrec;
		//	asd->detectRegions(img, mservec, mserrec);
		//	int i = 0;
		//	//result = Scalar(0, 0, 0);
		//	cv::Mat result(img.rows, img.cols, CV_8UC1);

		//	for (vector<vector <cv::Point> >::iterator itr = mservec.begin(); itr != mservec.end(); itr++, i++)
		//	{
		//		for (vector <cv::Point>::iterator itp = mservec[i].begin(); itp != mservec[i].end(); itp++)
		//		{
		//			// all pixels belonging to region become blue
		//			result.at<cv::Vec3b>(itp->y, itp->x) = cv::Vec3b(255);
		//		}
		//	}
		//	cv::namedWindow("Wow", cv::WINDOW_AUTOSIZE);
		//	cv::imshow("result", result);
		//	cv::imshow("Original", img);
		//	cv::waitKey();
		//}
		//catch (exception & e)
		//{
		//	cout << e.what() << endl;
		//}

		cv::SimpleBlobDetector::Params params;
		params.minArea = 300;
		params.minCircularity = 0.85;
		params.filterByArea = 1;
		params.filterByInertia = 1;
		params.filterByCircularity = 1;
		params.filterByColor = 1;
		std::vector<cv::KeyPoint> keypoints;
		cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);
		blobDetector->detect(Step1, keypoints);
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		cv::Mat imgWithKeypoints;
		for (int i = 0; i < keypoints.size(); i++) {
			keypoints[i].pt = cv::Point(keypoints[i].pt.x - 2, keypoints[i].pt.y - 2);
		}
		cv::drawKeypoints(img, keypoints, imgWithKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		if (Debug) cv::imwrite("4 detectBeads.jpg", imgWithKeypoints);
		cv::Mat Step3;
		Step3 = img.clone();
		if (Debug) cv::imwrite("5 remove Beads.jpg", Step3);
		high_resolution_clock::time_point t3 = high_resolution_clock::now();
		cv::Mat hist;
		int channels[] = { 0 };
		int histSize[] = { 256 };
		float range[] = { 0, 256 };
		const float* ranges[] = { range };
		cv::calcHist(&img, 1, channels, cv::Mat(), // do not use mask
			hist, 1, histSize, ranges,
			true, // the histogram is uniform
			false);
		int idx;
		//double maxVal;
		cv::minMaxIdx(hist, NULL, NULL, NULL, &idx);
		int len = keypoints.size();

		for (int i = 0; i < len; i++) {
			cv::circle(Step3, keypoints[i].pt, (keypoints[i].size) * 3 / 4, cv::Scalar(idx), -1);
		}
		high_resolution_clock::time_point t4 = high_resolution_clock::now();

		cv::Mat grey;
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.2, cv::Size(4, 4));
		clahe->apply(Step3, grey);
		//cv::medianBlur(grey, grey, 1);
		autoCanny(grey, edges);
		high_resolution_clock::time_point t5 = high_resolution_clock::now();

		if (Debug) cv::imwrite("6 edges.jpeg", edges);
		//cv::imshow("Display window", edges); // Show our image inside it.
		//cv::waitKey(0); // Wait for a keystroke in the window
		int kernelSize = 15;
		//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
		cv::Mat kernel = cv::Mat::ones(14, 14, CV_8U);
		cv::Mat dilatedImage;
		cv::dilate(edges, dilatedImage, kernel);
		if (Debug) cv::imwrite("7 dilation.jpg", dilatedImage);
		cv::Mat mask = cv::Mat::zeros(8, 8, CV_8U);
		int step = 1;
		cv::Mat dilCopy = dilatedImage.clone();
		//for i in range(0, dilcopy.shape[0] / mult) :
		//	if dilcopy[i*mult, 0] == 0 :
		//		cv2.floodFill(dilcopy, mask, (0, i*mult), 255);
		//if dilcopy[i*mult, w - 1] == 0:
		//cv2.floodFill(dilcopy, mask, (w - 1, i*mult), 255);
		for (int i = 0; i < w / step; i++) {
			if (dilCopy.at<uchar>(cv::Point(0, i)) == 0) {
				cv::floodFill(dilCopy, cv::Point(0, i*step), 255);
			}
		}
		for (int i = 0; i < h / step; i++) {
			if (dilCopy.at<uchar>(cv::Point(i, 0)) == 0) {
				cv::floodFill(dilCopy, cv::Point(i*step, 0), 255);
			}
		}
		if (Debug) cv::imwrite("8 floodFill.jpeg", dilCopy);
		high_resolution_clock::time_point t6 = high_resolution_clock::now();

		cv::Mat mid;
		cv::bitwise_not(dilCopy, dilCopy);
		cv::add(dilatedImage, dilCopy, mid);
		if (Debug) cv::imwrite("9 removeInternal.jpg", mid);

		cv::erode(mid, mid, kernel);
		//cv::morphologyEx(mid, mid, cv::MORPH_CLOSE, kernel);
		cv::Mat midBkp = mid.clone();

		for (int i = 0; i < w / step; i++) {
			if (mid.at<uchar>(cv::Point(0, i)) == 255) {
				cv::floodFill(mid, cv::Point(0, i*step), 0);
			}
		}
		for (int i = 0; i < h / step; i++) {
			if (mid.at<uchar>(cv::Point(i, 0)) == 255) {
				cv::floodFill(mid, cv::Point(i*step, 0), 0);
			}
		}
		cv::Mat imEdge;
		cv::subtract(midBkp, mid, imEdge);
		cv::erode(mid, mid, kernel);
		cv::dilate(mid, mid, kernel);
		high_resolution_clock::time_point t7 = high_resolution_clock::now();

		if (Debug) cv::imwrite("11 removeOuter.jpg", mid);
		if (Debug) cv::imwrite("12 Outer.jpg", imEdge);
		vector<vector<cv::Point> > contours;
		vector<cv::Vec4i> hierarchy;
		vector<vector<cv::Point> > contours1;
		vector<cv::Vec4i> hierarchy1;
		/// Find contours
		findContours(mid.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
		findContours(imEdge.clone(), contours1, hierarchy1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
		double maxContourArea = 0;
		vector <double> areas;
		vector <double> areas1;
		cv::Mat Step4 = imgWithKeypoints.clone();
		for (int i = 0; i < contours.size(); i++) {
			double temp = cv::contourArea(contours[i]);
			areas.push_back(temp);
			if (temp > maxContourArea) {
				maxContourArea = temp;
			}
		}
		for (int i = 0; i < contours1.size(); i++) {
			double temp = cv::contourArea(contours1[i]);
			areas1.push_back(temp);
			if (temp > maxContourArea) {
				maxContourArea = temp;
			}
		}
		high_resolution_clock::time_point t8 = high_resolution_clock::now();

		for (int i = 0; i < contours.size(); i++) {
			if (areas[i] > maxContourArea / 8 && areas[i] / cv::arcLength(contours[i], true) > 3) {
				cv::drawContours(Step4, contours, i, cv::Scalar(0, 0, 255), 2);
				cv::RotatedRect  rRect = cv::minAreaRect(contours[i]);
				cv::Point2f vertices[4];
				rRect.points(vertices);
				for (int i = 0; i < 4; i++) {
					cv::line(Step4, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0));
				}
			}
		}
		for (int i = 0; i < contours1.size(); i++) {
			if (areas1[i] > maxContourArea / 20 && areas1[i] / cv::arcLength(contours1[i], true) > 3) {
				cv::drawContours(Step4, contours1, i, cv::Scalar(0, 255, 255), 2);
				cv::RotatedRect  rRect = cv::minAreaRect(contours1[i]);
				cv::Point2f vertices[4];
				rRect.points(vertices);
				for (int i = 0; i < 4; i++) {
					cv::line(Step4, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 0));
				}
			}
		}
		high_resolution_clock::time_point t9 = high_resolution_clock::now();

		if (Debug) cv::imwrite("13 final.jpg", Step4);
		time_span = duration_cast<duration<double>>(time_span + t9 - t1);
	}

	//duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t3 - t2);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	// time_span = duration_cast<duration<double>>(t4 - t3);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t5 - t4);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t6 - t5);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t7 - t6);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t8 - t7);
	//std::cout << "It took me " << time_span.count() << " seconds." << endl;
	//time_span = duration_cast<duration<double>>(t9 - t8);
	//std::cout << "It took me " << time_span.count() << " seconds."<< endl;

	time_span = time_span / repeat;
	std::cout << "It took me " << time_span.count() << " seconds." << endl;
	cv::waitKey(0);

	return TRUE;
}


int main() {
	return 0;
}