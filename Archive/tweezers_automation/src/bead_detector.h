#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

cv::Mat ReadMatFromTxt(string filename, int rows, int cols);

void autoCanny(cv::Mat image, cv::Mat& output, float sigma = 0.33);

int detect_beads(string file_path);