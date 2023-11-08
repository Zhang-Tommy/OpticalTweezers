#pragma once
#ifndef _SAPMYPROCESSING_H_
#define _SAPMYPROCESSING_H_
//#include "C:\Program Files\Teledyne DALSA\Sapera\Classes\Basic\SapProcessing.h"
// SapMyProcessing.h : header file
//

#include "SapClassBasic.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
//
// SapMyProcessing class declaration
//
class SapMyProcessing : public SapProcessing
{
public:
	// Constructor/Destructor
	SapMyProcessing(SapBuffer* pBuffers);
	void PassMatrix(cv::Mat);
	virtual	BOOL Run();
public:
	cv::Mat ReadMatFromTxt(std::string filename, int rows, int cols);
public:
	void autoCanny(cv::Mat image, cv::Mat &output, float sigma = 0.33);

public:
	cv::Mat Image;
};

#endif   // _SAPMYPROCESSING_H_

