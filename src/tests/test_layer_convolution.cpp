#include <iostream>
#include <chrono>
using namespace std;

#include "LayerConvolution2D.h"

//////////////////////////////////////////////////////////////////////////////
void im2col_col2im()
{
	cout << "Simple im2col_col2im test:" << endl;

	MatrixFloat mIn, mCol, mColLUT, mIm;
	mIn.setZero(5, 5);
	mIn(0, 0) = 100;
	mIn(0, 4) = 104;

	mIn(4, 4) = 144;
	mIn(2, 2) = 122;
	mIn.resize(1, 5 * 5);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 1);
	conv2d.im2col(mIn, mCol);
	conv2d.im2col_LUT(mIn, mColLUT);
	conv2d.col2im(mCol,mIm);

	mIn.resize(5, 5);
	mIm.resize(5, 5);
	cout << "Image:" << endl << toString(mIn) << endl << endl;
	cout << "Im2Col:" << endl << toString(mCol) << endl << endl;
	cout << "Im2ColLUT:" << endl << toString(mColLUT) << endl << endl;
	cout << "Col2Im:" << endl << toString(mIm) << endl << endl;
}
//////////////////////////////////////////////////////////////////////////////
void simple_image_conv2d()
{
	cout << "Simple convolution test:" << endl;

	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(5, 5);
	mIn(2, 2) = 1;
	mIn.resize(1, 5 * 5);

	mKernel.setZero(3, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	mKernel.resize(1, 3 * 3);
	
	LayerConvolution2D conv2d(5,5,1,3,3,1);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	mOut.resize(3, 3);
	cout << "Image convoluted:" << endl;
	cout << toString(mOut) << endl<< endl;
}
//////////////////////////////////////////////////////////////////////////////
void batch_conv2d()
{
	cout << "Batch convolution test:" << endl;

	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(10, 5);
	mIn(2, 2) = 1;
	mIn(2+5, 2) = 3;
	mIn.resize(2, 5 * 5);

	mKernel.setZero(3, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	mKernel.resize(1, 3 * 3);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 1);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	mOut.resize(6, 3);
	cout << "Batch convoluted, 2 samples:" << endl;
	cout << toString(mOut) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void image_2_input_channels_conv2d()
{
	cout << "Image 2 input channels convolution test:" << endl;

	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(10, 5);
	mIn(2, 2) = 1;
	mIn(2 + 5, 2) = 3;
	mIn.resize(1, 2 * 5 * 5);

	mKernel.setZero(6, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	
	mKernel(3, 1) = 1;
	mKernel(4, 1) = 2;
	mKernel(5, 1) = 1;

	mKernel.resize(1, 2 * 3 * 3);

	LayerConvolution2D conv2d(5, 5, 2, 3, 3, 1);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	mOut.resize(3, 3);
	cout << "Image 2 input channels convoluted:" << endl;
	cout << toString(mOut) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void image_2_output_channels_conv2d()
{
	cout << "Image 2 ouput channels convolution test:" << endl;

	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(5, 5);
	mIn(2, 2) = 1;
	mIn.resize(1, 5 * 5);

	mKernel.setZero(6, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;

	mKernel(3, 1) = 3;
	mKernel(4, 1) = 5;
	mKernel(5, 1) = 3;

	mKernel.resize(2, 3 * 3);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 2);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	mOut.resize(6, 3);
	cout << "Image 2 output channels convoluted:" << endl;
	cout << toString(mOut) << endl << endl;
}
//////////////////////////////////////////////////////////////////////////////
void simple_image_conv2d_stride2()
{
	cout << "Simple convolution test stride2:" << endl;

	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(5, 5);
	mIn(0, 0) = 100;
	mIn(2, 2) = 122;
	mIn(3, 4) = 134;
	mIn.resize(1, 5 * 5);

	mKernel.setZero(3, 3);
	mKernel(0, 0) = 1;
	mKernel(0, 1) = 2;
	mKernel(0, 2) = 1;

	mKernel(1, 0) = 2;
	mKernel(1, 1) = 4;
	mKernel(1, 2) = 2;

	mKernel(2, 0) = 1;
	mKernel(2, 1) = 2;
	mKernel(2, 2) = 1;

	mKernel.resize(1, 3 * 3);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 1,2,2);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	mOut.resize(2, 2);
	cout << "Image convoluted stride2:" << endl;
	cout << toString(mOut) << endl << endl;
}
//////////////////////////////////////////////////////////////////////////////
void forward_conv2d_backprop_sgd()
{
	cout << "Forward Conv2D and Backpropagation test:" << endl << endl;

	MatrixFloat mIn, mOut, mKernel, mGradientOut, mGradientIn;
	mIn.setZero(5, 5);
	mIn(2, 2) = 1;
	mIn.resize(1, 5 * 5);

	mKernel.setZero(3, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	mKernel.resize(1, 3 * 3);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 1);

	//forward
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	//backpropagation
	mGradientOut = mOut* 0.1f;
	mGradientOut(3+1) = -1.f;
	conv2d.backpropagation(mIn, mGradientOut, mGradientIn);

	//disp forward
	mOut.resize(3, 3);
	cout << "Forward :" << endl;
	cout << toString(mOut) << endl << endl;

	//disp backpropagation
	conv2d.gradient_weights().resize(3, 3);
	cout << "Backprop Weight gradient :" << endl;
	cout << toString(conv2d.gradient_weights()) << endl << endl;

	mGradientIn.resize(5, 5);
	cout << "Backprop Input gradient :" << endl;
	cout << toString(mGradientIn) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void forward_conv2d_stride2_backprop_sgd()
{
	cout << "Forward Conv2D and Backpropagation test:" << endl << endl;

	MatrixFloat mIn, mOut, mKernel, mGradientOut, mGradientIn;
	mIn.setZero(5, 5);
	mIn(2, 2) = 1;
	mIn.resize(1, 5 * 5);

	mKernel.setZero(3, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	mKernel.resize(1, 3 * 3);

	LayerConvolution2D conv2d(5, 5, 1, 3, 3, 1, 2, 2);

	//forward
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	//backpropagation
	mGradientOut = mOut * 0.1f;
	mGradientOut(2 + 1) = -1.f;
	conv2d.backpropagation(mIn, mGradientOut, mGradientIn);

	//disp forward
	mOut.resize(2, 2);
	cout << "Forward :" << endl;
	cout << toString(mOut) << endl << endl;

	//disp backpropagation
	conv2d.gradient_weights().resize(3, 3);
	cout << "Backprop Weight gradient :" << endl;
	cout << toString(conv2d.gradient_weights()) << endl << endl;

	mGradientIn.resize(5, 5);
	cout << "Backprop Input gradient :" << endl;
	cout << toString(mGradientIn) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void forward_time()
{
	cout << "Forward conv2d time estimation" << endl;

	int iNbSamples = 32;
	int iInRows = 64;
	int iInCols = 64;
	int iInChannels = 16;

	int iKernelRows = 3;
	int iKernelCols = 3;
	int iOutChannels = 32;
	int iNbConv = 10;

	MatrixFloat mIn;
	mIn.setRandom(iNbSamples, iInRows*iInCols*iInChannels);

	MatrixFloat mOut;

	LayerConvolution2D conv2d(iInRows, iInCols, iInChannels, iKernelRows, iKernelCols, iOutChannels);

	chrono::steady_clock::time_point start = chrono::steady_clock::now();
	for(int i=0;i< iNbConv;i++)
		conv2d.forward(mIn, mOut);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	
	cout << "Time elapsed: " << delta << " ms" << endl;
}
/////////////////////////////////////////////////////////////////
int main()
{	
	im2col_col2im();
	simple_image_conv2d();
	batch_conv2d();
	image_2_input_channels_conv2d();
	image_2_output_channels_conv2d();
	simple_image_conv2d_stride2();
	forward_conv2d_backprop_sgd();
	simple_image_conv2d_stride2();
	forward_conv2d_stride2_backprop_sgd();
	forward_time();	
}
/////////////////////////////////////////////////////////////////
