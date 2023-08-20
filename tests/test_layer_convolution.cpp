#include <iostream>
#include <chrono>
using namespace std;

#include "LayerConvolution2D.h"
using namespace bee;

void set_weight(Layer& l, MatrixFloat& m)
{
	*(l.weights()[0]) = m;
}

MatrixFloat& get_gradient(Layer& l)
{
	return *(l.gradient_weights()[0]);
}
//////////////////////////////////////////////////////////////////////////////
void compare_im2col()
{
	cout << "Comparing im2col() and im2col_LUT():" << endl;

	Index iNbSamples=7, inRows = 31, inCols = 23, inChannels = 13, outChannels = 17; // all primes numbers
	MatrixFloat mIn, mCol, mColLUT, mIm, mImLUT;

	//fill with random data
	mIn.resize(iNbSamples, inRows * inCols*inChannels);
	mIn.setRandom();

	//compare legacy and optimized forward computation
	LayerConvolution2D conv2d(inRows, inCols, inChannels, 5, 3, outChannels);
	conv2d.im2col(mIn, mCol);
	conv2d.im2col_LUT(mIn, mColLUT);
	float fMaxDiff = (mCol - mColLUT).cwiseAbs().maxCoeff();

	//mIn.resize(iNbSamples*inRows*inChannels, inCols);
	//cout << "Image:" << endl << toString(mIn) << endl << endl;
	//cout << "Im2Col:" << endl << toString(mCol) << endl << endl;
	//cout << "Im2ColLUT:" << endl << toString(mColLUT) << endl << endl;

	//testu function
	if (fMaxDiff > 1.e-10)
	{
		cout << "Test failed! MaxDifference = " << fMaxDiff << endl;
		exit(-1);
	}
	else
		cout << "Test Succeded. MaxDifference = " << fMaxDiff << endl;
}

//////////////////////////////////////////////////////////////////////////////
void compare_fastlut_slow_computation()
{
	cout << "Comparing fastlut and slow computation mode:" << endl;

	Index iNbSamples = 7, inRows = 31, inCols = 23, inChannels = 13, outChannels = 17; // all primes numbers
	MatrixFloat mIn, mOut, mOutFast, mIm, mImFast;

	//fill with random data
	mIn.resize(iNbSamples, inRows * inCols*inChannels);
	mIn.setRandom();

	LayerConvolution2D conv2d(inRows, inCols, inChannels, 5, 3, outChannels);
	conv2d.fastLUT = false;
	conv2d.forward(mIn, mOut);
	conv2d.fastLUT = true;
	conv2d.forward(mIn, mOutFast);

	conv2d.fastLUT = false;
	conv2d.backpropagation(mIn, mOut, mIm);
	conv2d.fastLUT = true;
	conv2d.backpropagation(mIn, mOutFast, mImFast);
	
	float fMaxDiffOut = (mOut - mOutFast).cwiseAbs().maxCoeff();
	float fMaxDiffIm = (mIm - mImFast).cwiseAbs().maxCoeff();

	//testu function
	if (fMaxDiffOut > 1.e-10)
	{
		cout << "Test failed! MaxDifferenceOut = " << fMaxDiffOut << endl;
		exit(-1);
	}
	else
		cout << "Test Succeded. MaxDifferenceOut = " << fMaxDiffOut << endl;

	//testu function
	if (fMaxDiffIm > 1.e-6)
	{
		cout << "Test failed! MaxDifferenceIm = " << fMaxDiffIm << endl;
		exit(-1);
	}
	else
		cout << "Test Succeded. MaxDifferenceIm = " << fMaxDiffIm << endl;
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
	set_weight(conv2d,mKernel);
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
	set_weight(conv2d, mKernel);
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
	set_weight(conv2d, mKernel);
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
	set_weight(conv2d, mKernel);
	conv2d.forward(mIn, mOut);

	mOut.resize(6, 3);
	cout << "Image 2 output channels convoluted:" << endl;
	cout << toString(mOut) << endl << endl;
}
//////////////////////////////////////////////////////////////////////////////
void forward_backward()
{
	cout << "Forward then backward test:" << endl;

	Index iNbSamples = 5, inRows = 7, inCols = 11, inChannels = 13, outChannels = 17; // all primes numbers
	MatrixFloat mIn, mCol, mColLUT, mIm, mImLUT;

	//fill with incremented data
	mIn.resize(iNbSamples, inRows * inCols * inChannels);
	for (Index i = 0; i < mIn.size(); i++)
		mIn.data()[i] = (float)i;
	
	//forward and backward
	LayerConvolution2D conv2d(inRows, inCols, inChannels, 2, 3, outChannels);
	conv2d.forward(mIn, mCol);
	conv2d.backpropagation(mIn, mCol, mIm);

	cout << "mIm:" << endl << toString(mIm) << endl << endl;
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
	set_weight(conv2d, mKernel);
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
	set_weight(conv2d, mKernel);
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
	MatrixFloat& gw = get_gradient(conv2d);
	gw.resize(3, 3);
	cout << "Backprop Weight gradient :" << endl;
	cout << toString(gw) << endl << endl;

	mGradientIn.resize(5, 5);
	cout << "Backprop Input gradient :" << endl;
	cout << toString(mGradientIn) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void forward_stride2_backward()
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
	set_weight(conv2d, mKernel);
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
	MatrixFloat& gw = get_gradient(conv2d);
	gw.resize(3, 3);
	cout << "Backprop Weight gradient :" << endl;
	cout << toString(gw) << endl << endl;

	mGradientIn.resize(5, 5);
	cout << "Backprop Input gradient :" << endl;
	cout << toString(mGradientIn) << endl << endl;
}
/////////////////////////////////////////////////////////////////
void forward_time()
{
	cout << "Forward conv2d time estimation:" << endl;

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

	//measure forward time slow
	conv2d.fastLUT = false;
	auto start = chrono::steady_clock::now();
	for(int i=0;i< iNbConv;i++)
		conv2d.forward(mIn, mOut);
	auto end = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Time elapsed slow: " << delta << " ms" << endl;

	//measure forward time fastlut
	conv2d.fastLUT = true;
	start = chrono::steady_clock::now();
	for (int i = 0; i < iNbConv; i++)
		conv2d.forward(mIn, mOut);
	end = chrono::steady_clock::now();
	delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Time elapsed fastlut: " << delta << " ms" << endl << endl;
}
/////////////////////////////////////////////////////////////////
void backward_time()
{
	cout << "Backward conv2d time estimation:" << endl;

	int iNbSamples = 32;
	int iInRows = 64;
	int iInCols = 64;
	int iInChannels = 16;

	int iKernelRows = 3;
	int iKernelCols = 3;
	int iOutChannels = 32;
	int iNbConv = 10;

	MatrixFloat mIn,mOut, mOutGradient, mInGradient;
	mIn.setRandom(iNbSamples, iInRows*iInCols*iInChannels);

	LayerConvolution2D conv2d(iInRows, iInCols, iInChannels, iKernelRows, iKernelCols, iOutChannels);
	conv2d.forward(mIn, mOut); // init backward internal state
	
	 //create random gradient
	mOutGradient = mOut;
	mOutGradient.setRandom();
	
	//measure backward time slow
	conv2d.fastLUT = false;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < iNbConv; i++)
		conv2d.backpropagation(mIn, mOutGradient, mInGradient);
	auto end = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Time elapsed slow: " << delta << " ms" << endl;

	conv2d.fastLUT = true;
	 start = chrono::steady_clock::now();
	for (int i = 0; i < iNbConv; i++)
		conv2d.backpropagation(mIn, mOutGradient, mInGradient);
	 end = chrono::steady_clock::now();
	delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Time elapsed fast: " << delta << " ms" << endl << endl;

}
/////////////////////////////////////////////////////////////////
int main()
{	
	compare_im2col(); 
	compare_fastlut_slow_computation();
	simple_image_conv2d();
	batch_conv2d();
	image_2_input_channels_conv2d();
	image_2_output_channels_conv2d();
	simple_image_conv2d_stride2();
	forward_conv2d_backprop_sgd();
	simple_image_conv2d_stride2();
	
	forward_backward();
	forward_stride2_backward();
	forward_time();	
	backward_time();
}
/////////////////////////////////////////////////////////////////
