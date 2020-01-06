#include <iostream>
using namespace std;

#include "LayerConvolution2D.h"

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
/////////////////////////////////////////////////////////////////
int main()
{
	simple_image_conv2d();
	batch_conv2d();
	image_2_input_channels_conv2d();
	image_2_output_channels_conv2d();
}
/////////////////////////////////////////////////////////////////
