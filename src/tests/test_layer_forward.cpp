#include <iostream>
using namespace std;

#include "LayerConvolution2D.h"

//////////////////////////////////////////////////////////////////////////////
void test_layer_conv2d()
{
	MatrixFloat mIn, mOut, mKernel;
	mIn.setZero(5, 5);
	mIn(2, 2) = 1;

	mKernel.setZero(3, 3);
	mKernel(1, 0) = 1;
	mKernel(1, 1) = 2;
	mKernel(1, 2) = 1;
	
	LayerConvolution2D conv2d(5,5,1,3,3,1);
	conv2d.weights() = mKernel;
	conv2d.forward(mIn, mOut);

	cout << toString(mOut) << endl;
}

int main()
{
	test_layer_conv2d();
}
