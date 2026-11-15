#include <cmath>
#include <cstdlib>
#include <iostream>

#include "LayerRandomAffine2D.h"

using namespace std;
using namespace beednn;

void test(bool condition, const char* message)
{
	if (!condition)
	{
		cout << "Test failed: " << message << endl;
		exit(-1);
	}
}

int main()
{
	const Index rows = 7;
	const Index cols = 7;
	const Index channels = 2;
	const Index sampleSize = rows * cols * channels;
	MatrixFloat input, output, gradientOut, gradientIn;
	input.resize(2, sampleSize);
	gradientOut.resize(2, sampleSize);
	for (Index i = 0; i < input.size(); i++)
	{
		input(i) = 0.01f * (float)(i + 1);
		gradientOut(i) = 0.02f * (float)((i % 11) - 5);
	}

	LayerRandomAffine2D augmentation(rows, cols, channels, 10.f, 1.5f, 1.5f);
	augmentation.forward(input, output);
	for (Index i = 0; i < input.size(); i++)
		test(output(i) == input(i), "augmentation is disabled in inference mode");

	augmentation.set_train_mode(true);
	augmentation.forward(input, output);
	float difference = 0.f;
	for (Index i = 0; i < input.size(); i++)
		difference += fabsf(output(i) - input(i));
	test(difference > 1.e-3f, "training applies random affine transforms");

	augmentation.backpropagation(input, gradientOut, gradientIn);
	float outputGradientProduct = 0.f;
	float inputGradientProduct = 0.f;
	for (Index i = 0; i < output.size(); i++)
	{
		outputGradientProduct += output(i) * gradientOut(i);
		inputGradientProduct += input(i) * gradientIn(i);
	}
	test(fabsf(outputGradientProduct - inputGradientProduct) < 1.e-4f,
		"backpropagation is the adjoint of bilinear resampling");

	augmentation.set_train_mode(false);
	augmentation.forward(input, output);
	for (Index i = 0; i < input.size(); i++)
		test(output(i) == input(i), "inference remains unaugmented after training");

	cout << "RandomAffine2D tests passed." << endl;
	return 0;
}
