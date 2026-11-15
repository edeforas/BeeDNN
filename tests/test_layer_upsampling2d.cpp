#include <iostream>
#include <string>
#include <cstdlib>

#include "LayerUpSampling2D.h"

using namespace std;
using namespace beednn;

void test(bool condition, const string& message)
{
	if (!condition)
	{
		cout << "Test failed: " << message << endl;
		exit(-1);
	}
}

int main()
{
	LayerUpSampling2D upsample(2, 2, 2);
	MatrixFloat input, output, gradientOut, gradientIn;
	input.resize(1, 8);
	for (Index i = 0; i < input.size(); i++)
		input(i) = (float)(i + 1);

	upsample.forward(input, output);
	test(output.rows() == 1 && output.cols() == 32, "output shape");
	for (Index channel = 0; channel < 2; channel++)
	{
		for (Index row = 0; row < 2; row++)
		{
			for (Index col = 0; col < 2; col++)
			{
				const float expected = input(channel * 4 + row * 2 + col);
				for (Index rowOffset = 0; rowOffset < 2; rowOffset++)
				{
					for (Index colOffset = 0; colOffset < 2; colOffset++)
					{
						const Index outIndex = channel * 16 + (row * 2 + rowOffset) * 4 + col * 2 + colOffset;
						test(output(outIndex) == expected, "nearest-neighbor forward values");
					}
				}
			}
		}
	}

	gradientOut.setOnes(1, 32);
	upsample.backpropagation(input, gradientOut, gradientIn);
	test(gradientIn.rows() == 1 && gradientIn.cols() == 8, "gradient shape");
	for (Index i = 0; i < gradientIn.size(); i++)
		test(gradientIn(i) == 4.f, "backpropagation sums repeated-pixel gradients");

	cout << "UpSampling2D tests passed." << endl;
	return 0;
}
