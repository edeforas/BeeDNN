#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "LayerTransformerEncoder.h"
#include "LayerDense.h"
#include "LayerSoftmax.h"
#include "Net.h"
#include "NetTrain.h"

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

float objective(LayerTransformerEncoder& layer, const MatrixFloat& input, const MatrixFloat& gradient)
{
	MatrixFloat output;
	layer.forward(input, output);
	float value = 0.f;
	for (Index i = 0; i < output.size(); i++)
		value += output(i) * gradient(i);
	return value;
}

int main()
{
	LayerTransformerEncoder layer(3, 4, 5, 2);
	MatrixFloat input, output, gradientOut, gradientIn;
	input.resize(1, 12);
	gradientOut.resize(1, 12);
	for (Index i = 0; i < input.size(); i++)
	{
		input(i) = 0.1f * (float)(i + 1) - 0.45f;
		gradientOut(i) = 0.07f * (float)(i - 4);
	}

	layer.forward(input, output);
	test(output.rows() == 1 && output.cols() == 12, "output shape");
	for (Index i = 0; i < output.size(); i++)
		test(std::isfinite(output(i)), "finite forward output");

	layer.backpropagation(input, gradientOut, gradientIn);
	test(gradientIn.rows() == 1 && gradientIn.cols() == 12, "input gradient shape");

	MatrixFloat* weights = layer.weights()[0];
	const MatrixFloat& gradients = *layer.gradient_weights()[0];
	const Index weightIndices[] = { 0, 3 * 4 * 4, 4 * 4 * 4, weights->size() - 1 };
	const float epsilon = 1.e-3f;
	for (Index index : weightIndices)
	{
		const float original = (*weights)(index);
		(*weights)(index) = original + epsilon;
		const float plus = objective(layer, input, gradientOut);
		(*weights)(index) = original - epsilon;
		const float minus = objective(layer, input, gradientOut);
		(*weights)(index) = original;
		const float numerical = (plus - minus) / (2.f * epsilon);
		test(std::fabs(numerical - gradients(index)) < 2.e-2f, "weight gradient matches finite differences");
	}

	MatrixFloat* biases = layer.biases()[0];
	const MatrixFloat& biasGradients = *layer.gradient_biases()[0];
	const Index biasIndices[] = { 0, 5 * 4 + 5, 8 * 4 + 5 };
	for (Index index : biasIndices)
	{
		const float original = (*biases)(index);
		(*biases)(index) = original + epsilon;
		const float plus = objective(layer, input, gradientOut);
		(*biases)(index) = original - epsilon;
		const float minus = objective(layer, input, gradientOut);
		(*biases)(index) = original;
		const float numerical = (plus - minus) / (2.f * epsilon);
		test(std::fabs(numerical - biasGradients(index)) < 2.e-2f, "bias gradient matches finite differences");
	}

	const Index inputIndex = 5;
	const float originalInput = input(inputIndex);
	input(inputIndex) = originalInput + epsilon;
	const float plusInput = objective(layer, input, gradientOut);
	input(inputIndex) = originalInput - epsilon;
	const float minusInput = objective(layer, input, gradientOut);
	input(inputIndex) = originalInput;
	const float numericalInput = (plusInput - minusInput) / (2.f * epsilon);
	test(std::fabs(numericalInput - gradientIn(inputIndex)) < 2.e-2f, "input gradient matches finite differences");

	MatrixFloat mask;
	mask.setZero(3, 3);
	for (Index token = 0; token < 3; token++)
		mask(token, token) = 1.f;
	LayerTransformerEncoder masked(3, 4, 5, 2);
	masked.set_attention_mask(mask);
	MatrixFloat maskedOutput, changedInput = input, changedOutput;
	masked.forward(input, maskedOutput);
	for (Index feature = 0; feature < 4; feature++)
		changedInput(4 + feature) += 1.f;
	masked.forward(changedInput, changedOutput);
	for (Index feature = 0; feature < 4; feature++)
		test(std::fabs(maskedOutput(feature) - changedOutput(feature)) < 1.e-6f,
			"custom attention mask blocks other tokens");

	LayerTransformerEncoder causal(3, 4, 5, 2);
	causal.set_causal_mask(true);
	test(causal.causal_mask(), "causal mask setting");
	MatrixFloat causalOutput;
	causal.forward(input, causalOutput);
	causal.forward(changedInput, changedOutput);
	for (Index feature = 0; feature < 4; feature++)
		test(std::fabs(causalOutput(feature) - changedOutput(feature)) < 1.e-6f,
			"causal attention blocks future tokens");

	LayerTransformerEncoder regularized(3, 4, 5, 2, 0.5f);
	regularized.set_train_mode(true);
	regularized.forward(input, output);
	regularized.backpropagation(input, gradientOut, gradientIn);
	for (Index i = 0; i < output.size(); i++)
		test(std::isfinite(output(i)) && std::isfinite(gradientIn(i)), "finite dropout forward and backward");
	regularized.set_train_mode(false);
	regularized.forward(input, output);
	MatrixFloat outputAgain;
	regularized.forward(input, outputAgain);
	for (Index i = 0; i < output.size(); i++)
		test(output(i) == outputAgain(i), "attention dropout disabled during inference");

	Net model;
	model.add(new LayerTransformerEncoder(3, 4, 5, 2));
	model.add(new LayerDense(12, 2));
	model.add(new LayerSoftmax());
	MatrixFloat trainData, trainLabels;
	trainData.resize(4, 12);
	trainLabels.resize(4, 1);
	for (Index sample = 0; sample < trainData.rows(); sample++)
	{
		trainLabels(sample) = (float)(sample % 2);
		for (Index feature = 0; feature < trainData.cols(); feature++)
			trainData(sample, feature) = input(feature) + 0.05f * (float)sample;
	}

	NetTrain trainer;
	trainer.set_epochs(1);
	trainer.set_batchsize(2);
	trainer.set_loss("SparseCategoricalCrossEntropy");
	trainer.set_train_data(trainData, trainLabels);
	trainer.fit(model);
	MatrixFloat predictedClasses;
	model.predict_classes(trainData, predictedClasses);
	test(predictedClasses.rows() == trainData.rows() && predictedClasses.cols() == 1, "training integration");

	cout << "TransformerEncoder tests passed." << endl;
	return 0;
}
