#include <cmath>
#include <cstdlib>
#include <iostream>

#include "LayerActivation.h"
#include "Net.h"
#include "NetTrain.h"

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
	Net model;
	model.set_classification_mode(false);
	model.add(new LayerActivation("Identity"));

	MatrixFloat samples, truth;
	samples.setZero(3, 2);
	truth.resize(3, 2);
	truth(0, 0) = 1.f;
	truth(0, 1) = 0.f;
	truth(1, 0) = 0.f;
	truth(1, 1) = 2.f;
	truth(2, 0) = 1.f;
	truth(2, 1) = 1.f;

	NetTrain trainer;
	trainer.set_epochs(1);
	trainer.set_batchsize(2);
	trainer.set_validation_batchsize(2);
	trainer.set_loss("MeanSquaredError");
	trainer.set_train_data(samples, truth);
	trainer.set_validation_data(samples, truth);
	trainer.fit(model);

	const float expectedLoss = (0.5f + 2.f + 1.f) / 3.f;
	test(fabsf(trainer.get_current_train_loss() - expectedLoss) < 1.e-6f,
		"training loss averages samples across a partial batch");
	test(fabsf(trainer.get_current_validation_loss() - expectedLoss) < 1.e-6f,
		"validation loss averages samples across a partial batch");
	trainer.set_validation_batchsize(1);
	test(fabsf(trainer.compute_loss_accuracy(samples, truth) - expectedLoss) < 1.e-6f,
		"reported loss is independent of validation batch size");

	cout << "NetTrain loss aggregation test passed." << endl;
	return 0;
}
