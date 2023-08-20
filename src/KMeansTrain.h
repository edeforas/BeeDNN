/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef KMeansTrain_
#define KMeansTrain_

#include "Matrix.h"

#include <functional>
#include <string>
using namespace std;

namespace bee{

class KMeans;
class KMeansTrain
{
public:
	KMeansTrain();
    virtual ~KMeansTrain();
	
	void set_kmeans(KMeans& km);
	KMeans& kmeans();
    void set_train_data(const bee::MatrixFloat& mSamples, const bee::MatrixFloat& mTruth);
	void set_validation_data(const bee::MatrixFloat& mSamplesValidation, const bee::MatrixFloat& mTruthValidation);

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs (evaluated on the test database)
	bool get_keepbest() const;

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

	void fit();
	
	float compute_accuracy(const bee::MatrixFloat & mSamples, const bee::MatrixFloat& mTruth) const;

	void set_batchsize(Eigen::Index iBatchSize); // 1024 by default
	Eigen::Index get_batchsize() const;

	float get_current_validation_accuracy() const;
	float get_current_train_accuracy() const;

protected:
	KMeans* _pKm;

private:
	Eigen::Index _iBatchSize;

	int _iEpochs;
	Eigen::Index _iValidationBatchSize;

	bool _bKeepBest;
    const bee::MatrixFloat* _pmSamplesTrain;
    const bee::MatrixFloat* _pmTruthTrain;

	const bee::MatrixFloat* _pmSamplesValidation;
	const bee::MatrixFloat* _pmTruthValidation;

	std::function<void()> _epochCallBack;

	float _fTrainAccuracy;
	float _fValidationAccuracy;
};
}
#endif
