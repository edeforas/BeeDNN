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

namespace bee {

class KMeans;
class KMeansTrain
{
public:
	KMeansTrain();
    virtual ~KMeansTrain();
	
	void set_kmeans(KMeans& km);
	KMeans& kmeans();
    void set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth);
	void set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation);

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs (evaluated on the test database)
	bool get_keepbest() const;

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

	void fit();
	
	float compute_accuracy(const MatrixFloat & mSamples, const MatrixFloat& mTruth) const;

	void set_batchsize(Index iBatchSize); // 1024 by default
	Index get_batchsize() const;

	float get_current_validation_accuracy() const;
	float get_current_train_accuracy() const;

protected:
	KMeans* _pKm;

private:
	Index _iBatchSize;

	int _iEpochs;
	Index _iValidationBatchSize;

	bool _bKeepBest;
    const MatrixFloat* _pmSamplesTrain;
    const MatrixFloat* _pmTruthTrain;

	const MatrixFloat* _pmSamplesValidation;
	const MatrixFloat* _pmTruthValidation;

	std::function<void()> _epochCallBack;

	float _fTrainAccuracy;
	float _fValidationAccuracy;
};
}
#endif
