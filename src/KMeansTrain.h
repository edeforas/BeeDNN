/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef KMeansTrain_
#define KMeansTrain_

#include "Matrix.h"

#include <vector>
#include <functional>
#include <string>
using namespace std;

class Loss;
class KMeans;

class KMeansTrain
{
public:
	KMeansTrain();
    virtual ~KMeansTrain();
	//KMeansTrain& operator=(const KMeansTrain& other);

    void clear();
	
	void set_kmeans(KMeans& km);
	KMeans& kmeans();
    void set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth);
	void set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation);

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

	void set_loss(const string&  sLoss); // "MeanSquareError" by default, ex "MeanSquareError" "CategoricalCrossEntropy"
	string get_loss() const;

	void train();
	/*
	float compute_loss_accuracy(const MatrixFloat & mSamples, const MatrixFloat& mTruth,float* pfAccuracy = nullptr) const;

	const vector<float>& get_train_loss() const;
	const vector<float>& get_validation_loss() const;
	const vector<float>& get_train_accuracy() const;
	const vector<float>& get_validation_accuracy() const;

	float get_current_train_loss() const;
	float get_current_train_accuracy() const;
	float get_current_validation_loss() const;
	float get_current_validation_accuracy() const;
	*/
	virtual void train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth); //all the backprop is here	

protected:
	virtual void train_one_epoch(const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled);
//	void add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth);	//online statistics, i.e. loss, accuracy ..
	Loss* _pLoss;
	KMeans* _pKm;

private:

	int _iOnlineAccuracyGood;
	float _fOnlineLoss;

	int _iEpochs;

    const MatrixFloat* _pmSamplesTrain;
    const MatrixFloat* _pmTruthTrain;

	const MatrixFloat* _pmSamplesValidation;
	const MatrixFloat* _pmTruthValidation;

	std::function<void()> _epochCallBack;

    vector<float> _trainLoss;
    vector<float> _trainAccuracy;
	
	vector<float> _validationLoss;
	vector<float> _validationAccuracy;

	float _fTrainLoss;
	float _fTrainAccuracy;

	float _fValidationLoss;
	float _fValidationAccuracy;
};

#endif
