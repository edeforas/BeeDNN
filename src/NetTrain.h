/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef NetTrain_
#define NetTrain_

#include "Matrix.h"

#include <vector>
#include <functional>
#include <string>
using namespace std;

class Optimizer;
class Loss;

class TrainResult
{
public:
    TrainResult()
    { 
		finalLoss=-1.;
	}

    void reset()
    {
        loss.clear();
        accuracy.clear();
    }

    vector<double> loss;
    vector<double> accuracy;
    vector<double> euclidian_distance;
    double finalLoss;
};

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    float compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);

    TrainResult train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth);
    TrainResult fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth);

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

	void set_optimizer(string sOptimizer,float fLearningRate=-1.f, float fDecay = -1.f, float fMomentum = -1.f); //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov" ... -1.s is for default settings
//	void get_optimizer(string& sOptimizer, float& fLearningRate, float& fDecay, float& fMomentum) const;

	void set_batchsize(int iBatchSize); //16 by default
	int get_batchsize() const;

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs
	bool get_keepbest() const;

	void set_loss(string sLoss); // "MeanSquareError" by default, ex "MeanSquareError" "CategorialCrossEntropy"
	string get_loss() const;

private:
	bool _bKeepBest;
	int _iBatchSize;
	int _iEpochs;

	float _fLearningRate;
	float _fDecay;
	float _fMomentum;

	std::function<void()> _epochCallBack;

	string _sOptimizer;
	Loss* _pLoss;
};

#endif
