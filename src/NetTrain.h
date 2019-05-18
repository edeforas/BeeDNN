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

class TrainOption
{
public:
    TrainOption():
		epochCallBack(nullptr)
    {
        epochs=100;
        batchSize=16;
        learningRate=0.001f;
        decay=0.9f;
        momentum=0.9f;
        keepBest = false;
        testEveryEpochs=-1;
    }

    int  epochs;
    int batchSize;
    float learningRate;
    float decay;
    float momentum;
	bool keepBest;
    int testEveryEpochs; //set to 1 to test at each epoch, 10 to test only 1/10 of the time, etc, set to -1 for no test //todo remove
 
	std::function<void()> epochCallBack;
};

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

    TrainResult train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt);
    TrainResult fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt);

	void set_optimizer(string sOptimizer); //ex "SGD" "Adam" "Nadam" "Nesterov" ...
	string get_optimizer() const;

	void set_loss(string sLoss); //ex "MeanSquareError" "CategorialCrossEntropy"
	string get_loss() const;

private:
	string _sOptimizer;
	Loss* _pLoss;
};

#endif
