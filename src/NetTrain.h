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
class Net;

class TrainResult //todo remove
{
public:
    vector<float> loss;
    vector<float> accuracy;
};

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    void clear();
    float compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);
    float compute_accuracy(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);

    TrainResult train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth);
    TrainResult fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth);

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

    /// reboost helps the optimizer to have a new fresh start every N epochs, improve Adamax, for example
	void set_reboost_every_epochs(int iReboostEveryEpochs); //-1 by default -> disabled
	int get_reboost_every_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

    void set_optimizer(string sOptimizer); //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov" ...
    string get_optimizer() const;

    void set_learningrate(float fLearningRate=-1.f ); // -1.f is for default settings
    float get_learningrate() const;

    void set_decay( float fDecay = -1.f); // -1.s is for default settings
    float get_decay() const;

    void set_momentum( float fMomentum = -1.f); //" -1.f is for default settings
    float get_momentum() const;

	void set_batchsize(int iBatchSize); //16 by default
	int get_batchsize() const;

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs
	bool get_keepbest() const;

	void set_loss(string sLoss); // "MeanSquareError" by default, ex "MeanSquareError" "CategorialCrossEntropy"
	string get_loss() const;

	void set_problem(bool bClassification);
	bool is_classification_problem();

private:
	void train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth);

	//online statistics, i.e. loss, accuracy ...
	void add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth);
	int _iOnlineAccuracyGood;
	float _fOnlineLoss;

	bool _bKeepBest;
	int _iBatchSize;
	int _iEpochs;
	int _iNbLayers;
	int _iReboostEveryEpochs;

    string _sOptimizer;
    float _fLearningRate;
	float _fDecay;
	float _fMomentum;
    bool _bIsclassificationProblem;

	vector<Optimizer*> _optimizers;
	vector<MatrixFloat> _inOut;
	vector<MatrixFloat> _delta;

	std::function<void()> _epochCallBack;

	Net* _pNet;
	Loss* _pLoss;
};

#endif
