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

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();
	NetTrain& operator=(const NetTrain& other);

    void clear();
	
	void set_net(Net& net);
    void set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth);
	void set_test_data(const MatrixFloat& mSamplesTest, const MatrixFloat& mTruthTest);

	void train();

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

    /// reboost helps the optimizer to have a new fresh start every N epochs, improve Adamax for example
	void set_reboost_every_epochs(int iReboostEveryEpochs); //-1 by default -> disabled
	int get_reboost_every_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

    void set_optimizer(const string& sOptimizer); //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov" "iRPROP-" ...
    string get_optimizer() const;

    void set_learningrate(float fLearningRate=-1.f ); // -1.f is for default settings
    float get_learningrate() const;

    void set_decay( float fDecay = -1.f); // -1.s is for default settings
    float get_decay() const;

    void set_momentum( float fMomentum = -1.f); //" -1.f is for default settings
    float get_momentum() const;

	void set_batchsize(int iBatchSize); //32 by default
	int get_batchsize() const;

	void set_classbalancing(bool bBalancing); //true by default //use weight loss algorithm
	bool get_classbalancing() const;

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs (evaluated on the test database)
	bool get_keepbest() const;

	void set_loss(const string&  sLoss); // "MeanSquareError" by default, ex "MeanSquareError" "CategoricalCrossEntropy"
	string get_loss() const;

	float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth) const;
	float compute_accuracy(const MatrixFloat & mSamples, const MatrixFloat& mTruth) const;

	const vector<float>& get_train_loss() const;
	const vector<float>& get_test_loss() const;
    const vector<float>& get_train_accuracy() const;
    const vector<float>& get_test_accuracy() const;

	float get_current_loss() const;
	float get_current_accuracy() const;

private:
	void train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth); //all the backprop is here	
    void update_class_weight(); // compute balanced class weight loss (if asked) and update loss
	void add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth);	//online statistics, i.e. loss, accuracy ..
	void clear_optimizers();

	int _iOnlineAccuracyGood;
	float _fOnlineLoss;

	bool _bKeepBest;
	int _iBatchSize;
	int _iEpochs;
	bool _bClassBalancingWeightLoss;
	int _iNbLayers;
	int _iReboostEveryEpochs;

    string _sOptimizer;
    float _fLearningRate;
	float _fDecay;
	float _fMomentum;

	Net* _pNet;
	Loss* _pLoss;

	vector<Optimizer*> _optimizers;
	vector<MatrixFloat> _inOut;
	vector<MatrixFloat> _gradient;

    const MatrixFloat* _pmSamplesTrain;
    const MatrixFloat* _pmTruthTrain;

	const MatrixFloat* _pmSamplesTest;
	const MatrixFloat* _pmTruthTest;

	std::function<void()> _epochCallBack;

    vector<float> _trainLoss;
    vector<float> _testLoss;
    vector<float> _trainAccuracy;
    vector<float> _testAccuracy;

	float _fCurrentLoss;
	float _fCurrentAccuracy;
};

#endif
