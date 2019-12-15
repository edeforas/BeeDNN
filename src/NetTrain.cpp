/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetTrain.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"

#include "Optimizer.h"
#include "Loss.h"

#include <cmath>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain():
    _sOptimizer("Adam")
{
	_epochCallBack = nullptr;
	_fTrainLoss=0.f;
	_fTrainAccuracy=0.f;

	_fTestLoss=0.f;
	_fTestAccuracy=0.f;

    _pLoss = create_loss("MeanSquaredError");
    _iBatchSize = 32;
    _bKeepBest = true;
    _iEpochs = 100;
    _iReboostEveryEpochs = -1; // -1 mean no reboost
	_iOnlineAccuracyGood= 0;

    _fLearningRate = -1.f; //default
    _fDecay = -1.f; //default
    _fMomentum = -1.f; //default

	_bClassBalancingWeightLoss = false;

	_iNbLayers=0;
	_fOnlineLoss = 0.f;
	_pNet = nullptr;

    _pmSamplesTrain = nullptr;
    _pmTruthTrain = nullptr;

	_pmSamplesTest = nullptr;
	_pmTruthTest = nullptr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::~NetTrain()
{
	clear_optimizers();
    delete _pLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::clear()
{ 
	clear_optimizers();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::clear_optimizers()
{ 
	for (unsigned int i = 0; i < _optimizers.size(); i++)
		delete _optimizers[i];

	_optimizers.clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain& NetTrain::operator=(const NetTrain& other)
{
    clear();

    set_keepbest(other._bKeepBest);
	set_classbalancing(other._bClassBalancingWeightLoss);
    set_batchsize(other._iBatchSize);
    set_epochs(other._iEpochs);
	set_reboost_every_epochs(other._iReboostEveryEpochs);
	set_loss(other._pLoss->name());

	_iOnlineAccuracyGood = other._iOnlineAccuracyGood;
	_fOnlineLoss = other._fOnlineLoss;

	_fTrainLoss = other._fTrainLoss;
	_fTrainAccuracy = other._fTrainAccuracy;
	_fTestLoss = other._fTestLoss;
	_fTestAccuracy = other._fTestAccuracy;
	_iNbLayers=other._iNbLayers;

	set_optimizer(other._sOptimizer);
    _fLearningRate=other._fLearningRate;
    _fDecay=other._fDecay;
    _fMomentum=other._fMomentum;

    _inOut = other._inOut;
    _gradient = other._gradient;


//	set_train_data(other._pmSamplesTrain, other._pmTruthTrain);
//	set_test_data(other._pmSamplesTest, other._pmTruthTest);

    _pmSamplesTrain = other._pmSamplesTrain;
    _pmTruthTrain = other._pmTruthTrain;

	_pmSamplesTest = other._pmSamplesTest;
	_pmTruthTest = other._pmTruthTest;
	
//	_epochCallBack = other._epochCallBack;
//	set_net(other._pNet);
/*
	_trainLoss= other._trainLoss;
	_testLoss= other._testLoss;
	_trainAccuracy= other._trainAccuracy;
	_testAccuracy= other._testAccuracy;
*/
	return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_net(Net& net)
{
	_pNet = &net;
	_iNbLayers = (int)_pNet->layers().size();
	if (_iNbLayers != 0)
		_pNet->layers()[0]->set_first_layer(true);
	
	clear_optimizers();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net& NetTrain::net()
{
	return *_pNet;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_optimizer(const string& sOptimizer) //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov"
{
    _sOptimizer = sOptimizer;
	clear_optimizers();
}
string NetTrain::get_optimizer() const
{
    return _sOptimizer;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_learningrate(float fLearningRate) //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov"
{
    _fLearningRate = fLearningRate;
}
float NetTrain::get_learningrate() const
{
    return _fLearningRate;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_decay(float fDecay) // -1.f is for default settings
{
    _fDecay=fDecay;
}
float NetTrain::get_decay() const
{
    return _fDecay;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_momentum(float fMomentum) // -1.f is for default settings
{
    _fMomentum=fMomentum;
}
float NetTrain::get_momentum() const
{
    return _fMomentum;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_epochs(int iEpochs) //100 by default
{
    _iEpochs = iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int NetTrain::get_epochs() const
{
    return _iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_reboost_every_epochs(int iReboostEveryEpochs) //-1 by default -> disabled
{
    _iReboostEveryEpochs = iReboostEveryEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int NetTrain::get_reboost_every_epochs() const
{
    return _iReboostEveryEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_epoch_callback(std::function<void()> epochCallBack)
{
    _epochCallBack = epochCallBack;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_loss(const string&  sLoss)
{
    delete _pLoss;
    _pLoss = create_loss(sLoss);
	assert(_pLoss);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
string NetTrain::get_loss() const
{
    return _pLoss->name();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_batchsize(int iBatchSize) //16 by default
{
    _iBatchSize = iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int NetTrain::get_batchsize() const
{
    return _iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_classbalancing(bool bBalancing) //true by default
{
	_bClassBalancingWeightLoss = bBalancing;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool NetTrain::get_classbalancing() const
{
	return _bClassBalancingWeightLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_keepbest(bool bKeepBest) //true by default
{
    _bKeepBest = bKeepBest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool NetTrain::get_keepbest() const
{
    return _bKeepBest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::compute_loss(const MatrixFloat &mSamples, const MatrixFloat &mTruth) const
{
	assert(_pNet);

	if (!_pNet->is_valid((int)mSamples.cols(), (int)mTruth.cols()))
		return 0.f;

    int iNbSamples = (int)mSamples.rows();

    if( (_pNet->layers().size()==0) || (iNbSamples==0) )
        return 0.f;

    MatrixFloat mOut;
	_pNet->forward(mSamples, mOut);
    return _pLoss->compute(mOut,mTruth);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::compute_accuracy(const MatrixFloat &mSamples, const MatrixFloat &mTruth) const
{
    int iNbSamples = (int)mSamples.rows();

    if ((_pNet->size() == 0) || (iNbSamples == 0))
        return 0.f;

    MatrixFloat mOut;
	_pNet->forward(mSamples, mOut);

    if (mTruth.cols() != 1)
    {
		if (mOut.cols() == mTruth.cols())
		{
			//one hot everywhere
			int iGood = 0;
			for (int i = 0; i < iNbSamples; i++)
			{
				iGood += (argmax(mOut.row(i)) == argmax(mTruth.row(i)));
			}

			return 100.f*iGood / iNbSamples;
		}
       return 0.f;   //todo
    }

    int iGood=0;
    if(mOut.cols()!=1)
    {
        for(int i=0;i<iNbSamples;i++)
            iGood += (argmax(mOut.row(i)) ==mTruth(i));
    }
    else
    {
        for(int i=0;i<iNbSamples;i++)
            iGood += roundf(mOut(i))==mTruth(i);
    }

    return 100.f*iGood /iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth)
{
    _pmSamplesTrain = &mSamples;
    _pmTruthTrain = &mTruth;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_test_data(const MatrixFloat& mSamplesTest, const MatrixFloat& mTruthTest)
{
	_pmSamplesTest = &mSamplesTest;
	_pmTruthTest = &mTruthTest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::train()
{
//	MatrixFloat mTruthOneHot;

	if (_pNet == nullptr)
		return;

	if (_pNet->layers().size() == 0)
		return; //nothing to do

	update_class_weight();

    const MatrixFloat& mSamples = *_pmSamplesTrain;
    const MatrixFloat& mTruth = *_pmTruthTrain;

	if (_pNet->input_size() != (int)mSamples.cols())
		_pNet->set_input_shape((int)mSamples.cols());

	if (!_pNet->is_valid((int)mSamples.cols(),(int) mTruth.cols()))
		return ;

    _trainLoss.clear();
    _testLoss.clear();
    _trainAccuracy.clear();
    _testAccuracy.clear();
	_fTrainLoss = 1.e10f;
	_fTrainAccuracy = 0;
	_fTestLoss = 1.e10f;
	_fTestAccuracy = 0;

    int iNbSamples=(int)mSamples.rows();
    int iReboost = 0;

    Net bestNet;

    //accept batch size == 0 or greater than nb samples  -> full size
    int iBatchSizeLocal=_iBatchSize;
    if( (iBatchSizeLocal >iNbSamples) || (iBatchSizeLocal ==0) )
        iBatchSizeLocal =iNbSamples;

    _inOut.resize(_iNbLayers + 1);

    _gradient.resize(_iNbLayers +1);

	if (_optimizers.empty())
	{
		//init all optimizers
		for (int i = 0; i < _iNbLayers; i++)
		{
			_optimizers.push_back(create_optimizer(_sOptimizer));
			_optimizers[i]->set_params(_fLearningRate, _fDecay, _fMomentum);
		}
	}

    //compute the accuracy at epoch 0, if keepbest is selected
	float fMaxAccuracy = 0.f;
	float fMinLoss = 1.e10f;
	if(_bKeepBest)
    {
        if (_pmSamplesTest == nullptr)
        {
            fMaxAccuracy = compute_accuracy( *_pmSamplesTrain, *_pmTruthTrain);
            fMinLoss=compute_loss(*_pmSamplesTrain, *_pmTruthTrain);
        }
        else
        {
            fMaxAccuracy = compute_accuracy( *_pmSamplesTest, *_pmTruthTest);
            fMinLoss=compute_loss( *_pmSamplesTest, *_pmTruthTest);
        }
        bestNet= *_pNet;
    }

    for(int iEpoch=0;iEpoch<_iEpochs;iEpoch++)
    {
        _fOnlineLoss=0.f;
        _iOnlineAccuracyGood = 0;

        MatrixFloat mSampleShuffled;
        MatrixFloat mTruthShuffled;

        if (iBatchSizeLocal < iNbSamples)
        {
            auto vShuffle = randPerm(iNbSamples);
            applyRowPermutation(vShuffle, mSamples, mSampleShuffled);
            applyRowPermutation(vShuffle, mTruth, mTruthShuffled);
        }
        else
        {
            mSampleShuffled = mSamples; //todo remove copy
            mTruthShuffled = mTruth;
        }

		_pNet->set_train_mode(true);

        int iBatchStart=0;

        while(iBatchStart<iNbSamples)
        {
            int iBatchEnd=iBatchStart+iBatchSizeLocal;
            if(iBatchEnd>iNbSamples)
                iBatchEnd=iNbSamples;

            const MatrixFloat mSample = rowRange(mSampleShuffled, iBatchStart, iBatchEnd);
            const MatrixFloat mTarget = rowRange(mTruthShuffled, iBatchStart, iBatchEnd);

			train_batch(mSample, mTarget);
		
            iBatchStart=iBatchEnd;
        }

		_pNet->set_train_mode(false);

		_fTrainLoss =_fOnlineLoss/iNbSamples;
        _trainLoss.push_back(_fTrainLoss);

        if(_pNet->is_classification_mode())
        {
			_fTrainAccuracy =100.f*_iOnlineAccuracyGood/ iNbSamples;
            _trainAccuracy.push_back(_fTrainAccuracy);
        }
		float fSelectedLoss = _fTrainLoss;
		float fSelectedAccuracy = _fTrainAccuracy;

		// if having test data, compute stats with it
        if (_pmSamplesTest != nullptr)
        { 	
			//use the test_db to keep the best model
			_fTestLoss =compute_loss( *_pmSamplesTest, *_pmTruthTest);
            _testLoss.push_back(_fTestLoss);

			_fTestAccuracy = compute_accuracy( *_pmSamplesTest, *_pmTruthTest);
            _testAccuracy.push_back(_fTestAccuracy);

			fSelectedLoss = _fTestLoss;
			fSelectedAccuracy = _fTestAccuracy;
		}

        if (_epochCallBack)
            _epochCallBack();

        //keep the best model if asked
        if(_bKeepBest) // todo do it after n epochs
        {
            if(_pNet->is_classification_mode())
            {   //use accuracy
                if(fMaxAccuracy< fSelectedAccuracy)
                {
                    fMaxAccuracy= fSelectedAccuracy;
                    bestNet= *_pNet;
                }
            }
            else
            {   //use loss
                if(fMinLoss> fSelectedLoss)
                {
                    fMinLoss= fSelectedLoss;
                    bestNet= *_pNet;
                }
            }
        }

        //reboost optimizers every epochs if asked
        if (_iReboostEveryEpochs != -1)
        {
            if (iReboost < _iReboostEveryEpochs)
                iReboost++;
            else
            {
                iReboost = 0;
                for (int i = 0; i < _iNbLayers; i++)
                    _optimizers[i]->init();
            }
        }
    }

	if(_bKeepBest)
		(*_pNet).operator=(bestNet);

    return ;
}
/////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth)
{
	//forward pass with store
	_inOut[0] = mSample;
	for (int i = 0; i < _iNbLayers; i++)
		_pNet->layer(i).forward(_inOut[i], _inOut[i + 1]);

	//compute error gradient
	_pLoss->compute_gradient(_inOut[_iNbLayers], mTruth, _gradient[_iNbLayers]);

	//backward pass with optimizer
	for (int i = _iNbLayers - 1; i >= 0; i--)
	{
		Layer& l = _pNet->layer(i);
		l.backpropagation(_inOut[i], _gradient[i + 1], _gradient[i]);

		if (l.has_weight())
		{
			_optimizers[i]->optimize(l.weights(), l.gradient_weights());
		}
	}

	//compute and save statistics
	add_online_statistics(_inOut[_iNbLayers], mTruth);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth )
{
    //update loss
    _fOnlineLoss += _pLoss->compute(mPredicted, mTruth);

    if (!_pNet->is_classification_mode())
        return;

    int iNbRows = (int)mPredicted.rows();
    if (mPredicted.cols() == 1)
    {
        //categorical predicted, categorical truth
        assert(mTruth.cols() == 1);
        for (int i = 0; i < iNbRows; i++)
            _iOnlineAccuracyGood += (roundf(mPredicted(i)) == mTruth(i));
    }
    else
    {
        //one hot predicted
        if (mTruth.cols() == 1)
        {
            // categorical truth
            for (int i = 0; i < iNbRows; i++)
                _iOnlineAccuracyGood += (argmax(mPredicted.row(i)) == mTruth(i) );
        }
        else
        {
            // one hot truth
            assert(mTruth.cols() == mPredicted.cols());
            for (int i = 0; i < iNbRows; i++)
                _iOnlineAccuracyGood += (argmax(mPredicted.row(i)) == argmax(mTruth.row(i)));
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_train_loss() const
{
    return _trainLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_test_loss() const
{
	return _fTestLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_test_loss() const
{
    return _testLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_train_accuracy() const
{
    return _trainAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_test_accuracy() const
{
	return _fTestAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_test_accuracy() const
{
    return _testAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_train_loss() const
{
	return _fTrainLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_train_accuracy() const
{
	return _fTrainAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::update_class_weight()
{
	// do not recompute each time

	MatrixFloat mClassWeight;
	if ( (!_pNet->is_classification_mode()) || (!_bClassBalancingWeightLoss))
	{
		mClassWeight.resize(0,0);
	}
	else
	{
		//guess the nb of class and compute occurences
		if (_pmTruthTrain->cols() != 1)
		{
			//convert ot categorical
			int iNbClass = (int)_pmTruthTrain->cols();
			mClassWeight.setZero(iNbClass, 1);
			MatrixFloat mCategory;

			rowsArgmax(*_pmTruthTrain, mCategory);

			for (int i = 0; i < _pmTruthTrain->rows(); i++)
				mClassWeight((int)(mCategory(i)), 0)++;
		}
		else
		{
			int iNbClass = (int)_pmTruthTrain->maxCoeff() + 1;
			mClassWeight.setZero(iNbClass, 1);

			for (int i = 0; i < _pmTruthTrain->rows(); i++)
				mClassWeight((int)(_pmTruthTrain->operator()(i)), 0)++;
		}

		mClassWeight *= mClassWeight.rows() / mClassWeight.sum();

		for (int i = 0; i < mClassWeight.size(); i++)
			mClassWeight(i) = 1.f / mClassWeight(i);
	}

	_pLoss->set_class_balancing(mClassWeight);
}
/////////////////////////////////////////////////////////////////////////////////////////////
