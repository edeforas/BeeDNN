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
#include "Regularizer.h"
#include "Loss.h"

#include <cmath>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain():
    _sOptimizer("Adam"),
	_epochCallBack(nullptr)
{
	_fTrainLoss=0.f;
	_fTrainAccuracy=0.f;

	_fValidationLoss=0.f;
	_fValidationAccuracy=0.f;

	_pRegularizer = nullptr;

    _pLoss = create_loss("MeanSquaredError");
    _iBatchSize = 32;
	_iBatchSizeAdjusted=-1; //invalid
	_iValidationBatchSize = 128;
	_bKeepBest = true;
    _iEpochs = 100;
    _iReboostEveryEpochs = -1; // -1 mean no reboost
	_iOnlineAccuracyGood= 0;

    _fLearningRate = -1.f; //default
    _fDecay = -1.f; //default
    _fMomentum = -1.f; //default
	_iPatience = -1; //-1 mean no limit 
	_iCurrentPatience = 0; // nb of epochs without progress

	_bClassBalancingWeightLoss = false;

	_iNbLayers=0;
	_fOnlineLoss = 0.f;
	_pNet = nullptr;

    _pmSamplesTrain = nullptr;
    _pmTruthTrain = nullptr;

	_pmSamplesValidation = nullptr;
	_pmTruthValidation = nullptr;
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

	_iBatchSizeAdjusted=-1; //invalid
	
    set_keepbest(other._bKeepBest);
	set_classbalancing(other._bClassBalancingWeightLoss);
    set_batchsize(other._iBatchSize);
	set_validation_batchsize(_iValidationBatchSize);
	set_epochs(other._iEpochs);
	set_reboost_every_epochs(other._iReboostEveryEpochs);
	set_loss(other._pLoss->name());
	set_regularizer(other.get_regularizer(),other.get_regularizer_parameter());
	
	_iOnlineAccuracyGood = other._iOnlineAccuracyGood;
	_fOnlineLoss = other._fOnlineLoss;

	_fTrainLoss = other._fTrainLoss;
	_fTrainAccuracy = other._fTrainAccuracy;
	_fValidationLoss = other._fValidationLoss;
	_fValidationAccuracy = other._fValidationAccuracy;
	_iNbLayers=other._iNbLayers;

	clear_optimizers();
	_sOptimizer = other._sOptimizer;
	for (size_t i = 0; i < other._optimizers.size(); i++)
		_optimizers.push_back(other._optimizers[i]->clone());

    _fLearningRate=other._fLearningRate;
    _fDecay=other._fDecay;
    _fMomentum=other._fMomentum;
	_iPatience = other._iPatience;
	_iCurrentPatience = other._iCurrentPatience;

    _inOut = other._inOut;
    _gradient = other._gradient;

	_trainLoss = other._trainLoss;
	_trainAccuracy = other._trainAccuracy;

	_validationLoss = other._validationLoss;
	_validationAccuracy = other._validationAccuracy;

	_epochCallBack = other._epochCallBack;

    _pmSamplesTrain = other._pmSamplesTrain;
    _pmTruthTrain = other._pmTruthTrain;

	_pmSamplesValidation = other._pmSamplesValidation;
	_pmTruthValidation = other._pmTruthValidation;

	set_net(*(other._pNet));

	return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_net(Net& net)
{
	_pNet = &net;
	_iNbLayers = (int)_pNet->layers().size();
	if (_iNbLayers != 0)
		_pNet->layers()[0]->set_first_layer(true);
	
//	clear_optimizers(); todo keep ?
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
void NetTrain::set_regularizer(const string& sRegularizer,float fParameter) // "" by default, can be also "Identity", "Clamp" ...
{
	delete _pRegularizer;
	if (sRegularizer != "")
	{
		_pRegularizer = create_regularizer(sRegularizer);
		_pRegularizer->set_parameter(fParameter);
	}
	else
		_pRegularizer = nullptr;
}
string NetTrain::get_regularizer() const
{
	if (_pRegularizer != nullptr)
		return _pRegularizer->name();
	else
		return "";
}
float NetTrain::get_regularizer_parameter() const
{
	if (_pRegularizer != nullptr)
		return _pRegularizer->get_parameter();
	return -1.f;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_learningrate(float fLearningRate) //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov"
{
    _fLearningRate = fLearningRate;

	for (size_t i = 0; i < _optimizers.size(); i++)
		_optimizers[i]->set_learningrate(_fLearningRate);
}
float NetTrain::get_learningrate() const
{
	if (!_optimizers.empty())
		return _optimizers[0]->get_learningrate(); // get real lr if initialized with -1.f ;

    return _fLearningRate;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_patience(int iPatience)
{
	_iPatience = iPatience;
	_iCurrentPatience = 0;
}
int NetTrain::get_patience() const
{
	return _iPatience;
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
void NetTrain::set_batchsize(Index iBatchSize) //16 by default
{
    _iBatchSize = iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Index NetTrain::get_batchsize() const
{
    return _iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_validation_batchsize(Index iValBatchSize) //128 by default
{
	_iValidationBatchSize = iValBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Index NetTrain::get_validation_batchsize() const
{
	return _iValidationBatchSize;
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
float NetTrain::compute_loss_accuracy(const MatrixFloat &mSamples, const MatrixFloat &mTruth,float * pfAccuracy) const
{
    Index iNbSamples = mSamples.rows();
	float fLoss = 0.f;
	MatrixFloat mOut,mTruthBatch, mSamplesBatch,mLoss;

	if ((_pNet->layers().size() == 0) || (iNbSamples == 0))
	{
		if (pfAccuracy)
			*pfAccuracy = 0.f;
		return 0.f;
	}

	//cut in parts of size _iValidationBatchSize for a lower memory usage
	Index iGood = 0;

	for (Index iStart = 0; iStart < iNbSamples; iStart+=_iValidationBatchSize)
	{
		Index iEnd = iStart + _iValidationBatchSize;
		if (iEnd > iNbSamples)
			iEnd = iNbSamples;
		Index iBatchSize = iEnd - iStart;

		mSamplesBatch = viewRow(mSamples, iStart, iEnd);
		mTruthBatch = viewRow(mTruth, iStart, iEnd);
		
		_pNet->predict(mSamplesBatch, mOut);
		_pLoss->compute(mOut, mTruthBatch,mLoss);
		fLoss += mLoss.mean();

		if (pfAccuracy)
		{
			if (mTruthBatch.cols() != 1)
			{
				assert(mOut.cols() == mTruthBatch.cols());
				//one hot everywhere
				for (Index i = 0; i < iBatchSize; i++)
					iGood += (argmax(mOut.row(i)) == argmax(mTruthBatch.row(i)));
			}
			else
			{
				if (mOut.cols() != 1)
				{
					for (Index i = 0; i < iBatchSize; i++)
						iGood += (argmax(mOut.row(i)) == mTruthBatch(i));
				}
				else
				{
					for (int i = 0; i < iBatchSize; i++)
						iGood += roundf(mOut(i)) == mTruthBatch(i);
				}
			}

		}
	}

	if(pfAccuracy)
		*pfAccuracy = 100.f*iGood / iNbSamples;

	return fLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth)
{
    _pmSamplesTrain = &mSamples;
    _pmTruthTrain = &mTruth;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation)
{
	_pmSamplesValidation = &mSamplesValidation;
	_pmTruthValidation = &mTruthValidation;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::fit()
{
	if (_pNet == nullptr)
		return;

	_iNbLayers = _pNet->layers().size();
	if (_iNbLayers == 0)
		return; //nothing to do

	update_class_weight();

    const MatrixFloat& mSamples = *_pmSamplesTrain;
    const MatrixFloat& mTruth = *_pmTruthTrain;

    _trainLoss.clear();
    _validationLoss.clear();
    _trainAccuracy.clear();
    _validationAccuracy.clear();
	_fTrainLoss = 1.e10f;
	_fTrainAccuracy = 0;
	_fValidationLoss = 1.e10f;
	_fValidationAccuracy = 0;
	_iCurrentPatience = 0;

    int iNbSamples=(int)mSamples.rows();
    int iReboost = 0;

    Net bestNet;

    //accept batch size == 0 or greater than nb samples  -> full size
    _iBatchSizeAdjusted=_iBatchSize;
    if( (_iBatchSizeAdjusted >iNbSamples) || (_iBatchSizeAdjusted ==0) )
        _iBatchSizeAdjusted =iNbSamples;

    _inOut.resize(_iNbLayers + 1);

    _gradient.resize(_iNbLayers +1);

	if (_optimizers.empty())
	{
		//init all optimizers
		for (size_t i = 0; i < _iNbLayers*2; i++)
		{
			// one optimizer for weight, one for bias
			_optimizers.push_back(create_optimizer(_sOptimizer));
			_optimizers[i]->set_params(_fLearningRate, _fDecay, _fMomentum);
		}
	}

    //compute the accuracy at epoch 0, if keepbest is selected
	float fMaxAccuracy = 0.f;
	float fMinLoss = 1.e10f;
	if(_bKeepBest)
    {
        if (_pmSamplesValidation == nullptr)
        {
            fMinLoss=compute_loss_accuracy(*_pmSamplesTrain, *_pmTruthTrain,&fMaxAccuracy);
        }
        else
        {
            fMinLoss=compute_loss_accuracy( *_pmSamplesValidation, *_pmTruthValidation,&fMaxAccuracy);
        }
        bestNet= *_pNet;
    }

    for(int iEpoch=0;iEpoch<_iEpochs;iEpoch++)
    {
        _fOnlineLoss=0.f;
        _iOnlineAccuracyGood = 0;

        MatrixFloat mSampleShuffled;
        MatrixFloat mTruthShuffled;

        if (_iBatchSizeAdjusted < iNbSamples)
        {
            auto vShuffle = randPerm(iNbSamples);
            applyRowPermutation(vShuffle, mSamples, mSampleShuffled);
            applyRowPermutation(vShuffle, mTruth, mTruthShuffled);
        }
        else
        {
			// no need to shuffle
            mSampleShuffled = mSamples; //todo remove copy
            mTruthShuffled = mTruth;
        }

		_pNet->set_train_mode(true);

		train_one_epoch(mSampleShuffled, mTruthShuffled);

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
        if (_pmSamplesValidation != nullptr)
        { 	
			//use the test_db to keep the best model
			_fValidationLoss =compute_loss_accuracy( *_pmSamplesValidation, *_pmTruthValidation,&_fValidationAccuracy);
            _validationLoss.push_back(_fValidationLoss);
            _validationAccuracy.push_back(_fValidationAccuracy);

			fSelectedLoss = _fValidationLoss;
			fSelectedAccuracy = _fValidationAccuracy;
		}

        if (_epochCallBack)
            _epochCallBack();

        //keep the best model if asked
        if(_bKeepBest || (_iPatience!=-1) )
        {
            if(_pNet->is_classification_mode())
            {   //use accuracy
				if (fMaxAccuracy < fSelectedAccuracy)
				{
					fMaxAccuracy = fSelectedAccuracy;
					bestNet = *_pNet;
					_iCurrentPatience = 0;
				}
				else
					_iCurrentPatience++;
            }
            else
            {   //use loss
                if(fMinLoss> fSelectedLoss)
                {
                    fMinLoss= fSelectedLoss;
                    bestNet= *_pNet;
					_iCurrentPatience = 0;
				}
				else
					_iCurrentPatience++;
			}
        
			if ( (_iCurrentPatience > _iPatience) && (_iPatience!=-1) )
			{
				_iCurrentPatience = 0;
				set_learningrate(get_learningrate() / 2.f);
				(*_pNet).operator=(bestNet);
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
                for (size_t i = 0; i < _optimizers.size(); i++)
                    _optimizers[i]->init();
            }
        }
    }

	if(_bKeepBest)
		(*_pNet).operator=(bestNet);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth)
{
	assert(_pNet);
	assert(_optimizers.size() == _iNbLayers * 2);

	//forward pass with store
	_inOut[0] = mSample;
	for (size_t i = 0; i < _iNbLayers; i++)
		_pNet->layer(i).forward(_inOut[i], _inOut[i + 1]);

	//compute error gradient
	_pLoss->compute_gradient(_inOut[_iNbLayers], mTruth, _gradient[_iNbLayers]);

	//backward pass with optimizer
	for (int i = (int)_iNbLayers - 1; i >= 0; i--)
	{
		Layer& l = _pNet->layer(i);
		l.backpropagation(_inOut[i], _gradient[i + 1], _gradient[i]);

		if (l.has_weight())
		{
			if (_pRegularizer)
				_pRegularizer->apply(l.weights(),l.gradient_weights());
			
			_optimizers[2*i]->optimize(l.weights(), l.gradient_weights());
		}

		if (l.has_bias())
		{
			//bias does not need regularization 
			_optimizers[2 * i+1]->optimize(l.bias(), l.gradient_bias());
		}
	}

	//compute and save statistics
	add_online_statistics(_inOut[_iNbLayers], mTruth);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth )
{
    //update loss
	MatrixFloat mLoss;
    _pLoss->compute(mPredicted, mTruth,mLoss);
	_fOnlineLoss += mLoss.mean();

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
float NetTrain::get_current_validation_loss() const
{
	return _fValidationLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_validation_loss() const
{
    return _validationLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_train_accuracy() const
{
    return _trainAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_validation_accuracy() const
{
	return _fValidationAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<float>& NetTrain::get_validation_accuracy() const
{
    return _validationAccuracy;
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
void NetTrain::train_one_epoch(const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled)
{
	Index iNbSamples = mSampleShuffled.rows();
	Index iBatchStart = 0;

	while (iBatchStart < iNbSamples)
	{
		Index iBatchEnd = iBatchStart + _iBatchSizeAdjusted;
		if (iBatchEnd > iNbSamples)
			iBatchEnd = iNbSamples;

		auto mSample = viewRow(mSampleShuffled, iBatchStart, iBatchEnd);
		auto mTarget = viewRow(mTruthShuffled, iBatchStart, iBatchEnd);
		train_batch(mSample, mTarget);

		iBatchStart = iBatchEnd;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
