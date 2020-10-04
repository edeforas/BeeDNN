/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "KMeansTrain.h"

#include "KMeans.h"
#include "Matrix.h"

#include "Loss.h"

#include <cmath>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////////////////////
KMeansTrain::KMeansTrain():
	_epochCallBack(nullptr)
{
	_fTrainLoss=0.f;
	_fTrainAccuracy=0.f;

	_fValidationLoss=0.f;
	_fValidationAccuracy=0.f;

    _pLoss = create_loss("MeanSquaredError"); //todo use
    _iEpochs = 100;
	_iOnlineAccuracyGood= 0;

	_fOnlineLoss = 0.f;
	_pKm = nullptr;

    _pmSamplesTrain = nullptr;
    _pmTruthTrain = nullptr;

	_pmSamplesValidation = nullptr;
	_pmTruthValidation = nullptr;

	_iValidationBatchSize = 1024;
	_iBatchSize = 1024;
	_iBatchSizeAdjusted=0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeansTrain::~KMeansTrain()
{
    delete _pLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::clear()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
/*NetTrain& NetTrain::operator=(const NetTrain& other)
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
*/
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_kmeans(KMeans& km)
{
	_pKm = &km;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans& KMeansTrain::kmeans()
{
	return *_pKm;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_epochs(int iEpochs) //100 by default
{
    _iEpochs = iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int KMeansTrain::get_epochs() const
{
    return _iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_epoch_callback(std::function<void()> epochCallBack)
{
    _epochCallBack = epochCallBack;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_loss(const string&  sLoss)
{
    delete _pLoss;
    _pLoss = create_loss(sLoss);
	assert(_pLoss);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
string KMeansTrain::get_loss() const
{
    return _pLoss->name();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_batchsize(Index iBatchSize) //16 by default
{
	_iBatchSize = iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Index KMeansTrain::get_batchsize() const
{
	return _iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::compute_accuracy(const MatrixFloat &mSamples, const MatrixFloat &mTruth) const
{
    Index iNbSamples = mSamples.rows();
	//float fLoss = 0.f;
	MatrixFloat mOut,mTruthBatch, mSamplesBatch;

	//cut in parts of size _iValidationBatchSize for a lower memory usage
	Index iGood = 0;

	for (Index iStart = 0; iStart < iNbSamples; iStart+=_iValidationBatchSize)
	{
		Index iEnd = iStart + _iValidationBatchSize;
		if (iEnd > iNbSamples)
			iEnd = iNbSamples;
		Index iBatchSize = iEnd - iStart;

		mSamplesBatch = rowView(mSamples, iStart, iEnd);
		mTruthBatch = rowView(mTruth, iStart, iEnd);
		
		_pKm->predict_class(mSamplesBatch, mOut);
			
		for (int i = 0; i < iBatchSize; i++)
			iGood += mOut(i) == mTruthBatch(i);
	}

	return 100.f*iGood / iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth)
{
    _pmSamplesTrain = &mSamples;
    _pmTruthTrain = &mTruth;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation)
{
	_pmSamplesValidation = &mSamplesValidation;
	_pmTruthValidation = &mTruthValidation;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::train()
{
	if (_pKm == nullptr)
		return;
	
	_trainLoss.clear();
	_validationLoss.clear();
	_trainAccuracy.clear();
	_validationAccuracy.clear();
	_fTrainAccuracy = 0;
	_fValidationAccuracy = 0;

	const MatrixFloat& mSamples = *_pmSamplesTrain;
	const MatrixFloat& mTruth = *_pmTruthTrain;
	Index iNbSamples = mSamples.rows();

	MatrixFloat & mRefVectors = _pKm->ref_vectors();
	MatrixFloat & mRefClasses = _pKm->ref_classes();
	MatrixFloat mRefCentroid;
	mRefCentroid.resize(mRefVectors.rows(), mRefVectors.cols());
	
	_mRefCentroidCount.resize(mRefVectors.rows(), 1); 

	Index iNbRef = mRefVectors.rows();

	//init ref vectors, select randomly in full test base
	for (Index i = 0; i < iNbRef; i++)
	{
		Index iPos = randomEngine()() % iNbSamples;

		mRefVectors.row(i) = mSamples.row(iPos);
		mRefClasses.row(i) = mTruth.row(iPos);
	}

	for (int iEpoch = 0; iEpoch < _iEpochs; iEpoch++)
	{
		mRefCentroid.setZero();
		_mRefCentroidCount.setZero();

		for (Index iS = 0; iS < mSamples.rows(); iS++)
		{
			int iClass = (int)mTruth(iS);
			Index iPosBest = -1;
			float fDistBest = 1.e38f;

			for (Index iR = 0; iR < iNbRef; iR++)
			{
				if (mRefClasses(iR) == iClass)
				{
					float fDist = _pKm->compute_dist(mSamples.row(iS), mRefVectors.row(iR));
					if (fDist < fDistBest)
					{
						fDistBest = fDist;
						iPosBest = iR;
					}
				}
			}

			if (iPosBest != -1)
			{
				// update centroid
				mRefCentroid.row(iPosBest) += mSamples.row(iS);
				_mRefCentroidCount(iPosBest)++;
			}
		}

		for (Index iR = 0; iR < iNbRef; iR++)
		{
			if(_mRefCentroidCount(iR)!=0)
				mRefVectors.row(iR) = mRefCentroid.row(iR) / _mRefCentroidCount(iR);
		}

		_fTrainAccuracy = compute_accuracy(mSamples, mTruth);

		if (_pmSamplesValidation != nullptr)
			_fValidationAccuracy = compute_accuracy(*_pmSamplesValidation, *_pmTruthValidation);

		if (_epochCallBack)
			_epochCallBack();
	}

	/*

	_fTrainLoss = 1.e10f;
	_fValidationLoss = 1.e10f;

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
		x
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
*/
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/*void NetTrain::add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth )
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
}*/
/////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::get_current_validation_accuracy() const
{
	return _fValidationAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
/*const vector<float>& NetTrain::get_validation_accuracy() const
{
    return _validationAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::get_current_train_loss() const
{
	return _fTrainLoss;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::get_current_train_accuracy() const
{
	return _fTrainAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
const MatrixFloat& KMeansTrain::ref_count() const
{
	return _mRefCentroidCount;
}
/////////////////////////////////////////////////////////////////////////////////////////////