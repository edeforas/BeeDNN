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

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain():
    _sOptimizer("Adam")
{
    _pLoss = create_loss("MeanSquaredError");
    _iBatchSize = 32;
    _bKeepBest = true;
    _iEpochs = 100;
    _iReboostEveryEpochs = -1; // -1 mean no reboost
	_iOnlineAccuracyGood= 0;

    _fLearningRate = -1.f; //default
    _fDecay = -1.f; //default
    _fMomentum = -1.f; //default

	_iNbLayers=0;
	_fOnlineLoss = 0.f;
	_pNet = nullptr;

	_pmSamples = nullptr;
	_pmTruth = nullptr;

	_pmSamplesTest = nullptr;
	_pmTruthTest = nullptr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::~NetTrain()
{
	for (unsigned int i = 0; i < _optimizers.size(); i++)
		delete _optimizers[i];

	_optimizers.clear();

    delete _pLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::clear()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain& NetTrain::operator=(const NetTrain& other)
{
	clear();

	_iOnlineAccuracyGood=other._iOnlineAccuracyGood;
	_fOnlineLoss=other._fOnlineLoss;

	_bKeepBest=other._bKeepBest;
	_iBatchSize=other._iBatchSize;
	_iEpochs=other._iEpochs;
	_iNbLayers=other._iNbLayers;
	_iReboostEveryEpochs=other._iReboostEveryEpochs;

	_sOptimizer=other._sOptimizer;
	_fLearningRate=other._fLearningRate;
	_fDecay=other._fDecay;
	_fMomentum=other._fMomentum;

	for (unsigned int i = 0; i < other._optimizers.size(); i++)
		_optimizers.push_back(create_optimizer(other._optimizers[i]->name())); // initialized

	_inOut = other._inOut;
	_gradient = other._gradient;

	_pmSamples = other._pmSamples;
	_pmTruth = other._pmTruth;

	_epochCallBack = other._epochCallBack;
	_pNet = nullptr;
	_pLoss = create_loss(other._pLoss->name());

	return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_optimizer(const string& sOptimizer) //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov"
{
    _sOptimizer = sOptimizer;
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
float NetTrain::compute_loss(const Net& net, const MatrixFloat &mSamples, const MatrixFloat &mTruth)
{
	if (!net.is_valid((int)mSamples.cols(), (int)mTruth.cols()))
		return 0.f;

    int iNbSamples = (int)mSamples.rows();

    if( (net.layers().size()==0) || (iNbSamples==0) )
        return 0.f;

    MatrixFloat mOut;
    net.predict(mSamples, mOut);

    float fLoss = 0.f;

	if (mTruth.cols() == mOut.cols())
	{
		for (int i = 0; i < iNbSamples; i++) //todo optimize
			fLoss += _pLoss->compute(mOut.row(i), mTruth.row(i));
	}
	else if((mTruth.cols() == 1) && (mOut.cols() >1) ) //categorical
	{
		MatrixFloat mTruthOneHot;
		labelToOneHot(mTruth, mTruthOneHot);
		for (int i = 0; i < iNbSamples; i++) //todo optimize
			fLoss += _pLoss->compute(mOut.row(i), mTruthOneHot.row(i)); //todo optimize
	}

    return fLoss /iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::compute_accuracy(const Net& net, const MatrixFloat &mSamples, const MatrixFloat &mTruth)
{
    int iNbSamples = (int)mSamples.rows();

    if ((net.size() == 0) || (iNbSamples == 0))
        return 0.f;

    MatrixFloat mOut;
    net.predict(mSamples, mOut);

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
	_pmSamples = &mSamples;
	_pmTruth = &mTruth;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_test_data(const MatrixFloat& mSamplesTest, const MatrixFloat& mTruthTest)
{
	_pmSamplesTest = &mSamplesTest;
	_pmTruthTest = &mTruthTest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::train(Net& net)//,const MatrixFloat& mSamples,const MatrixFloat& mTruth)
{
    if(net.layers().size()==0)
        return TrainResult(); //nothing to do

    bool bTruthIsLabel= (_pmTruth->cols()==1);
    if(bTruthIsLabel && (net.output_size()!=1) )
    {
        //create binary label
        MatrixFloat mTruthOneHot;
        labelToOneHot(*_pmTruth, mTruthOneHot);
        set_train_data(*_pmSamples, mTruthOneHot);
		TrainResult tr = fit(net);
		return tr; //todo remove
    }
    else
    {
		TrainResult tr = fit(net);
        return tr; //todo remove
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::fit(Net& net)
{
	const MatrixFloat& mSamples = *_pmSamples;
	const MatrixFloat& mTruth = *_pmTruth;

	if (net.input_size() != (int)mSamples.cols())
		net.set_input_size((int)mSamples.cols());

    TrainResult tr;	
	
	if (!net.is_valid((int)mSamples.cols(),(int) mTruth.cols()))
		return tr;

	_iNbLayers =(int)net.layers().size();
    if(_iNbLayers ==0)
		return tr;

    int iNbSamples=(int)mSamples.rows();
    int iReboost = 0;

	_pNet = &net;

    Net bestNet;

    if(_iBatchSize >iNbSamples)
		_iBatchSize =iNbSamples;

	_inOut.resize(_iNbLayers + 1);

    _gradient.resize(_iNbLayers +1);

	for (unsigned int i = 0; i < _optimizers.size(); i++)
		delete _optimizers[i];
	_optimizers.clear();

    float fMinLoss=1.e10f, fLoss;
    float fAccuracy=0.f , fMaxAccuracy=-1.f;

    //init all optimizers
    for (int i = 0; i < _iNbLayers; i++)
    {
        _optimizers.push_back( create_optimizer(_sOptimizer));
        _optimizers[i]->set_params(_fLearningRate,_fDecay, _fMomentum);
    }

    for(int iEpoch=0;iEpoch<_iEpochs;iEpoch++)
    {
        _fOnlineLoss=0.f;
		_iOnlineAccuracyGood = 0;

        MatrixFloat mSampleShuffled;
        MatrixFloat mTruthShuffled;

		if (_iBatchSize < iNbSamples)
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

        net.set_train_mode(true);

        int iBatchStart=0;

        while(iBatchStart<iNbSamples)
        {
            int iBatchEnd=iBatchStart+_iBatchSize;
            if(iBatchEnd>iNbSamples)
                iBatchEnd=iNbSamples;

            const MatrixFloat mSample = rowRange(mSampleShuffled, iBatchStart, iBatchEnd);
            const MatrixFloat mTarget = rowRange(mTruthShuffled, iBatchStart, iBatchEnd);

			train_batch(mSample, mTarget);
		
            iBatchStart=iBatchEnd;
        }

        net.set_train_mode(false);
        _fOnlineLoss/=iNbSamples;

        tr.trainLoss.push_back(_fOnlineLoss);

        if(net.is_classification_mode())
        {
			fAccuracy=100.f*_iOnlineAccuracyGood/ iNbSamples;
			tr.trainAccuracy.push_back(fAccuracy);
        }

		if (_pmSamplesTest != 0)
		{
			fAccuracy = compute_accuracy(net, *_pmSamplesTest, *_pmTruthTest);
			tr.testAccuracy.push_back(fAccuracy);

			fLoss=compute_loss(net, *_pmSamplesTest, *_pmTruthTest);
			tr.testLoss.push_back(fLoss);
		}

        if (_epochCallBack)
            _epochCallBack();

        //keep the best model if asked
        if(_bKeepBest)
        {
            if(net.is_classification_mode())
            {   //use accuracy
                if(fMaxAccuracy<fAccuracy)
                {
                    fMaxAccuracy=fAccuracy;
                    bestNet=net;
                }
            }
            else
            {   //use loss
                if(fMinLoss>_fOnlineLoss)
                {
                    fMinLoss=_fOnlineLoss;
                    bestNet=net;
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

	if( (_bKeepBest) && (bestNet.size()!=0) )
        net=bestNet;

    return tr;
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
