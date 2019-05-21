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

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain():
	_sOptimizer("Adam")
{
	_pLoss = create_loss("MeanSquaredError");
	_iBatchSize = 16;
	_bKeepBest = true;
	_iEpochs = 100;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::~NetTrain()
{
	delete _pLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::set_optimizer(string sOptimizer)
{
	_sOptimizer = sOptimizer;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
string NetTrain::get_optimizer() const
{
	return _sOptimizer;
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
void NetTrain::set_loss(string sLoss)
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
	int iNbSamples = (int)mSamples.rows();
	
	if( (net.layers().size()==0) || (iNbSamples==0) )
        return 0.f;

    MatrixFloat mOut;
	net.forward(mSamples, mOut);

	float fLoss = 0.f;
    for(int i=0;i<iNbSamples;i++) //todo optimize
		fLoss += _pLoss->compute(mOut.row(i), mTruth.row(i));

    return fLoss /iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    if(net.layers().size()==0)
        return TrainResult(); //nothing to do

    bool bTruthIsLabel= (mTruth.cols()==1);
    if(bTruthIsLabel)
    {
		//create binary label
        MatrixFloat mTruthOneHot;
		labelToOneHot(mTruth, mTruthOneHot);

        return fit(net,mSamples, mTruthOneHot,topt);
    }
    else
        return fit(net,mSamples,mTruth,topt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::fit(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    TrainResult tr;
    int iNbSamples=(int)mSamples.rows();
    int nLayers=(int)net.layers().size();

    Net bestNet;

    int iBatchSize=_iBatchSize;
    if(iBatchSize>iNbSamples)
        iBatchSize=iNbSamples;

    float fInvBatchSize=1.f/(float)iBatchSize;

    vector<MatrixFloat> inOut(nLayers+1);
    vector<MatrixFloat> deltaWeight(nLayers+1);
    vector<MatrixFloat> delta(nLayers+1);

    vector<Optimizer*> optimizers(nLayers);

    MatrixFloat mLoss;
    double dLoss=0.,dMinLoss=1.e99;

    tr.reset();

    if(nLayers==0)
        return tr;

    //init all optimizers
    for (int i = 0; i < nLayers; i++)
    {
        optimizers[i] = create_optimizer(_sOptimizer);
        optimizers[i]->fLearningRate = topt.learningRate;
        optimizers[i]->fMomentum=topt.momentum;
        optimizers[i]->fDecay=topt.decay;
        optimizers[i]->init(net.layer(i));
    }

    for(int iEpoch=0;iEpoch<_iEpochs;iEpoch++)
    {
        dLoss=0.;

        MatrixFloat mShuffle=randPerm(iNbSamples);
		MatrixFloat mSampleShuffled;
		MatrixFloat mTruthShuffled;
		applyRowPermutation(mShuffle, mSamples, mSampleShuffled);
		applyRowPermutation(mShuffle, mTruth, mTruthShuffled);

        net.set_train_mode(true);

        int iBatchStart=0;

        while(iBatchStart<iNbSamples)
        {
            int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iNbSamples)
                iBatchEnd=iNbSamples;

            for(int i=0;i<nLayers+1;i++)
                deltaWeight[i].setZero();
			
			const MatrixFloat mSample = rowRange(mSampleShuffled, iBatchStart, iBatchEnd);
			const MatrixFloat mTarget = rowRange(mTruthShuffled, iBatchStart, iBatchEnd);

            //forward pass with store
            inOut[0]=mSample;
            for(int i=0;i<nLayers;i++)
                 net.layer(i).forward(inOut[i],inOut[i+1]);

            //compute loss
			_pLoss->compute_gradient(inOut[nLayers], mTarget, delta[nLayers]);
			dLoss += _pLoss->compute(inOut[nLayers], mTarget);
				
			//backward pass with store , compute deltaWeight, optimize
			for (int i = nLayers - 1; i >= 0; i--)
			{
				Layer& l = net.layer(i);
				l.backpropagation(inOut[i], delta[i + 1], delta[i]);

				if (l.has_weight())
				{
					optimizers[i]->optimize(l.weights(), l.gradient_weights()* fInvBatchSize);
				}
			}

            iBatchStart=iBatchEnd;
        }

        net.set_train_mode(false);
        dLoss/=iNbSamples;

        tr.loss.push_back(dLoss);

        if (topt.epochCallBack)
            topt.epochCallBack();

        //keep the best model if asked
        if(_bKeepBest)
        {
            if(dMinLoss>dLoss)
            {
                dMinLoss=dLoss;
                bestNet=net;
            }
        }
    }

    for (int i = 0; i < nLayers; i++)
        delete optimizers[i];

    if(_bKeepBest)
    {
        net=bestNet;
        tr.finalLoss=dMinLoss;
    }
    else
        tr.finalLoss=dLoss;

    return tr;
}
/////////////////////////////////////////////////////////////////////////////////////////////
