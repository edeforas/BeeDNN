#include "NetTrainSGD.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "Optimizer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainSGD::NetTrainSGD(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainSGD::~NetTrainSGD()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainSGD::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt)
{
    //create a kronecker matrix, one line by sample
    int iMax=(int)mTruthLabel.maxCoeff();
    MatrixFloat mTruth(mTruthLabel.rows(),iMax+1);
    mTruth.setZero();

    for(int i=0;i<mTruth.rows();i++)
        mTruth(i,(int)mTruthLabel(i,0))=1;

    fit(net,mSamples,mTruth,topt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainSGD::fit(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    int iNbSamples=(int)mSamples.rows();
    int nLayers=(int)net.layers().size();
    vector<MatrixFloat> inOut(nLayers+1);
    vector<MatrixFloat> delta(nLayers+1);
	vector<Optimizer*> optimizers(nLayers);
	
	double dLoss=0.;

    _vdLoss.clear();

    if(nLayers==0)
        return;

	for (int i = 0; i < nLayers; i++)
	{
		optimizers[i] = get_optimizer(topt.sOptimizer);
		optimizers[i]->fLearningRate = topt.learningRate;
        optimizers[i]->fMomentum=topt.momentum;
		optimizers[i]->init(*(net.layer(i)));
	}

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        MatrixFloat mShuffle=rand_perm(iNbSamples);
		net.set_train_mode(true);

        for(int iSample=0;iSample<iNbSamples;iSample++)
        {
            int iIndexSample=(int)mShuffle(iSample,0);
            const MatrixFloat& mSample=mSamples.row(iIndexSample);
            const MatrixFloat& mTarget=mTruth.row(iIndexSample);

            //forward pass with store and add
            inOut[0]=mSample;
            for(int i=0;i<nLayers;i++)
                net.layer(i)->forward(inOut[i],inOut[i+1]);

            //backward pass
            delta[nLayers]=inOut[nLayers]-mTarget;
            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer* l=net.layer(i);
                l->backpropagation(inOut[i],delta[i+1], optimizers[i],delta[i]);
            }
        }

		net.set_train_mode(false);

		if (topt.epochCallBack)
			topt.epochCallBack();

        if(topt.testEveryEpochs!=-1)
            if( (iEpoch% topt.testEveryEpochs) == 0)
                dLoss=compute_loss(net,mSamples,mTruth);
        _vdLoss.push_back(dLoss);
    }

	for (int i = 0; i < nLayers; i++)
		delete optimizers[i];
}
/////////////////////////////////////////////////////////////////////////////////////////////
