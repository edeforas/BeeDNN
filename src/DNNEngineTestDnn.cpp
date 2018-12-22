#include "DNNEngineTestDnn.h"

#include "Net.h"
#include "ActivationLayer.h"

//////////////////////////////////////////////////////////////////////////////
DNNEngineTestDnn::DNNEngineTestDnn()
{
	_pNet=new Net;
}
//////////////////////////////////////////////////////////////////////////////
DNNEngineTestDnn::~DNNEngineTestDnn()
{
    clear();
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::clear()
{
    _pNet->clear();
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::add_layer_and_activation(int inSize,int outSize, eLayerType layer, string sActivation)
{
    (void)layer;
    _pNet->add(new ActivationLayer(inSize,outSize,sActivation));
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
   _pNet->forward(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
DNNTrainResult DNNEngineTestDnn::train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    TrainOption tOpt;
    tOpt.epochs=dto.epochs;
    tOpt.earlyAbortMaxError=dto.earlyAbortMaxError;
    tOpt.earlyAbortMeanError=dto.earlyAbortMeanError;
    tOpt.learningRate=dto.learningRate;
    tOpt.batchSize=dto.batchSize;
    tOpt.momentum=dto.momentum;
    tOpt.bTrainMore=dto.bTrainMore;
    tOpt.observer=0;//dto.observer;

    TrainResult tr=_pNet->train(mSamples,mTruth,tOpt);

    DNNTrainResult dtr;

    dtr.loss=tr.loss;
    dtr.maxError=tr.maxError;
    dtr.computedEpochs=tr.computedEpochs;
    dtr.epochDuration=tr.epochDuration; //in second

    return dtr;
}
//////////////////////////////////////////////////////////////////////////////
