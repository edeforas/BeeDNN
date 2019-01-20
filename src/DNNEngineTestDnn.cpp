#include "DNNEngineTestDnn.h"

#include "Net.h"
#include "NetTrainMomentum.h"
#include "ActivationLayer.h"
#include "NetUtil.h"

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
string DNNEngineTestDnn::to_string()
{
    return NetUtil::to_string(_pNet);
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::init()
{
   // _pNet->init(); todo
    DNNEngine::init();
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
int DNNEngineTestDnn::train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    TrainOption tOpt;
    tOpt.epochs=dto.epochs;
    tOpt.earlyAbortMaxError=dto.earlyAbortMaxError;
    tOpt.earlyAbortMeanError=dto.earlyAbortMeanError;
    tOpt.learningRate=dto.learningRate;
    tOpt.batchSize=dto.batchSize;
    tOpt.momentum=dto.momentum;
    tOpt.observer=0;//dto.observer;

    NetTrainMomentum netTrain;
    int epochs=netTrain.train(*_pNet,mSamples,mTruth,tOpt);

    return epochs;
}
//////////////////////////////////////////////////////////////////////////////
