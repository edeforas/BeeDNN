#include "DNNEngineTestDnn.h"

#include "Net.h"
#include "NetTrainLearningRate.h"
#include "LayerActivation.h"
#include "LayerDenseNoBias.h"
#include "LayerDenseAndBias.h"
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
void DNNEngineTestDnn::add_layer(int inSize, int outSize, string sLayerType)
{
    if(sLayerType=="DenseAndBias")
        _pNet->add(new LayerDenseAndBias(inSize,outSize));
    else if(sLayerType=="DenseNoBias")
        _pNet->add(new LayerDenseNoBias(inSize,outSize));
    else
         _pNet->add(new LayerActivation(sLayerType));
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    _pNet->forward(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    TrainOption tOpt;
    tOpt.epochs=dto.epochs;
    tOpt.learningRate=dto.learningRate;
    tOpt.batchSize=dto.batchSize;
    //tOpt.momentum=dto.momentum;
    tOpt.observer=0;//dto.observer;

    NetTrainLearningRate netTrain;
    netTrain.train(*_pNet,mSamples,mTruth,tOpt);
}
//////////////////////////////////////////////////////////////////////////////
