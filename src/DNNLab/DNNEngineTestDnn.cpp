#include "DNNEngineTestDnn.h"

#include "Net.h"
#include "NetTrainSGD.h"
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
    _pNet->clear();
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
    _pNet->init();
    DNNEngine::init();
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTestDnn::add_layer(int inSize, int outSize, string sLayerType)
{
    _pNet->add_layer(sLayerType,inSize,outSize);
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
    tOpt.observer=nullptr;//dto.observer;

    NetTrainSGD netTrain;
    netTrain.fit(*_pNet,mSamples,mTruth,tOpt);

    const auto& l=netTrain.loss();
    _vdLoss.insert(end(_vdLoss),begin(l),end(l)); //temp
}
//////////////////////////////////////////////////////////////////////////////
double DNNEngineTestDnn::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
    NetTrainSGD netTrain;
    return netTrain.compute_loss(*_pNet,mSamples,mTruth);
}
//////////////////////////////////////////////////////////////////////////////
