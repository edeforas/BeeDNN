#include "MLEngineBeeDnn.h"

#include "Net.h"
#include "NetTrain.h"
#include "LayerActivation.h"
#include "LayerDense.h"
#include "NetUtil.h"

//////////////////////////////////////////////////////////////////////////////
MLEngineBeeDnn::MLEngineBeeDnn()
{
    _pNet=new Net;
}
//////////////////////////////////////////////////////////////////////////////
MLEngineBeeDnn::~MLEngineBeeDnn()
{
    _pNet->clear();
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::clear()
{
    _pNet->clear();
}
//////////////////////////////////////////////////////////////////////////////
string MLEngineBeeDnn::to_string()
{
    return NetUtil::to_string(_pNet);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::init()
{
    _pNet->init();
    MLEngine::init();
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::add_dense_layer(int inSize, int outSize, bool bWithBias)
{
    _pNet->add_dense_layer(inSize,outSize,bWithBias);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::add_activation_layer(string sActivation)
{
    _pNet->add_activation_layer(sActivation);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::add_dropout_layer(int inSize,float fRatio)
{
    _pNet->add_dropout_layer(inSize,fRatio);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    _pNet->forward(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
/*int DNNEngineBeeDnn::classify(const MatrixFloat& mIn)
{
    return _pNet->classify(mIn);
}
*/
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    TrainOption tOpt;
    TrainResult tr;
    NetTrain netTrain;
    float fAccuracy=0.f;
    vector<double> vdAccuracy;

    int iEpoch=0;
    tOpt.epochs=dto.epochs;
    tOpt.learningRate=dto.learningRate;
    tOpt.batchSize=dto.batchSize;
    tOpt.optimizer=dto.optimizer;
    tOpt.decay=dto.decay;
    tOpt.momentum=dto.momentum;
    tOpt.testEveryEpochs=dto.testEveryEpochs;
    tOpt.epochCallBack= [&]()
    {
        iEpoch++;
        if(_bClassification)
        {
            if( (iEpoch % tOpt.testEveryEpochs )==0)
            {
                MatrixFloat mConf;
                compute_confusion_matrix(mSamples, mTruth, mConf, fAccuracy);
            }
            vdAccuracy.push_back(fAccuracy);
        }
    };

    if(_bClassification)
    {
        tr=netTrain.train(*_pNet,mSamples,mTruth,tOpt);
        tr.accuracy=vdAccuracy;
    }
    else
        tr=netTrain.fit(*_pNet,mSamples,mTruth,tOpt);

    _vdLoss.insert(end(_vdLoss),begin(tr.loss),end(tr.loss));
    _vdAccuracy.insert(end(_vdAccuracy),begin(tr.accuracy),end(tr.accuracy));
}
//////////////////////////////////////////////////////////////////////////////
