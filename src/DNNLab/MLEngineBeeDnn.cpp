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
    _pTrain= new NetTrain;
}
//////////////////////////////////////////////////////////////////////////////
MLEngineBeeDnn::~MLEngineBeeDnn()
{
    delete _pNet;
    delete _pTrain;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::clear()
{
    _pNet->clear();
    _pTrain->clear();
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::write(string& s)
{
    NetUtil::write(*_pTrain,s);
    s+="\n";
    NetUtil::write(*_pNet,s);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::read(const string& s)
{
    (void)s;
    //todo
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
void MLEngineBeeDnn::add_gaussian_noise_layer(int inSize,float fStd)
{
    _pNet->add_gaussian_noise_layer(inSize,fStd);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::add_globalgain_layer(int inSize,float fGain)
{
    _pNet->add_globalgain_layer(inSize,fGain);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::add_poolaveraging1D_layer(int inSize,int iOutSize)
{
    _pNet->add_poolaveraging1D_layer(inSize,iOutSize);
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
    TrainResult tr;
    float fAccuracy=0.f;
    vector<double> vdAccuracy;

    int iEpoch=0;
    _pTrain->clear();
    _pTrain->set_epochs(dto.epochs);
    _pTrain->set_optimizer(dto.optimizer,dto.learningRate,dto.decay,dto.momentum);
    _pTrain->set_batchsize(dto.batchSize);
    _pTrain->set_keepbest(dto.keepBest);
    _pTrain->set_optimizer(dto.optimizer);
    _pTrain->set_loss(dto.lossFunction);
    _pTrain->set_reboost_every_epochs(dto.reboostEveryEpoch);

    _pTrain->set_epoch_callback( [&]()
    {
        iEpoch++;
        if(_bClassification)
        {
            MatrixFloat mConf;
            compute_confusion_matrix(mSamples, mTruth, mConf, fAccuracy);
            vdAccuracy.push_back((double)fAccuracy);
        }
    }
    );

    if(_bClassification)
    {
        tr=_pTrain->train(*_pNet,mSamples,mTruth);
        tr.accuracy=vdAccuracy;
    }
    else
        tr=_pTrain->fit(*_pNet,mSamples,mTruth);

    _vdLoss.insert(end(_vdLoss),begin(tr.loss),end(tr.loss));
    _vdAccuracy.insert(end(_vdAccuracy),begin(tr.accuracy),end(tr.accuracy));
}
//////////////////////////////////////////////////////////////////////////////
