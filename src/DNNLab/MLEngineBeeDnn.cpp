#include "MLEngineBeeDnn.h"

#include <chrono>
#include <string>

#include "ConfusionMatrix.h"
#include "Net.h"
#include "NetTrain.h"
#include "LayerActivation.h"
#include "LayerDense.h"
#include "NetUtil.h"

//////////////////////////////////////////////////////////////////////////////
MLEngineBeeDnn::MLEngineBeeDnn()
{
    _iComputedEpochs=0;
    _bClassification=true;

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
    s+="Problem="+string(_bClassification?"Classification":"Regression")+"\n";

    NetUtil::write(*_pTrain,s);
    s+="\n";
    NetUtil::write(*_pNet,s);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::read(const string& s)
{
    (void)s;

    string sProblem=NetUtil::find_key(s,"Problem");

    _bClassification=(sProblem=="Classification");

    NetUtil::read(s,*_pNet);
    NetUtil::read(s,*_pTrain);

    //todo
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::init()
{
    _pNet->init();

    _vdLoss.clear();
    _vdAccuracy.clear();
    _iComputedEpochs=0;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    _pNet->forward(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth)
{
    TrainResult tr;
    float fAccuracy=0.f;
    vector<double> vdAccuracy;

    int iEpoch=0;
    _pTrain->clear();

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
Net& MLEngineBeeDnn::net()
{
    return *_pNet;
}
//////////////////////////////////////////////////////////////////////////////
const Net& MLEngineBeeDnn::net() const
{
    return *_pNet;
}

//////////////////////////////////////////////////////////////////////////////
NetTrain& MLEngineBeeDnn::netTrain()
{
    return *_pTrain;
}
//////////////////////////////////////////////////////////////////////////////
const NetTrain& MLEngineBeeDnn::netTrain() const
{
    return *_pTrain;
}
//////////////////////////////////////////////////////////////////////////////
DNNTrainResult MLEngineBeeDnn::learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth)
{
    DNNTrainResult r;

    auto beginDuration = std::chrono::steady_clock::now();
    learn_epochs(mSamples,mTruth);
    auto endDuration = std::chrono::steady_clock::now();

    _iComputedEpochs+= _pTrain->get_epochs();
    r.epochDuration=chrono::duration_cast<chrono::microseconds> (endDuration-beginDuration).count()/1.e6/_pTrain->get_epochs();
    r.computedEpochs=_iComputedEpochs;

    r.loss=_vdLoss;

    if(_bClassification)
        r.accuracy=_vdAccuracy;

    return r;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::set_problem(bool bClassification)
{
    _bClassification=bClassification;
}
//////////////////////////////////////////////////////////////////////////////
bool MLEngineBeeDnn::is_classification_problem()
{
    return _bClassification;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth,MatrixFloat& mConfusionMatrix, float& fAccuracy)
{
    MatrixFloat mTest;
    classify_all(mSamples,mTest);
    ConfusionMatrix cm;
    ClassificationResult result=cm.compute(mTruth,mTest);

    mConfusionMatrix=result.mConfMat;
    fAccuracy=(float)result.accuracy;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult)
{
    MatrixFloat temp;
    for(int i=0;i<mSamples.rows();i++)
    {
        predict(mSamples.row(i),temp);
        if(i==0)
            mResult.resize(mSamples.rows(),temp.cols());
        mResult.row(i)=temp;
    }
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel)
{
    MatrixFloat temp;
    mResultLabel.resize(mSamples.rows(),1);
    for(int i=0;i<mSamples.rows();i++)
    {
        predict(mSamples.row(i),temp);
        if(temp.cols()!=1)
            mResultLabel(i,0)=(float)argmax(temp);
        else
            mResultLabel(i,0)=temp(0,0); //case of "output is a label"
    }
}
//////////////////////////////////////////////////////////////////////////////
float MLEngineBeeDnn::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
    MatrixFloat mPredicted;
    int iNbSamples=(int)mSamples.rows();

    predict_all(mSamples,mPredicted);
    return (mPredicted-mTruth).cwiseAbs2().sum()/iNbSamples;
}
//////////////////////////////////////////////////////////////////////////////

