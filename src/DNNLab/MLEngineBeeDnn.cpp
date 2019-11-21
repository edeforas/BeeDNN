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

    _pNet=new Net;
    _pTrain= new NetTrain;
	_pTrain->set_net(*_pNet);
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
    NetUtil::read(s,*_pNet);
    NetUtil::read(s,*_pTrain);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::init()
{
    _pNet->init();

    _vfTrainLoss.clear();
    _vfTestLoss.clear();

    _vfTrainAccuracy.clear();
    _vfTestAccuracy.clear();
    
    _iComputedEpochs=0;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    if(!_pNet->is_valid((int)mIn.cols(), _pNet->output_size()))
        return;

    _pNet->forward(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth)
{
    _pTrain->clear(); //todo remove
    _pTrain->set_train_data(mSamples, mTruth);
    _pTrain->set_net(*_pNet);

    _pTrain->train();

    _vfTrainLoss.insert(_vfTrainLoss.end(),_pTrain->get_train_loss().begin(),_pTrain->get_train_loss().end());
    _vfTrainAccuracy.insert(_vfTrainAccuracy.end(),_pTrain->get_train_accuracy().begin(),_pTrain->get_train_accuracy().end());

    if (!_pTrain->get_test_accuracy().empty())
    {
        _vfTestAccuracy.insert(_vfTestAccuracy.end(), _pTrain->get_test_accuracy().begin(), _pTrain->get_test_accuracy().end());
        _vfTestLoss.insert(_vfTestLoss.end(), _pTrain->get_test_loss().begin(), _pTrain->get_test_loss().end());
    }
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
    r.trainLoss=_vfTrainLoss;
    r.testLoss = _vfTestLoss;

    r.trainAccuracy=_vfTrainAccuracy;
    r.testAccuracy=_vfTestAccuracy;

    return r;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::set_classification_mode(bool bClassification)
{
    _pNet->set_classification_mode(bClassification);
}
//////////////////////////////////////////////////////////////////////////////
bool MLEngineBeeDnn::is_classification_mode()
{
    return _pNet->is_classification_mode();
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
    _pNet->forward(mSamples,mResult);
    /*
    for(int i=0;i<mSamples.rows();i++)
    {
        predict(mSamples.row(i),temp);
        if(i==0)
            mResult.resize(mSamples.rows(),temp.cols());
        mResult.row(i)=temp;
    }*/
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel)
{
    MatrixFloat temp;
    mResultLabel.resize(mSamples.rows(),1);
    for(int i=0;i<mSamples.rows();i++)
    {
        _pNet->classify(mSamples.row(i),temp);
        mResultLabel(i,0)=temp(0,0); //case of "output is a label"
    }
}
//////////////////////////////////////////////////////////////////////////////
float MLEngineBeeDnn::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
    return _pTrain->compute_loss(mSamples,mTruth);
}
//////////////////////////////////////////////////////////////////////////////

