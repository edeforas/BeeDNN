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
    s+="Problem="+string(_pTrain->is_classification_problem()?"Classification":"Regression")+"\n";  //todo remove?

    NetUtil::write(*_pTrain,s);
    s+="\n";
    NetUtil::write(*_pNet,s);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::read(const string& s)
{  
    NetUtil::read(s,*_pNet);
    NetUtil::read(s,*_pTrain);

	string sProblem = NetUtil::find_key(s, "Problem");
	bool bClassification = (sProblem == "Classification");
	_pTrain->set_problem(bClassification);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::init()
{
    _pNet->init();

    _vfLoss.clear();
    _vfAccuracy.clear();
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

    _pTrain->clear(); //todo remove

    if(_pTrain->is_classification_problem()) //todo call 1 function in_pTrain , learn?
        tr=_pTrain->train(*_pNet,mSamples,mTruth);
    else
        tr=_pTrain->fit(*_pNet,mSamples,mTruth);

    _vfLoss.insert(end(_vfLoss),begin(tr.loss),end(tr.loss));
    _vfAccuracy.insert(end(_vfAccuracy),begin(tr.accuracy),end(tr.accuracy));
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
    r.loss=_vfLoss;
    r.accuracy=_vfAccuracy;

    return r;
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::set_problem(bool bClassification)
{
	_pTrain->set_problem(bClassification);
}
//////////////////////////////////////////////////////////////////////////////
bool MLEngineBeeDnn::is_classification_problem()
{
	return _pTrain->is_classification_problem();
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
    return _pTrain->compute_loss(*_pNet,mSamples,mTruth);
}
//////////////////////////////////////////////////////////////////////////////

