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

    _pNet->predict(mIn,mOut);
}
//////////////////////////////////////////////////////////////////////////////
void MLEngineBeeDnn::learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth)
{
    TrainResult tr;

    _pTrain->clear(); //todo remove

	_pTrain->set_train_data(mSamples, mTruth);

    if(_pNet->is_classification_mode()) //todo call 1 function in_pTrain , learn?
        tr=_pTrain->train(*_pNet);
    else
        tr=_pTrain->fit(*_pNet);

    _vfTrainLoss.insert(end(_vfTrainLoss),begin(tr.trainLoss),end(tr.trainLoss));
    _vfTrainAccuracy.insert(end(_vfTrainAccuracy),begin(tr.trainAccuracy),end(tr.trainAccuracy));

	if (!tr.testAccuracy.empty())
	{
		_vfTestAccuracy.insert(end(_vfTestAccuracy), begin(tr.testAccuracy), end(tr.testAccuracy));
		_vfTestLoss.insert(end(_vfTestLoss), begin(tr.testLoss), end(tr.testLoss));
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

