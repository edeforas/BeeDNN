#include "MLEngineBeeDnn.h"

#include <chrono>
#include <string>

#include "ConfusionMatrix.h"
#include "Net.h"
#include "NetTrain.h"
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

	_pTrain->clear(); //todo remove
	_pTrain->set_train_data(mSamples, mTruth);
	_pTrain->set_net(*_pNet);

	_pTrain->train();

	_vfTrainLoss.insert(_vfTrainLoss.end(), _pTrain->get_train_loss().begin(), _pTrain->get_train_loss().end());
	_vfTrainAccuracy.insert(_vfTrainAccuracy.end(), _pTrain->get_train_accuracy().begin(), _pTrain->get_train_accuracy().end());

	if (!_pTrain->get_test_accuracy().empty())
	{
		_vfTestAccuracy.insert(_vfTestAccuracy.end(), _pTrain->get_test_accuracy().begin(), _pTrain->get_test_accuracy().end());
		_vfTestLoss.insert(_vfTestLoss.end(), _pTrain->get_test_loss().begin(), _pTrain->get_test_loss().end());
	}

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
