#include "DNNEngine.h"

#include <chrono>
#include <string>

//////////////////////////////////////////////////////////////////////////////
DNNEngine::DNNEngine()
{
    _iComputedEpochs=0;
}
//////////////////////////////////////////////////////////////////////////////
DNNEngine::~DNNEngine()
{ }
//////////////////////////////////////////////////////////////////////////////
void DNNEngine::init()
{
    _vdLoss.clear();
    _iComputedEpochs=0;
}
//////////////////////////////////////////////////////////////////////////////
DNNTrainResult DNNEngine::train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    DNNTrainResult r;

    auto beginDuration = std::chrono::steady_clock::now();
    train_epochs(mSamples,mTruth,dto);
    auto endDuration = std::chrono::steady_clock::now();

    _iComputedEpochs+= dto.epochs;
    r.epochDuration=chrono::duration_cast<chrono::microseconds> (endDuration-beginDuration).count()/1.e6/dto.epochs;
    r.computedEpochs=_iComputedEpochs;

    r.loss=_vdLoss;
    return r;
}
//////////////////////////////////////////////////////////////////////////////
