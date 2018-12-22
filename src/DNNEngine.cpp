#include "DNNEngine.h"

#include <chrono>

//////////////////////////////////////////////////////////////////////////////
DNNEngine::DNNEngine()
{
    _iComputedEpochs=0;
}
//////////////////////////////////////////////////////////////////////////////
DNNEngine::~DNNEngine()
{ }
//////////////////////////////////////////////////////////////////////////////
DNNTrainResult DNNEngine::train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    DNNTrainResult r;

    if(!dto.bTrainMore)
        _iComputedEpochs=0;

    auto beginDuration = std::chrono::steady_clock::now();
    int iEpochs=train_epochs(mSamples,mTruth,dto);
    auto endDuration = std::chrono::steady_clock::now();

    _iComputedEpochs+= iEpochs;
    r.epochDuration=chrono::duration_cast<chrono::microseconds> (endDuration-beginDuration).count()/1.e6/iEpochs;
    r.computedEpochs=_iComputedEpochs;
    return r;
}
//////////////////////////////////////////////////////////////////////////////
