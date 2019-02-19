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
double DNNEngine::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
  /*  if(net.layers().size()==0)
         return -1.;
 */
     int iNbSamples=mSamples.rows();
     MatrixFloat mOut,mError;
    double dError=0.;

     for(int i=0;i<iNbSamples;i++)
     {
         predict(mSamples.row(i),mOut);

         if(i==0)
             dError=(mOut-mTruth.row(i)).cwiseAbs2().sum();
         else
             dError+=(mOut-mTruth.row(i)).cwiseAbs2().sum();
     }

     return dError/(double)iNbSamples;
}
//////////////////////////////////////////////////////////////////////////////
