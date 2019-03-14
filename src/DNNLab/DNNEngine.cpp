#include "DNNEngine.h"

#include "ConfusionMatrix.h"

#include <chrono>
#include <string>

//////////////////////////////////////////////////////////////////////////////
DNNEngine::DNNEngine()
{
    _iComputedEpochs=0;
    _bClassification=true;
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
DNNTrainResult DNNEngine::learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    DNNTrainResult r;

    auto beginDuration = std::chrono::steady_clock::now();
    learn_epochs(mSamples,mTruth,dto);
    auto endDuration = std::chrono::steady_clock::now();

    _iComputedEpochs+= dto.epochs;
    r.epochDuration=chrono::duration_cast<chrono::microseconds> (endDuration-beginDuration).count()/1.e6/dto.epochs;
    r.computedEpochs=_iComputedEpochs;

    r.loss=_vdLoss;
    return r;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngine::set_problem(bool bClassification)
{
    _bClassification=bClassification;
}
//////////////////////////////////////////////////////////////////////////////
bool DNNEngine::problem()
{
    return _bClassification;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngine::compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth,MatrixFloat& mConfusionMatrix, float& fAccuracy)
{
    MatrixFloat mTest;
    classify_all(mSamples,mTest);
    ConfusionMatrix cm;
    ClassificationResult result=cm.compute(mTruth,mTest);

    mConfusionMatrix=result.mConfMat;
    fAccuracy=(float)result.accuracy;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngine::predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult)
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
void DNNEngine::classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel)
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
