#include "LayerActivation.h"

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string& sActivation):
    Layer(0,0,sActivation)
{
    _pActivation=get_activation(sActivation);
}
///////////////////////////////////////////////////////////////////////////////
LayerActivation::~LayerActivation()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    mOut.resize(mIn.rows(),mIn.cols());

    for(int i=0;i<mOut.size();i++)
        mOut(i)=_pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    (void)fLearningRate;

    mNewDelta.resize(mDelta.rows(),mDelta.cols());
    for(int i=0;i<mNewDelta.size();i++)
        mNewDelta(i)=_pActivation->derivation(mInput(i))*mDelta(i);
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::to_string(string& sBuffer)
{
    sBuffer+="Activation: "+_pActivation->name()+" InSize: "+std::to_string(_iInSize) +" OutSize: "+std::to_string(_iOutSize);
}
///////////////////////////////////////////////////////////////////////////////
