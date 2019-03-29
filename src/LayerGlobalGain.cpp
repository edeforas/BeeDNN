#include "LayerGlobalGain.h"

#include <cmath> // for sqrt
#include "Optimizer.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain(int iInSize, float fGlobalGain) :
    Layer(iInSize , iInSize, "GlobalGain"),
    _fGlobalGain(fGlobalGain)
{
    _bLearnable=(_fGlobalGain==0.f); //temp code

    _globalGain.resize(1,1);

    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalGain::clone() const
{
    LayerGlobalGain* pLayer=new LayerGlobalGain(_iInSize,_fGlobalGain);
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn * _fGlobalGain;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta)
{
    mNewDelta = mDelta * _fGlobalGain;
    _mDx = mInput*(mDelta.transpose());

    if(_bLearnable)
    {
        _globalGain(0,0)=_fGlobalGain;
        pOptim->optimize(_globalGain, _mDx);
        _fGlobalGain=_globalGain(0,0);
    }
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalGain::gain() const
{
    return _fGlobalGain;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalGain::is_learned() const
{
    return _bLearnable;
}
///////////////////////////////////////////////////////////////////////////////
