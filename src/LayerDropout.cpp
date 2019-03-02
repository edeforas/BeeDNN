#include "LayerDropout.h"

///////////////////////////////////////////////////////////////////////////////
LayerDropout::LayerDropout(int iSize,float fRate):
    Layer(iSize,iSize,"Dropout"),
    _fRate(fRate)
{
    create_mask(iSize);
}
///////////////////////////////////////////////////////////////////////////////
LayerDropout::~LayerDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    if(_bTrainMode)
        mOut = mIn.cwiseProduct(_mask); //in train mode
    else
        mOut = mIn; // in test mode
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    (void)fLearningRate;
    mNewDelta= mDelta.cwiseProduct(_mask);

    create_mask((int)mInput.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::create_mask(int iSize)
{
    _mask.resize(1, iSize);
    _mask.setConstant(1.f/(1.f - _fRate)); //inverse dropout as in: https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/);

    for (int i = 0; i < iSize; i++)
    {
        if ( (rand()/(double)RAND_MAX) < _fRate)
            _mask(0, i) = 0.f;
    }
}
///////////////////////////////////////////////////////////////////////////////
float LayerDropout::get_rate() const
{
    return _fRate;
}
///////////////////////////////////////////////////////////////////////////////
