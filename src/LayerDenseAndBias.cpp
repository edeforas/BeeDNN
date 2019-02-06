#include "LayerDenseAndBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
//temp code
MatrixFloat plus_col_one(const MatrixFloat& m)
{
	MatrixFloat mOut;
	mOut.resize(m.rows(),m.cols()+1);
	
	for(int i=0;i<m.rows();i++)
		for(int j=0;j<m.cols();j++)
			mOut(i,j)=m(i,j);
		
	for(int i=0;i<m.rows();i++)
        mOut(i,m.cols())=1;
	
	return mOut;
}
//temp code
MatrixFloat minus_last_row(const MatrixFloat& m)
{
	MatrixFloat mOut;
	mOut.resize(m.rows()-1,m.cols());

	for(int i=0;i<m.rows()-1;i++)
		for(int j=0;j<m.cols();j++)
			mOut(i,j)=m(i,j);
	
	return mOut;
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseAndBias::LayerDenseAndBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize)
{
    _weight.resize(_iInSize+1,_iOutSize);
 //   _bias.resize(1,_iOutSize);
    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseAndBias::~LayerDenseAndBias()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::init()
{
    float a =4.f*sqrtf(6.f/(_iInSize+_iOutSize));

    for(int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
	
  //  for(unsigned int i=0;i<_bias.size();i++)
  //      _bias(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
	MatrixFloat mPlusOne=plus_col_one(mMatIn);
    mMatOut= mPlusOne*_weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=minus_last_row(_weight)*mDelta;

    _weight-=((mDelta*plus_col_one(mInput)).transpose())*fLearningRate;
}
///////////////////////////////////////////////////////////////////////////////
