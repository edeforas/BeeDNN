#include "Net.h"
#include "Layer.h"

#include "Matrix.h"
#include "MatrixUtil.h"

#include "LayerActivation.h"
#include "LayerDenseNoBias.h"
#include "LayerDenseAndBias.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::~Net()
{
    clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::clear()
{
    for(unsigned int i=0;i<_layers.size();i++)
    {
        delete _layers[i];
    }

    _layers.clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_layer(string sType,int inSize,int outSize)
{
    if(sType=="DenseAndBias")
        _layers.push_back(new LayerDenseAndBias(inSize,outSize));

    else if(sType=="DenseNoBias")
        _layers.push_back(new LayerDenseNoBias(inSize,outSize));

    else
         _layers.push_back(new LayerActivation(sType));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    MatrixFloat mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward(mTemp,mOut);
        mTemp=mOut; //todo avoid resize
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::classify(const MatrixFloat& mIn,MatrixFloat& mClass) const
{
    mClass.resize(mIn.rows(),1);

    MatrixFloat mOut;
    for(int i=0;i<mIn.rows();i++)
    {
        forward(mIn.row(i),mOut);
        mClass(i)=argmax(mOut)(0);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
const vector<Layer*> Net::layers() const
{
    return _layers;
}
/////////////////////////////////////////////////////////////////////////////////////////////
Layer* Net::layer(size_t iLayer)
{
    return _layers[iLayer];
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::init()
{
	for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->init();
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
