#include "Net.h"
#include "Layer.h"

#include "Matrix.h"

#include "LayerActivation.h"
#include "LayerDenseNoBias.h"
#include "LayerDenseAndBias.h"
#include "LayerDropout.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ 
	_bTrainMode = false;
}
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
void Net::add_dropout_layer(int iSize,float fRatio)
{
     _layers.push_back(new LayerDropout(iSize, fRatio));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_activation_layer(string sType)
{
	_layers.push_back(new LayerActivation(sType));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_dense_layer(string sType,int inSize,int outSize)
{
    if(sType=="DenseAndBias")
        _layers.push_back(new LayerDenseAndBias(inSize,outSize));

    else if(sType=="DenseNoBias")
        _layers.push_back(new LayerDenseNoBias(inSize,outSize));

  //  else
   //     _layers.push_back(new LayerActivation(sType));
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
int Net::classify(const MatrixFloat& mIn) const
{
    MatrixFloat mOut;
    forward(mIn,mOut);
    return argmax(mOut);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::classify_all(const MatrixFloat& mIn, MatrixFloat& mClass) const
{
    MatrixFloat mOut;

    mClass.resize(mIn.rows(),1);
    for(int i=0;i<mIn.rows();i++)
    {
        forward(mIn.row(i),mOut);
        mClass(i,0)= (float)argmax(mOut);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_train_mode(bool bTrainMode)
{
	_bTrainMode = bTrainMode;

	for (unsigned int i = 0; i < _layers.size(); i++)
	{
		_layers[i]->set_train_mode(bTrainMode);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////
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
