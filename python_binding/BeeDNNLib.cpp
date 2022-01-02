#include "BeeDNNLib.h"

#include "Net.h"
#include "NetTrain.h"
#include "LayerFactory.h"
#include "NetUtil.h"
////////////////////////////////////////////////////////////////////////////
class BeeDNN
{
public:
	BeeDNN(int iInputSize)
	{
		pNet = new Net();
		pTrain = new NetTrain();
		_iInputSize = iInputSize;
	}

	~BeeDNN()
	{
		delete pNet;
		delete pTrain;
	}

	Net* pNet;
	NetTrain* pTrain;

	int _iInputSize;

};
////////////////////////////////////////////////////////////////////////////
void* create(int32_t iInputSize)
{
	return new BeeDNN(iInputSize);
}
////////////////////////////////////////////////////////////////////////////
void set_classification_mode(void* pNN, int32_t _iClassificationMode)
{
	((BeeDNN*)pNN)->pNet->set_classification_mode(_iClassificationMode != 0);
}
////////////////////////////////////////////////////////////////////////////
void add_layer(void* pNN, char *layer)
{
	Layer* pLayer = LayerFactory::create(layer);
	((BeeDNN*)pNN)->pNet->add(pLayer);
}
////////////////////////////////////////////////////////////////////////////
void predict(void* pNN,const float *pIn, float *pOut,int32_t iNbSamples)
{ 
	MatrixFloat mIn = fromRawBuffer(pIn, iNbSamples, ((BeeDNN*)pNN)->_iInputSize);
	MatrixFloat mOut(iNbSamples, 1); //TODO only one output, todo remove temp
	((BeeDNN*)pNN)->pNet->predict(mIn, mOut);

	for(Index i=0;i< iNbSamples;i++)
		pOut[i] =mOut(i);
}
////////////////////////////////////////////////////////////////////////////
void save(void* pNN,char *filename)
{
	NetUtil::save(filename,*((BeeDNN*)pNN)->pNet,*((BeeDNN*)pNN)->pTrain);
}
////////////////////////////////////////////////////////////////////////////