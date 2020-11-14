#include "BeeDNNLib.h"

#include "Activations.h"
#include "Net.h"
#include "NetTrain.h"
#include "LayerFactory.h"

class BeeDNN
{
public:
	BeeDNN()
	{
		pNet = new Net();
		pTrain = new NetTrain();
	}

	~BeeDNN()
	{
		delete pNet;
		delete pTrain;
	}

	Net* pNet;
	NetTrain* pTrain;

};

void* create()
{
	return new BeeDNN();
}

void set_classification_mode(void* pNN, int32_t _iClassificationMode)
{
	((BeeDNN*)pNN)->pNet->set_classification_mode(_iClassificationMode != 0);
}

void add_layer(void* pNN, char *layer)
{
	Layer* pLayer = LayerFactory::create(layer);
	((BeeDNN*)pNN)->pNet->add(pLayer);
}

void predict(void* pNN,const float *pIn, float *pOut)
{
	MatrixFloat mIn = fromRawBuffer(pIn, 1, 1);
	MatrixFloat mOut = fromRawBuffer(pOut, 1, 1);
	((BeeDNN*)pNN)->pNet->predict(mIn, mOut);

	pOut[0] = mOut(0);
}