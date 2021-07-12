#include "LayerFactory.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerCRelu.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGaussianDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalBias.h"
#include "LayerGlobalAffine.h"
#include "LayerBias.h"
#include "LayerGain.h"
#include "LayerAffine.h"
#include "LayerUniformNoise.h"
#include "LayerGaussianNoise.h"
#include "LayerPRelu.h"
#include "LayerRRelu.h"
#include "LayerMaxPool2D.h"
#include "LayerGlobalMaxPool2D.h"
#include "LayerAveragePooling2D.h"
#include "LayerGlobalAveragePooling2D.h"
#include "LayerSoftmax.h"
#include "LayerSoftmin.h"
#include "LayerZeroPadding2D.h"
#include "LayerRandomFlipLeftRight2D.h"

//////////////////////////////////////////////////////////////////////////////
Layer* LayerFactory::create(const string& sLayer,float fArg1,float fArg2,float fArg3,float fArg4,float fArg5)
{
	(void)fArg4;
	(void)fArg5;

	if(sLayer=="Dense")
		return new LayerDense((Index)fArg1,(Index)fArg2);

	if (sLayer == "CRelu")
		return new LayerCRelu();

	if (sLayer == "RRelu")
		return new LayerRRelu();

	if (sLayer == "PRelu")
		return new LayerPRelu();

	if (sLayer == "Gain")
		return new LayerGain();

	if (sLayer == "Bias")
		return new LayerBias();

	if (sLayer == "Affine")
		return new LayerAffine();

	if (sLayer == "GlobalGain")
		return new LayerGlobalGain();

	if (sLayer == "GlobalAffine")
		return new LayerGlobalAffine();

	if (sLayer == "GlobalBias")
		return new LayerGlobalBias();

	if (sLayer == "ChannelBias")
		return new LayerChannelBias((Index)fArg1, (Index)fArg2,(Index)fArg3);

	if (sLayer == "MaxPool2D")
		return new LayerMaxPool2D((Index)fArg1, (Index)fArg2, (Index)fArg3);

	if (sLayer == "GlobalMaxPool2D")
		return new LayerGlobalMaxPool2D((Index)fArg1, (Index)fArg2, (Index)fArg3);

	if (sLayer == "AveragePooling2D")
		return new LayerAveragePooling2D((Index)fArg1, (Index)fArg2, (Index)fArg3);

	if (sLayer == "GlobalAveragePooling2D")
		return new LayerGlobalAveragePooling2D((Index)fArg1, (Index)fArg2, (Index)fArg3);

	if (sLayer == "Softmax")
		return new LayerSoftmax();

	if (sLayer == "Softmin")
		return new LayerSoftmin();
	
	if (sLayer == "ZeroPadding2D")
		return new LayerZeroPadding2D((Index)fArg1, (Index)fArg2, (Index)fArg3, (Index)fArg4);

	if (sLayer == "RandFlipLeftRight2D")
		return new LayerRandomFlipLeftRight2D((Index)fArg1, (Index)fArg2, (Index)fArg3);

	return new LayerActivation(sLayer);
}
//////////////////////////////////////////////////////////////////////////////
