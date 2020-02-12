#include "LayerFactory.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGaussianDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalBias.h"
#include "LayerBias.h"
#include "LayerGain.h"
#include "LayerUniformNoise.h"
#include "LayerGaussianNoise.h"
#include "LayerPRelu.h"
#include "LayerRRelu.h"
#include "LayerPoolMax2D.h"
#include "LayerSoftmax.h"

//////////////////////////////////////////////////////////////////////////////
Layer* LayerFactory::create(string sLayer,float fArg1,float fArg2,float fArg3,float fArg4,float fArg5)
{
    (void)fArg1;
    (void)fArg2;
    (void)fArg3;
    (void)fArg4;
    (void)fArg5;

	if(sLayer=="Dense")
		return new LayerDense((Index)fArg1,(Index)fArg2);

	if (sLayer == "RRelu")
		return new LayerRRelu();

	if (sLayer == "PRelu")
		return new LayerPRelu();

	if (sLayer == "GlobalBias")
		return new LayerGlobalBias();

	if (sLayer == "Gain")
		return new LayerGain();

	if (sLayer == "Bias")
		return new LayerBias();

	if (sLayer == "GlobalGain")
		return new LayerGlobalGain();

	return new LayerActivation(sLayer);
}
//////////////////////////////////////////////////////////////////////////////
