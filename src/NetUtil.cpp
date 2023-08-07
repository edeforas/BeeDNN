/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetUtil.h"
#include "Net.h"
#include "NetTrain.h"
#include "Layer.h"
#include "Matrix.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerDense.h"
#include "LayerDot.h"
#include "LayerDropout.h"
#include "LayerGain.h"
#include "LayerBias.h"
#include "LayerAffine.h"
#include "LayerGaussianNoise.h"
#include "LayerGlobalBias.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalAffine.h"
#include "LayerMaxPool2D.h"
#include "LayerGlobalMaxPool2D.h"
#include "LayerCRelu.h"
#include "LayerPRelu.h"
#include "LayerRRelu.h"
#include "LayerSoftmax.h"
#include "LayerSoftmin.h"
#include "LayerUniformNoise.h"
#include "LayerTimeDistributedBias.h"
#include "LayerTimeDistributedDot.h"
#include "LayerTimeDistributedDense.h"
#include "LayerSimplestRNN.h"

#include "JsonFile.h"

#include <string>
#include <sstream>
using namespace std;

namespace NetUtil {
	/////////////////////////////////////////////////////////////////////////////////////////////////
	void save(const string& sFile,const Net& model, const NetTrain& trainParams)
	{
		// save trained model and train parameters
		JsonFileWriter jf;

		jf.add("Engine", string("BeeDNN"));
		jf.add("Problem", string(model.is_classification_mode() ? "Classification" : "Regression"));

		// write optimizer settings
		jf.enter_section("Optimizer");
		jf.add("Optimizer", trainParams.get_optimizer());
		jf.add("LearningRate", trainParams.get_learningrate());
		jf.add("Epochs", trainParams.get_epochs());
		jf.add("Decay", trainParams.get_decay());
		jf.add("Momentum", trainParams.get_momentum());
		jf.add("Patience", trainParams.get_patience());
		jf.add("BatchSize", (int)trainParams.get_batchsize());
		jf.add("Loss", trainParams.get_loss());
		jf.add("KeepBest", trainParams.get_keepbest());
		jf.add("ReboostEveryEpochs", trainParams.get_reboost_every_epochs());
		jf.add("ClassBalancingWeightLoss", trainParams.get_classbalancing());
		if (!trainParams.get_regularizer().empty())
		{
			jf.add("Regularizer", trainParams.get_regularizer());
			jf.add("RegularizerParameter", trainParams.get_regularizer_parameter());
		}
		jf.leave_section();

		// write layers
		auto layers = model.layers();

		jf.add("NbLayers", (int)layers.size());

		for (size_t i = 0; i < layers.size(); i++)
		{
			Layer* layer = layers[i];

			stringstream ssi; ssi << "Layer_" << i;
			string sLayer = ssi.str();

			jf.enter_section(sLayer);
			jf.add("Type", layer->type());

			if (layer->has_weights())
			{
				jf.add("WeightInitializer", layer->weight_initializer());
				vector<MatrixFloat*> pW = layer->weights();
				for (size_t j = 0; j < pW.size(); j++)
					jf.add_array("Weight_" + to_string(j), (int)pW[j]->size(), pW[j]->data());
			}

			if (layer->has_biases())
			{
				jf.add("BiasInitializer", layer->bias_initializer());
				vector<MatrixFloat*> pB = layer->biases();
				for (size_t j = 0; j < pB.size(); j++)
					jf.add_array("Bias_" + to_string(j), (int)pB[j]->size(), pB[j]->data());
			}

			if (layer->type() == "Dense")
			{
				auto l = static_cast<const LayerDense*>(layer);
				jf.add("InputSize", (int)l->input_size());
				jf.add("OutputSize", (int)l->output_size());
			}

			if (layer->type() == "Dot")
			{
				auto l = static_cast<const LayerDense*>(layer);
				jf.add("InputSize", (int)l->input_size());
				jf.add("OutputSize", (int)l->output_size());
			}

			else if (layer->type() == "ChannelBias")
			{
				auto l = static_cast<const LayerChannelBias*>(layer);

				Index iRows, iCols, iChannels;
				l->get_params(iRows, iCols, iChannels);

				jf.add("Rows", (int)iRows);
				jf.add("Cols", (int)iCols);
				jf.add("Channels", (int)iChannels);
			}

			else if (layer->type() == "Dropout")
			{
				auto l = static_cast<const LayerDropout*>(layer);
				jf.add("Rate", l->get_rate());
			}

			else if (layer->type() == "RRelu")
			{
				auto l = static_cast<const LayerRRelu*>(layer);
				float alpha1, alpha2;
				l->get_params(alpha1, alpha2);
				jf.add("Alpha1", alpha1);
				jf.add("Alpha2", alpha2);
			}

			else if (layer->type() == "GaussianNoise")
			{
				auto l = static_cast<const LayerGaussianNoise*>(layer);
				jf.add("Noise", l->get_noise());
			}
			else if (layer->type() == "UniformNoise")
			{
				auto l = static_cast<const LayerUniformNoise*>(layer);
				jf.add("Noise", l->get_noise());
			}
			else if (layer->type() == "MaxPool2D")
			{
				auto l = static_cast<const LayerMaxPool2D*>(layer);

				Index inRows, inCols, iChannels, rowFactor, colFactor;
				l->get_params(inRows, inCols, iChannels, rowFactor, colFactor);

				jf.add("InRows", (int)inRows);
				jf.add("InCols", (int)inCols);
				jf.add("Channels", (int)iChannels);
				jf.add("RowFactor", (int)rowFactor);
				jf.add("ColFactor", (int)colFactor);
			}

			else if (layer->type() == "GlobalMaxPool2D")
			{
				auto l = static_cast<const LayerGlobalMaxPool2D*>(layer);

				Index inRows, inCols, iChannels;
				l->get_params(inRows, inCols, iChannels);

				jf.add("InRows", (int)inRows);
				jf.add("InCols", (int)inCols);
				jf.add("Channels", (int)iChannels);
			}

			else if (layer->type() == "Convolution2D")
			{
				auto l = static_cast<const LayerConvolution2D*>(layer);

				Index inRows, inCols, inChannels, kernelRows, kernelCols, outChannels, rowStride, colStride;
				l->get_params(inRows, inCols, inChannels, kernelRows, kernelCols, outChannels, rowStride, colStride);

				jf.add("InRows", (int)inRows);
				jf.add("InCols", (int)inCols);
				jf.add("InChannels", (int)inChannels);
				jf.add("KernelRows", (int)kernelRows);
				jf.add("KernelCols", (int)kernelCols);
				jf.add("RowStride", (int)rowStride);
				jf.add("ColStride", (int)colStride);
				jf.add("OutChannels", (int)outChannels);
			}

			else if (layer->type() == "TimeDistributedBias")
			{
				auto l = static_cast<const LayerTimeDistributedBias*>(layer);
				jf.add("FrameSize", l->frame_size());
			}

			else if (layer->type() == "TimeDistributedDot")
			{
				auto l = static_cast<const LayerTimeDistributedDot*>(layer);
				jf.add("InFrameSize", l->in_frame_size());
				jf.add("OutFrameSize", l->out_frame_size());
			}

			else if (layer->type() == "TimeDistributedDense")
			{
				auto l = static_cast<const LayerTimeDistributedDense*>(layer);
				jf.add("InFrameSize", l->in_frame_size());
				jf.add("OutFrameSize", l->out_frame_size());
			}

			else if (layer->type() == "SimplestRNN")
			{
				//LayerSimplestRNN* l = static_cast<LayerSimplestRNN*>(layer);
				//TODO
			}

			jf.leave_section();
		}

		jf.save(sFile);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////
}