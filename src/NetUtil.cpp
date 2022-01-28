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
	void save(string sFile,const Net& net, const NetTrain& train)
	{
		// save trained model and train parameters
		JsonFile jf;

		jf.add("Engine", string("BeeDNN"));
		jf.add("Problem", string(net.is_classification_mode() ? "Classification" : "Regression"));

		// write optimizer settings
		jf.enter_section("Optimizer");
		jf.add("Optimizer", train.get_optimizer());
		jf.add("LearningRate", train.get_learningrate());
		jf.add("Epochs", train.get_epochs());
		jf.add("Decay", train.get_decay());
		jf.add("Momentum", train.get_momentum());
		jf.add("Patience", train.get_patience());
		jf.add("BatchSize", (int)train.get_batchsize());
		jf.add("Loss", train.get_loss());
		jf.add("KeepBest", train.get_keepbest());
		jf.add("ReboostEveryEpochs", train.get_reboost_every_epochs());
		jf.add("ClassBalancingWeightLoss", train.get_classbalancing());
		if (!train.get_regularizer().empty())
		{
			jf.add("Regularizer", train.get_regularizer());
			jf.add("RegularizerParameter", train.get_regularizer_parameter());
		}
		jf.leave_section();

		// write layers
		auto layers = net.layers();

		jf.add("NbLayers", (int)layers.size());

		for (size_t i = 0; i < layers.size(); i++)
		{
			Layer* layer = layers[i];

			stringstream ssi; ssi << "Layer" << i;
			string sLayer = ssi.str();

			jf.enter_section(sLayer);
			jf.add("type", layer->type());

			if (layer->has_weights())
			{
				jf.add("weightInitializer", layer->weight_initializer());
				vector<MatrixFloat*> pW = layer->weights();
				for (int j = 0; j < pW.size();j++)
					jf.add_array("weight_"+to_string(j), (int)pW[j]->size(), pW[j]->data());	// todo add i			
			}

			if (layer->has_biases())
			{
				jf.add("biasInitializer", layer->bias_initializer());
				vector<MatrixFloat*> pB = layer->biases();
				for (int j = 0; j < pB.size(); j++)
					jf.add_array("bias_" + to_string(j), (int)pB[j]->size(), pB[j]->data()); // todo add i
			}

			if (layer->type() == "Dense")
			{
				LayerDense* l = static_cast<LayerDense*>(layer);
				jf.add("inputSize", (int)l->input_size());
				jf.add("outputSize", (int)l->output_size());
			}

			if (layer->type() == "Dot")
			{
				LayerDense* l = static_cast<LayerDense*>(layer);
				jf.add("inputSize", (int)l->input_size());
				jf.add("outputSize", (int)l->output_size());
			}

			else if (layer->type() == "ChannelBias")
			{
				LayerChannelBias* l = static_cast<LayerChannelBias*>(layer);

				Index iRows, iCols, iChannels;
				l->get_params(iRows, iCols, iChannels);

				jf.add("rows", (int)iRows);
				jf.add("cols", (int)iCols);
				jf.add("channels", (int)iChannels);
			}

			else if (layer->type() == "Dropout")
			{
				LayerDropout* l = static_cast<LayerDropout*>(layer);
				jf.add("rate", l->get_rate());
			}

			else if (layer->type() == "RRelu")
			{
				LayerRRelu* l = static_cast<LayerRRelu*>(layer);
				float alpha1, alpha2;
				l->get_params(alpha1, alpha2);
				jf.add("alpha1", alpha1);
				jf.add("alpha2", alpha2);
			}

			else if (layer->type() == "GaussianNoise")
			{
				LayerGaussianNoise* l = static_cast<LayerGaussianNoise*>(layer);
				jf.add("noise", l->get_noise());
			}
			else if (layer->type() == "UniformNoise")
			{
				LayerUniformNoise* l = static_cast<LayerUniformNoise*>(layer);
				jf.add("noise", l->get_noise());
			}
			else if (layer->type() == "MaxPool2D")
			{
				LayerMaxPool2D* l = static_cast<LayerMaxPool2D*>(layer);

				Index inRows, inCols, iChannels, rowFactor, colFactor;
				l->get_params(inRows, inCols, iChannels, rowFactor, colFactor);

				jf.add("inRows", (int)inRows);
				jf.add("inCols", (int)inCols);
				jf.add("channels", (int)iChannels);
				jf.add("rowFactor", (int)rowFactor);
				jf.add("colFactor", (int)colFactor);
			}

			else if (layer->type() == "GlobalMaxPool2D")
			{
				LayerGlobalMaxPool2D* l = static_cast<LayerGlobalMaxPool2D*>(layer);

				Index inRows, inCols, iChannels;
				l->get_params(inRows, inCols, iChannels);

				jf.add("inRows", (int)inRows);
				jf.add("inCols", (int)inCols);
				jf.add("channels", (int)iChannels);
			}

			else if (layer->type() == "Convolution2D")
			{
				LayerConvolution2D* l = static_cast<LayerConvolution2D*>(layer);

				Index inRows, inCols, inChannels, kernelRows, kernelCols, outChannels, rowStride, colStride;
				l->get_params(inRows, inCols, inChannels, kernelRows, kernelCols, outChannels, rowStride, colStride);

				jf.add("inRows", (int)inRows);
				jf.add("inCols", (int)inCols);
				jf.add("inChannels", (int)inChannels);
				jf.add("kernelRows", (int)kernelRows);
				jf.add("kernelCols", (int)kernelCols);
				jf.add("rowStride", (int)rowStride);
				jf.add("colStride", (int)colStride);
				jf.add("outChannels", (int)outChannels);
			}

			else if (layer->type() == "TimeDistributedBias")
			{
				LayerTimeDistributedBias* l = static_cast<LayerTimeDistributedBias*>(layer);
				jf.add("frameSize", l->frame_size());
			}

			else if (layer->type() == "TimeDistributedDot")
			{
				LayerTimeDistributedDot* l = static_cast<LayerTimeDistributedDot*>(layer);
				jf.add("inFrameSize", l->in_frame_size());
				jf.add("outFrameSize", l->out_frame_size());
			}

			else if (layer->type() == "TimeDistributedDense")
			{
				LayerTimeDistributedDense* l = static_cast<LayerTimeDistributedDense*>(layer);
				jf.add("inFrameSize", l->in_frame_size());
				jf.add("outFrameSize", l->out_frame_size());
			}

			else if (layer->type() == "SimplestRNN")
			{
				LayerSimplestRNN* l = static_cast<LayerSimplestRNN*>(layer);
				//TODO
			}

			jf.leave_section();
		}

		jf.save(sFile);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	void read(const string& s, Net& net)
	{
		net.clear();

		string sNbLayer=NetUtil::find_key(s,"NbLayers");
		int iNbLayers=stoi(sNbLayer);

		string sProblem = NetUtil::find_key(s, "Problem");
		bool bClassification = (sProblem == "Classification");
		net.set_classification_mode(bClassification);

		for(int i=0;i<iNbLayers;i++)
		{
			string sLayer="Layer"+to_string(i+1);

			string sType=find_key(s,sLayer+".type");

			if(sType=="Dense")
			{
				string sInputSize=find_key(s,sLayer+".inputSize");
				string sOutputSize=find_key(s,sLayer+".outputSize");
				string sWeightInitializer=find_key(s,sLayer+".weightInitializer");
				string sBiasInitializer=find_key(s,sLayer+".biasInitializer");

				Index iInputSize=stoi(sInputSize);
				Index iOutputSize=stoi(sOutputSize);

				net.add(new LayerDense(iInputSize,iOutputSize,sWeightInitializer,sBiasInitializer));

				string sWeight=find_key(s,sLayer+".weight");
				MatrixFloat mf = fromString(sWeight);
				mf.resize(iInputSize, iOutputSize);
				net.layer(net.size()-1).weights()= mf;

				string sBias = find_key(s, sLayer + ".bias");
				mf = fromString(sBias);
				mf.resize(1, iOutputSize);
				net.layer(net.size() - 1).bias() = mf;
			}

			if (sType == "Dot")
			{
				string sInputSize = find_key(s, sLayer + ".inputSize");
				string sOutputSize = find_key(s, sLayer + ".outputSize");
				Index iInputSize = stoi(sInputSize);
				Index iOutputSize = stoi(sOutputSize);

				net.add(new LayerDot(iInputSize, iOutputSize));

				string sWeight = find_key(s, sLayer + ".weight");
				MatrixFloat mf = fromString(sWeight);
				mf.resize(iInputSize, iOutputSize);
				net.layer(net.size() - 1).weights() = mf;
			}

			else if(sType=="GlobalGain")
			{
				float fGain= stof(find_key(s,sLayer+".globalGain"));
				net.add(new LayerGlobalGain());
				MatrixFloat mf(1, 1);
				mf(0) = fGain;
				net.layer(net.size() - 1).weights() = mf;
			}

			else if (sType == "Gain")
			{
				string sWeight = find_key(s, sLayer + ".gain");
				MatrixFloat mf = fromString(sWeight);
				net.add(new LayerGain());
				net.layer(net.size() - 1).weights() = mf;
			}

			else if (sType == "Bias")
			{
				string sBias = find_key(s, sLayer + ".bias");
				MatrixFloat mf = fromString(sBias);
				net.add(new LayerBias());
				net.layer(net.size() - 1).bias() = mf;
			}

			else if (sType == "Affine")
			{
				string sWeight = find_key(s, sLayer + ".gain");
				MatrixFloat mw = fromString(sWeight);
				string sBias = find_key(s, sLayer + ".bias");
				MatrixFloat mb = fromString(sBias);

				net.add(new LayerAffine());
				net.layer(net.size() - 1).bias() = mw;
				net.layer(net.size() - 1).weights() = mb;
			}

			else if(sType=="GlobalBias")
			{
				float fBias= stof(find_key(s,sLayer+".globalBias"));
				net.add(new LayerGlobalBias());
				MatrixFloat mf(1, 1);
				mf(0) = fBias;
				net.layer(net.size() - 1).bias() = mf;
			}

			else if (sType == "GlobalAffine")
			{
				float fGain = stof(find_key(s, sLayer + ".globalGain"));
				float fBias = stof(find_key(s, sLayer + ".globalBias"));
				net.add(new LayerGlobalAffine());

				MatrixFloat mb(1, 1);
				mb(0) = fBias;
				net.layer(net.size() - 1).bias() = mb;

				MatrixFloat mg(1, 1);
				mg(0) = fGain;
				net.layer(net.size() - 1).weights() = mg;
			}

			else if (sType == "ChannelBias")
			{
				string sBias = find_key(s, sLayer + ".bias");
				MatrixFloat mf = fromString(sBias);

				Index iRows = stoi(find_key(s, sLayer + ".rows"));
				Index iCols = stoi(find_key(s, sLayer + ".cols"));
				Index iChannels = stoi(find_key(s, sLayer + ".channels"));

				net.add(new LayerChannelBias(iRows, iCols, iChannels));
				net.layer(net.size() - 1).bias() = mf;
			}

			else if(sType=="Dropout")
			{
				string sRate=find_key(s,sLayer+".rate");
				net.add(new LayerDropout(stof(sRate)));
			}

			else if (sType == "PRelu")
			{
				net.add(new LayerPRelu());
				string sWeight = find_key(s, sLayer + ".weight");
				MatrixFloat mf = fromString(sWeight);
				mf.resize(1, mf.size());
				net.layer(net.size() - 1).weights() = mf;
			}

			else if (sType == "RRelu")
			{
				string sAlpha1 = find_key(s, sLayer + ".alpha1");
				string sAlpha2 = find_key(s, sLayer + ".alpha2");
				net.add(new LayerRRelu(stof(sAlpha1), stof(sAlpha2)));
			}

			else if (sType == "GaussianNoise")
			{
				string sNoise=find_key(s,sLayer+".stdNoise");
				net.add(new LayerGaussianNoise(stof(sNoise)));
			}

			else if (sType == "UniformNoise")
			{
				string sNoise = find_key(s, sLayer + ".noise");
				net.add(new LayerUniformNoise(stof(sNoise)));
			}

			else if (sType == "PoolMax2D")
			{
				Index inSizeX = stoi(find_key(s, sLayer + ".inRows"));
				Index inSizeY = stoi(find_key(s, sLayer + ".inCols"));
				Index inChannels = stoi(find_key(s, sLayer + ".inChannels"));
				Index factorX = stoi(find_key(s, sLayer + ".rowFactor"));
				Index factorY = stoi(find_key(s, sLayer + ".colFactor"));
				net.add(new LayerMaxPool2D(inSizeX, inSizeY, inChannels, factorX, factorY));
			}

			else if (sType == "Convolution2D")
			{
				Index inSizeX = stoi(find_key(s, sLayer + ".inRows"));
				Index inSizeY = stoi(find_key(s, sLayer + ".inCols"));
				Index inChannels = stoi(find_key(s, sLayer + ".inChannels"));
				Index kernelRows = stoi(find_key(s, sLayer + ".kernelRows"));
				Index kernelCols = stoi(find_key(s, sLayer + ".kernelCols"));
				Index rowStride = stoi(find_key(s, sLayer + ".rowStride"));
				Index colStride = stoi(find_key(s, sLayer + ".colStride"));
				Index outChannels = stoi(find_key(s, sLayer + ".outChannels"));
				net.add(new LayerConvolution2D(inSizeX, inSizeY, inChannels, kernelRows, kernelCols, outChannels, rowStride,colStride));

				string sWeight = find_key(s, sLayer + ".weight");
				MatrixFloat mf = fromString(sWeight);
				net.layer(net.size() - 1).weights() = mf;
			}

			else if (sType == "Softmax")
			{
				net.add(new LayerSoftmax());
			}

			else if (sType == "Softmin")
			{
				net.add(new LayerSoftmin());
			}

			else
			{
				//activation layer
				net.add(new LayerActivation(sType));
			}
		}
	}
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/*void write(const NetTrain& train, string& s)
	{
		JsonFile jf;

		stringstream ss;
		ss << "Epochs=" <<train.get_epochs() << endl;
		ss << "BatchSize=" <<train.get_batchsize() << endl;
		ss << "Loss=" << train.get_loss() << endl;
		ss << "KeepBest=" << train.get_keepbest() << endl;
		ss << "ReboostEveryEpochs=" << train.get_reboost_every_epochs() << endl;

		ss << "ClassBalancingWeightLoss=" << train.get_classbalancing() << endl;

		ss << "Optimizer=" << train.get_optimizer() << endl;
		ss << "LearningRate=" << train.get_learningrate() << endl;
		ss << "Decay=" << train.get_decay() << endl;
		ss << "Momentum=" << train.get_momentum() << endl;
		ss << "Patience=" << train.get_patience() << endl;

		if (!train.get_regularizer().empty())
		{
			ss << "Regularizer=" << train.get_regularizer() << endl;
			ss << "RegularizerParameter=" << train.get_regularizer_parameter() << endl;
		}

		s += ss.str();
	}
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	void read(const string& s,NetTrain& train)
	{
		train.clear();

		train.set_epochs(stoi(find_key(s,"Epochs")));
		train.set_batchsize(stoi(find_key(s,"BatchSize")));
		train.set_loss(find_key(s,"Loss"));
		train.set_keepbest(stoi(find_key(s,"KeepBest"))!=0);
		train.set_reboost_every_epochs(stoi(find_key(s,"ReboostEveryEpochs")));

		if (!find_key(s, "ClassBalancingWeightLoss").empty())
			train.set_classbalancing(stoi(find_key(s, "ClassBalancingWeightLoss")) != 0);
		else
			train.set_classbalancing(true);

		string sOptimizer=find_key(s,"Optimizer");
		float fLearningRate=stof(find_key(s,"LearningRate"));
		float fDecay=stof(find_key(s,"Decay"));
		float fMomentum=stof(find_key(s,"Momentum"));

		int iPatience= stoi(find_key(s, "Patience"));

		train.set_optimizer(sOptimizer);
		train.set_learningrate(fLearningRate);
		train.set_decay(fDecay);
		train.set_momentum(fMomentum);
		train.set_patience(iPatience);

		if (!find_key(s, "Regularizer").empty())
		{
			float fParameter = stof(find_key(s, "RegularizerParameter"));
			train.set_regularizer(find_key(s, "Regularizer"),fParameter);
		}
		else
			train.set_regularizer("",0.f);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	string find_key(string s,string sKey)
	{
		auto i = s.find(sKey+":");

		if(i==string::npos)
			return "";

		i+=sKey.size()+1;

		auto i2=s.find("\n",i);

		if(i2==string::npos)
			i2=s.size();

		string s2=s.substr(i,i2-i);

		//trim right
		auto i3=s2.find_last_not_of(" \t\r\n");
		if(i3!=string::npos)
			return s2.substr(0,i3+1);
		else
			return s2;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	void split(string s, vector<string>& vsItems, char cDelimiter)
	{
		vsItems.clear();

		istringstream f(s);
		string sitem;
		while (getline(f, sitem, cDelimiter))
			vsItems.push_back(sitem);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	*/
}