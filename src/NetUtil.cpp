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

#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGain.h"
#include "LayerGlobalBias.h"
#include "LayerBias.h"
#include "LayerSoftmax.h"
#include "LayerUniformNoise.h"
#include "LayerGaussianNoise.h"
#include "LayerPRelu.h"
#include "LayerPoolMax2D.h"

#include <sstream>
#include <fstream>
using namespace std;

namespace NetUtil {

/////////////////////////////////////////////////////////////////////////////////////////////////
void write(const Net& net,string & s)
{
    stringstream ss;

    auto layers=net.layers();
    ss << "Engine=BeeDNN" << endl;
    ss << "NbLayers=" << layers.size() << endl;

	string sProblem= net.is_classification_mode()?"Classification":"Regression";
	ss << "Problem=" << sProblem << endl;

    for(size_t i=0;i<layers.size();i++)
    {
        Layer* layer=layers[i];

        ss << endl;
        ss << "Layer" << i+1 << ".type=" << layer->type() << endl;
        if(layer->in_size())
            ss << "Layer" << i+1 << ".inSize=" << layer->in_size() << endl;
        if(layer->out_size())
            ss << "Layer" << i+1 << ".outSize=" << layer->out_size() << endl;

        if(layer->type()=="Dense")
        {
            LayerDense* l=static_cast<LayerDense*>(layer);
            ss << "Layer" << i+1 << ".hasBias=" << (l->has_bias()?1:0) << endl;
            ss << "Layer" << i+1 << ".weight=" << endl;
            ss << toString(layer->weights()) << endl;
        }

        else if(layer->type()=="GlobalGain")
        {
            LayerGlobalGain* l=static_cast<LayerGlobalGain*>(layer);
            ss << "Layer" << i+1 << ".globalGain=" << l->weights()(0) << endl;
        }

		else if (layer->type() == "Gain")
		{
			LayerGain* l = static_cast<LayerGain*>(layer);
			ss << "Layer" << i + 1 << ".gain=" << toString(l->weights()) << endl;
		}
		
		else if(layer->type()=="GlobalBias")
        {
            LayerGlobalBias* l=static_cast<LayerGlobalBias*>(layer);
            ss << "Layer" << i+1 << ".globalBias=" << l->weights()(0) << endl;
        }

		else if (layer->type() == "Bias")
		{
			LayerBias* l = static_cast<LayerBias*>(layer);
			ss << "Layer" << i + 1 << ".bias=" << toString(l->weights()) << endl;
		}

        else if(layer->type()=="Dropout")
        {
            LayerDropout* l=static_cast<LayerDropout*>(layer);
            ss << "Layer" << i+1 << ".rate=" << l->get_rate() << endl;
        }

		else if (layer->type() == "PRelu")
		{
			ss << "Layer" << i + 1 << ".weight=" << endl;
			ss << toString(layer->weights()) << endl;
		}

        else if (layer->type() == "GaussianNoise")
        {
            LayerGaussianNoise* l = static_cast<LayerGaussianNoise*>(layer);
            ss << "Layer" << i+1 << ".stdNoise=" << l->get_std() << endl;
        }
		else if (layer->type() == "UniformNoise")
		{
			LayerUniformNoise* l = static_cast<LayerUniformNoise*>(layer);
			ss << "Layer" << i + 1 << ".noise=" << l->get_noise() << endl;
		}
		else if (layer->type() == "PoolMax2D")
		{
			LayerPoolMax2D* l = static_cast<LayerPoolMax2D*>(layer);

			int inRows, inCols, iPlanes, rowFactor, colFactor;
			l->get_params(inRows, inCols, iPlanes, rowFactor, colFactor);
			ss << "Layer" << i + 1 << ".inRows=" << inRows << endl;
			ss << "Layer" << i + 1 << ".inCols=" << inCols << endl;
			ss << "Layer" << i + 1 << ".inPlanes=" << iPlanes << endl;
			ss << "Layer" << i + 1 << ".rowFactor=" << rowFactor << endl;
			ss << "Layer" << i + 1 << ".colFactor=" << colFactor << endl;
		}
	}

    s+=ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void read(const string& s,Net& net)
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

        string sInSize=find_key(s,sLayer+".inSize");
        string sOutSize=find_key(s,sLayer+".outSize");

        int iInSize=0;
        int iOutSize=0;

        if(!sInSize.empty())
            iInSize=stoi(sInSize);

        if(!sOutSize.empty())
            iOutSize=stoi(sOutSize);

        if(sType=="Dense")
        {
            string sHasBias=find_key(s,sLayer+".hasBias");
            bool bHasBias=sHasBias!="0";
            net.add_dense_layer(iInSize,iOutSize,bHasBias);

            string sWeight=find_key(s,sLayer+".weight");
            MatrixFloat mf=fromString(sWeight);
            mf.resize(iInSize+(bHasBias?1:0),iOutSize);
            net.layer(net.size()-1).weights()=mf;
        }

        else if(sType=="GlobalGain")
        {
            float fGain= stof(find_key(s,sLayer+".globalGain"));
            net.add_globalgain_layer();
			MatrixFloat mf(1, 1);
			mf(0) = fGain;
			net.layer(net.size() - 1).weights() = mf;
        }

		else if (sType == "Gain")
		{
			string sWeight = find_key(s, sLayer + ".gain");
			MatrixFloat mf = fromString(sWeight);
			net.add_gain_layer();
			net.layer(net.size() - 1).weights() = mf;
		}
		
		else if(sType=="GlobalBias")
        {
            float fBias= stof(find_key(s,sLayer+".globalBias"));
            net.add_globalbias_layer();
            MatrixFloat mf(1, 1);
            mf(0) = fBias;
            net.layer(net.size() - 1).weights() = mf;
        }

		else if (sType == "Bias")
		{
			string sWeight = find_key(s, sLayer + ".bias");
			MatrixFloat mf = fromString(sWeight);
			net.add_bias_layer();
			net.layer(net.size() - 1).weights() = mf;
		}

        else if(sType=="Dropout")
        {
            string sRate=find_key(s,sLayer+".rate");
            net.add_dropout_layer(stof(sRate));
        }

		else if (sType == "PRelu")
		{
			net.add_prelu_layer();
			string sWeight = find_key(s, sLayer + ".weight");
			MatrixFloat mf = fromString(sWeight);
			mf.resize(1,iInSize);
			net.layer(net.size() - 1).weights() = mf;
		}

        else if (sType == "GaussianNoise")
        {
            string sNoise=find_key(s,sLayer+".stdNoise");
            net.add_gaussian_noise_layer(stof(sNoise));
        }

		else if (sType == "UniformNoise")
		{
			string sNoise = find_key(s, sLayer + ".noise");
			net.add_uniform_noise_layer(stof(sNoise));
		}

		else if (sType == "PoolMax2D")
		{
			int inSizeX = stoi(find_key(s, sLayer + ".inRows"));
			int inSizeY = stoi(find_key(s, sLayer + ".inCols"));
			int inPlanes = stoi(find_key(s, sLayer + ".inPlanes"));
			int factorX = stoi(find_key(s, sLayer + ".rowFactor"));
			int factorY = stoi(find_key(s, sLayer + ".colFactor"));
			net.add_poolmax2D_layer(inSizeX, inSizeY, inPlanes, factorX, factorY);
		}

		else if (sType == "Softmax")
		{
			net.add_softmax_layer();
		}

        else
        {
            //activation layer
            net.add_activation_layer(sType);
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void write(const NetTrain& train,string & s)
{
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

    s += ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
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

    train.set_optimizer(sOptimizer);
    train.set_learningrate(fLearningRate);
    train.set_decay(fDecay);
    train.set_momentum(fMomentum);
}
////////////////////////////////////////////////////////////////////////////////////////////////
string find_key(string s,string sKey)
{
    auto i = s.find(sKey+"=");

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
}
