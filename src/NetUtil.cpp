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
#include "LayerGaussianNoise.h"

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
            ss << "Layer" << i+1 << ".globalGain=" << l->gain() << endl;
        }

        else if(layer->type()=="Dropout")
        {
            LayerDropout* l=static_cast<LayerDropout*>(layer);
            ss << "Layer" << i+1 << ".rate=" << l->get_rate() << endl;
        }

        else if (layer->type() == "GaussianNoise")
        {
            LayerGaussianNoise* l = static_cast<LayerGaussianNoise*>(layer);
            ss << "Layer" << i+1 << ".stdNoise=" << l->get_std() << endl;
        }
    }

    s+=ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void read(const string& s,Net& net)
{
    net.clear();

    (void)s;
    //todo

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

    float fLearningrate;
    float fDecay;
    float fMomentum;
    string sOptimizer;

    train.get_optimizer(sOptimizer,fLearningrate,fDecay,fMomentum);
    ss << "Optimizer=" << sOptimizer << endl;
    ss << "LearningRate=" << fLearningrate << endl;
    ss << "Decay=" << fDecay << endl;
    ss << "Momentum=" << fMomentum << endl;

    s += ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void read(const string& s,NetTrain& train)
{
    train.clear();

    (void)s;
    //todo

}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool save(string sFileName,const Net& net)
{
    string s;
    write(net,s);
    ofstream f(sFileName);
    f << s;
    f.close();

    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////
bool load(string sFileName,Net* pNet)
{
    (void)pNet;

    ifstream f(sFileName);
    /*
    string s=to_string(pNet);
    //TODO
    f << s;
    f.close();
*/
    return true;
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
        i2=s.size()-1;

    return s.substr(i,i2-i);
}
////////////////////////////////////////////////////////////////////////////////////////////////
}
