#include "ActivationManager.h"

#include "ActivationAtan.h"
#include "ActivationElliot.h"
#include "ActivationGauss.h"
#include "ActivationLinear.h"
#include "ActivationRelu.h"
#include "ActivationLeakyRelu.h"
#include "ActivationElu.h"
#include "ActivationSelu.h"
#include "ActivationSoftPlus.h"
#include "ActivationSoftSign.h"
#include "ActivationSigmoid.h"
#include "ActivationTanh.h"

ActivationManager::ActivationManager()
{
    _vActivations.push_back(new ActivationAtan);
    _vActivations.push_back(new ActivationElliot);
    _vActivations.push_back(new ActivationGauss);
    _vActivations.push_back(new ActivationLinear);
    _vActivations.push_back(new ActivationRelu);
    _vActivations.push_back(new ActivationLeakyRelu);
    _vActivations.push_back(new ActivationElu);
    _vActivations.push_back(new ActivationSelu);
    _vActivations.push_back(new ActivationSoftPlus);
    _vActivations.push_back(new ActivationSoftSign);
    _vActivations.push_back(new ActivationSigmoid);
    _vActivations.push_back(new ActivationTanh);
}

ActivationManager::~ActivationManager()
{
    for(unsigned int i=0; i<_vActivations.size();i++)
        delete _vActivations[i];
}

Activation* ActivationManager::get_activation(string sName) //do not delete: manager own it.
{
    for(unsigned int i=0; i<_vActivations.size();i++)
    {
        if(_vActivations[i]->name()==sName)
            return _vActivations[i];
    }

    return 0;
}

void ActivationManager::list_all(vector<string>& allActivationNames)
{
    allActivationNames.clear();
    for(unsigned int i=0; i<_vActivations.size();i++)
        allActivationNames.push_back(_vActivations[i]->name());
}
