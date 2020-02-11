#ifndef FrameNetwork_
#define FrameNetwork_

#include <string>
using namespace std;

class Layer;

class LayerFactory
{
public:
    static Layer* create(string sLayer,float fArg1=0.f,float fArg2=0.f,float fArg3=0.f,float fArg4=0.f,float fArg5=0.f);	
};

#endif
