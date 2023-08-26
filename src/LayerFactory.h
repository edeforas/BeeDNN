#ifndef LayerFactory_
#define LayerFactory_

#include <string>

namespace beednn {
class Layer;

class LayerFactory
{
public:
    static Layer* create(const std::string& sLayer ,float fArg1=0.f,float fArg2=0.f,float fArg3=0.f,float fArg4=0.f,float fArg5=0.f);
};
}
#endif
