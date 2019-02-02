#include "NetUtil.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

#include <sstream>
using namespace std;

namespace NetUtil {

/////////////////////////////////////////////////////////////////////////////////////////////////
string to_string(const Net* pNet)
{
    stringstream ss;

    auto layers=pNet->layers();
    ss << "Engine: testDnn" << endl;
    ss << endl;
    ss << "Nb Layers: " << layers.size() << endl;
    ss << endl;

    for(size_t i=0;i<layers.size();i++)
    {
        auto layer=layers[i];
        ss << "----------------------------------------------" << endl;
        ss << "Layer " << i+1 <<":" << endl;
        //ss << "type: " << "dense with bias and activation inSize= " << layer->get_weight().rows()-1 << "  outSize= " << layer->get_weight().cols() << endl;
        ss << "weight:" << endl;
  //      ss << MatrixUtil::to_string(layer->get_weight()) << endl;
    }
    ss << "----------------------------------------------" << endl;

    return ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

}
