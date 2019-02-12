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
        const auto& layer=layers[i];
        string sBuffer;
        layer->to_string(sBuffer);
        ss << "----------------------------------------------" << endl;
        ss << "Layer " << i+1 <<":" << endl;

        ss << sBuffer<< endl;
    }
    ss << "----------------------------------------------" << endl;

    return ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

}
