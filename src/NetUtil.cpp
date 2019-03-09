#include "NetUtil.h"
#include "Layer.h"
#include "Matrix.h"

#include "LayerDense.h"
#include "LayerDropout.h"

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
    ss << "NbLayers: " << layers.size() << endl;
    ss << endl;

    ss << "----------------------------------------------" << endl;
    for(size_t i=0;i<layers.size();i++)
    {
        Layer* layer=layers[i];

        if(layer->type()=="Dense")
        {
            LayerDense* l=(LayerDense*)layer;
            ss << "Dense:  InSize: " << l->in_size() << " OutSize: " << l->out_size() << endl;
            if(l->has_bias())
               ss << "with bias" << endl;
            ss << "Weight:\n";
            ss << toString(l->weight());
 /*           if(l->has_bias())
            {
                ss << "Bias:\n";
                ss << matrix_to_string(l->bias());
            }
  */      }
        else if(layer->type()=="Dropout")
        {
            LayerDropout* l=(LayerDropout*)layer;
            ss << "Dropout: rate=" << l->get_rate() << endl;
        }
        else
        {
            ss << "Activation: " << layer->type() << endl;
        }
        ss << "----------------------------------------------" << endl;
    }

    return ss.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

}
