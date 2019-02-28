#include "NetUtil.h"
#include "Layer.h"
#include "Matrix.h"

#include "LayerDenseAndBias.h"
#include "LayerDenseNoBias.h"
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

        if(layer->type()=="DenseNoBias")
        {
            LayerDenseNoBias* l=(LayerDenseNoBias*)layer;
            ss << "DenseNoBias:  InSize: " << l->in_size() << " OutSize: " << l->out_size() << endl;
            ss << "Weight:\n";
            ss << matrix_to_string(l->weight());
        }
        else if(layer->type()=="DenseAndBias")
        {
            LayerDenseAndBias* l=(LayerDenseAndBias*)layer;
            ss << "DenseAndBias:  InSize: " << l->in_size() << " OutSize: " << l->out_size() << endl;
            ss << "Weight:\n";
            ss << matrix_to_string(l->weight());
            ss << "Bias:\n";
            ss << matrix_to_string(l->bias());
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
