#ifndef DNNEngine_
#define DNNEngine_

#include <string>
using namespace std;


#include "Matrix.h"

enum eLayerType
{
	FullyConnected=1
};




class DNNEngine
{
public:
    DNNEngine();
    virtual ~DNNEngine();

    virtual void clear()=0;
    virtual void add_layer_and_activation(int inSize,int outSize, eLayerType layer, string sActivation)=0;
	
	
    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut)=0;

};

#endif
