#ifndef NNEngineTinyDnn_
#define NNEngineTinyDnn_

#include "NNEngine.h"

// forward declaration of tiny_dnn
namespace tiny_dnn {
template <typename NetType>
class network;
class sequential;
}

class NNEngineTinyDnn : public NNEngine
{
public:
    NNEngineTinyDnn();
    virtual ~NNEngineTinyDnn();

    virtual void predict(const MatrixFloat mIn, MatrixFloat mOut);

private:
    tiny_dnn::network<tiny_dnn::sequential>* _pNet;
};

#endif
