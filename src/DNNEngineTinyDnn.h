#ifndef DNNEngineTinyDnn_
#define DNNEngineTinyDnn_

#include "DNNEngine.h"

// forward declaration of tiny_dnn
namespace tiny_dnn {
template <typename NetType>
class network;
class sequential;
}

class DNNEngineTinyDnn : public DNNEngine
{
public:
    DNNEngineTinyDnn();
    virtual ~DNNEngineTinyDnn();

    virtual void clear();

    virtual void predict(const MatrixFloat mIn, MatrixFloat mOut);

private:
    tiny_dnn::network<tiny_dnn::sequential>* _pNet;
};

#endif
