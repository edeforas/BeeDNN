#ifndef NetTrain_
#define NetTrain_

#include "Matrix.h"

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    double compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);
};

#endif
