#ifndef NetTrain_
#define NetTrain_

#include "Matrix.h"

#include <vector>
using namespace std; //temp

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    double compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);

    vector<double> loss(); //temp

protected:
    vector<double> _vdLoss; //temp

};

#endif
