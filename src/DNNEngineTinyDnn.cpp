#include "DNNEngineTinyDnn.h"

#include "tiny_dnn/tiny_dnn.h"

//////////////////////////////////////////////////////////////////////////////
DNNEngineTinyDnn::DNNEngineTinyDnn()
{
	_pNet=new tiny_dnn::network<tiny_dnn::sequential>;
}
//////////////////////////////////////////////////////////////////////////////
DNNEngineTinyDnn::~DNNEngineTinyDnn()
{
    delete _pNet;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::clear()
{
    delete _pNet;
    _pNet=new tiny_dnn::network<tiny_dnn::sequential>;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::add_layer_and_activation(int inSize,int outSize, eLayerType layer, string sActivation)
{
    *_pNet << tiny_dnn::fully_connected_layer(inSize, outSize);

    if(sActivation=="Tanh")
        *_pNet <<  tiny_dnn::tanh_layer();

    if(sActivation=="Sigmoid")
        *_pNet <<  tiny_dnn::sigmoid_layer();

    if(sActivation=="Relu")
        *_pNet <<  tiny_dnn::relu_layer();

    if(sActivation=="SoftPlus")
        *_pNet <<  tiny_dnn::softplus_layer();

    if(sActivation=="SoftPlus")
        *_pNet <<  tiny_dnn::softplus_layer();

    if(sActivation=="SoftMax")
        *_pNet <<  tiny_dnn::softmax_layer();

    if(sActivation=="SoftSign")
        *_pNet <<  tiny_dnn::softsign_layer();
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    tiny_dnn::vec_t vIn;
    vIn.assign(mIn.data(),mIn.data()+mIn.size());

    tiny_dnn::vec_t vOut=_pNet->predict(vIn);
    mOut.assign(vOut.data(),vOut.data()+vOut.size());
}
//////////////////////////////////////////////////////////////////////////////
DNNTrainResult DNNEngineTinyDnn::train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    /*
    TrainOption tOpt;

    tOpt.epochs=dto.epochs;
    tOpt.earlyAbortMaxError=dto.earlyAbortMaxError;
    tOpt.earlyAbortMeanError=dto.earlyAbortMeanError;
    tOpt.learningRate=dto.learningRate;
    tOpt.batchSize=dto.batchSize;
    tOpt.momentum=dto.momentum;
    tOpt.observer=0;//dto.observer;
*/
    /*
    epochs=1000;
    earlyAbortMaxError=0.;
    earlyAbortMeanError=0.;
    batchSize=32;
    learningRate=0.1f;
    momentum=0.1f;
    subSamplingRatio=1;

    TrainResult tr=_pNet->train(mSamples,mTruth,tOpt);
*/
DNNTrainResult dtr;

return dtr;

}
//////////////////////////////////////////////////////////////////////////////

