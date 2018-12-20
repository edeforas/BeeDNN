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
    assert(mSamples.rows()==mTruth.rows());

    tiny_dnn::adamax opt;

    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;

    for(int i=0;i<mSamples.rows();i++)
    {
        tiny_dnn::vec_t tS;
        tS.assign(mSamples.row(i).data(),mSamples.row(i).data()+mSamples.columns());
        vSamples.push_back(tS);

        tiny_dnn::vec_t tT;
        tT.assign(mTruth.row(i).data(),mTruth.row(i).data()+mTruth.columns());
        vTruth.push_back(tT);
    }

    _pNet->fit<tiny_dnn::mse>(opt, vSamples, vTruth, dto.batchSize, dto.epochs, []() {},[]() {});//  on_enumerate_epoch);

    DNNTrainResult dtr;
    return dtr;

 /*

  // this lambda function will be called after each epoch
  auto on_enumerate_epoch = [&]() {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_dnn::mse>(X, sinusX);
    std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss
              << std::endl;
  };

 */
}
//////////////////////////////////////////////////////////////////////////////
