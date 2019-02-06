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
string DNNEngineTinyDnn::to_string()
{
    return "tiny-dnn";
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::clear()
{
    delete _pNet;
    _pNet=new tiny_dnn::network<tiny_dnn::sequential>;
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::init()
{
    _pNet->init_weight();
    DNNEngine::init();
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::add_layer(int inSize, int outSize, string sLayerType)
{
    if(sLayerType=="DenseAndBias")
        *_pNet << tiny_dnn::fully_connected_layer(inSize, outSize,true);

    if(sLayerType=="DenseNoBias")
        *_pNet << tiny_dnn::fully_connected_layer(inSize, outSize,false);

    if(sLayerType=="Tanh")
        *_pNet <<  tiny_dnn::tanh_layer();

    else if(sLayerType=="Sigmoid")
        *_pNet <<  tiny_dnn::sigmoid_layer();

    else if(sLayerType=="Relu")
        *_pNet <<  tiny_dnn::relu_layer();

    /*
    else if(sActivation=="Asinh")
        *_pNet <<  tiny_dnn::asinh_layer();
    else if(sActivation=="Relu")
        *_pNet <<  tiny_dnn::relu_layer();
    else if(sActivation=="LeakyRelu")
        *_pNet <<  tiny_dnn::leaky_relu_layer();
    else if(sActivation=="Selu")
        *_pNet <<  tiny_dnn::selu_layer();
    else if(sActivation=="Elu")
        *_pNet <<  tiny_dnn::elu_layer();
    else if(sActivation=="SoftPlus")
        *_pNet <<  tiny_dnn::softplus_layer();
    else if(sActivation=="Linear")
         ; // nothing to do for now
    else if(sActivation=="SoftMax")
        *_pNet <<  tiny_dnn::softmax_layer();
    else if(sActivation=="SoftSign")
        *_pNet <<  tiny_dnn::softsign_layer();*/

    else
        ; // todo error activation does not exist
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    tiny_dnn::vec_t vIn;
    vIn.assign(mIn.data(),mIn.data()+mIn.size());

    tiny_dnn::vec_t vOut=_pNet->predict(vIn);

    mOut.resize((int)vOut.size(),1);
    std::copy(vOut.data(),vOut.data()+vOut.size(),mOut.data());

    // was mOut.assign(vOut.data(),vOut.data()+vOut.size());
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    assert(mSamples.rows()==mTruth.rows());

    tiny_dnn::adamax opt;

    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;

    for(int i=0;i<mSamples.rows();i++)
    {
        tiny_dnn::vec_t tS;
        tS.assign(mSamples.row(i).data(),mSamples.row(i).data()+mSamples.cols());
        vSamples.push_back(tS);

        tiny_dnn::vec_t tT;
        tT.assign(mTruth.row(i).data(),mTruth.row(i).data()+mTruth.cols());
        vTruth.push_back(tT);
    }

    _pNet->fit<tiny_dnn::mse>(opt, vSamples, vTruth, dto.batchSize, dto.epochs, []() {},[]() {});//  on_enumerate_epoch);


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
