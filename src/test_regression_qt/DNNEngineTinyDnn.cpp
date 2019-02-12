#include "DNNEngineTinyDnn.h"

#include "tiny_dnn/tiny_dnn.h"

//////////////////////////////////////////////////////////////////////////////
void matrix_to_tinydnnmatrix(const MatrixFloat & m1,std::vector<tiny_dnn::vec_t>& _tinyMatrix)
{
    _tinyMatrix.clear();
    for(int i=0;i<m1.rows();i++)
    {
        tiny_dnn::vec_t tS;
        tS.assign(m1.row(i).data(),m1.row(i).data()+m1.cols());
        _tinyMatrix.push_back(tS);
    }
}
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
    string s;
    int iNbLayer=_pNet->depth();
    s+=std::to_string(iNbLayer);

    return s;
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
        *_pNet << tiny_dnn::fully_connected_layer((size_t)inSize, (size_t)outSize,true);

    if(sLayerType=="DenseNoBias")
        *_pNet << tiny_dnn::fully_connected_layer((size_t)inSize, (size_t)outSize,false);

    if(sLayerType=="Tanh")
        *_pNet <<  tiny_dnn::tanh_layer();

    if(sLayerType=="TanhP1M2")
        *_pNet <<  tiny_dnn::tanh_p1m2_layer();

    else if(sLayerType=="Sigmoid")
        *_pNet <<  tiny_dnn::sigmoid_layer();

    else if(sLayerType=="Relu")
        *_pNet <<  tiny_dnn::relu_layer();

    else if(sLayerType=="LeakyRelu")
        *_pNet <<  tiny_dnn::leaky_relu_layer();

    else if(sLayerType=="Elu")
        *_pNet <<  tiny_dnn::elu_layer();

    else if(sLayerType=="Selu")
        *_pNet <<  tiny_dnn::selu_layer();

    else if(sLayerType=="Asinh")
        *_pNet <<  tiny_dnn::asinh_layer();

    else if(sLayerType=="SoftMax")
        *_pNet <<  tiny_dnn::softmax_layer();

    else if(sLayerType=="SoftPlus")
        *_pNet <<  tiny_dnn::softplus_layer();

    else if(sLayerType=="SoftSign")
        *_pNet <<  tiny_dnn::softsign_layer();

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
    //tiny_dnn::adagrad opt;
    //tiny_dnn::adam opt;
   // tiny_dnn::RMSprop opt;
    //tiny_dnn::momentum opt;
   // tiny_dnn::nesterov_momentum opt;
   // tiny_dnn::gradient_descent opt; //test
   // opt.alpha=dto.learningRate;

    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;
    matrix_to_tinydnnmatrix(mSamples,vSamples);
    matrix_to_tinydnnmatrix(mTruth,vTruth);

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
double DNNEngineTinyDnn::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;
    matrix_to_tinydnnmatrix(mSamples,vSamples);
    matrix_to_tinydnnmatrix(mTruth,vTruth);

    return _pNet->get_loss<tiny_dnn::mse>(vSamples,vTruth);
}
//////////////////////////////////////////////////////////////////////////////
