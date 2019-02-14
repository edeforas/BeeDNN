#include "DNNEngineTinyDnn.h"

#include "tiny_dnn/tiny_dnn.h"
#include "MatrixUtil.h"

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
void tinydnnmatrix_to_matrix(const tiny_dnn::vec_t& tinyMatrix, MatrixFloat& m1)
{
    if(tinyMatrix.empty())
    {
        m1.resize(0,0);
        return;
    }

    //row major
    int iSize=tinyMatrix.size();

/*
    int iNbCols=tinyMatrix[0]->size();

    m1.resize(iNbRows,iNbCols);

    for(int r=0;r<iNbRows;r++)
        for(int c=0;c<iNbCols;c++)
            m1(r,c)=(*_tinyMatrix[r])[c];
*/
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
    int iNbLayer=_pNet->layer_size();
    stringstream ss;
    ss << "Engine: tiny-dnn" << endl;
    ss << endl;
    ss << "NbLayers: " << iNbLayer << endl;
    ss << endl;

    ss << "----------------------------------------------" << endl;
    for(size_t i=0;i<iNbLayer;i++)
    {
        const tiny_dnn::layer* l= _pNet->operator[](i);

        if(l->layer_type()=="fully-connected")
        {
            ss << "fully-connected: InSize: " << l->fan_in_size() << " OutSize: " << l->fan_out_size() << endl;

            MatrixFloat mf;
     //       tinydnnmatrix_to_matrix(l->weights(),mf);

            ss << "Weight:\n";
            ss << MatrixUtil::to_string(mf);
            //           ss << "Bias:\n";
            //          ss << MatrixUtil::to_string(l->bias());

        }

        if(l->layer_type()=="tanh-activation")
        {
            ss << "Activation: " << l->layer_type() << endl;
        }

        if(l->layer_type()=="relu-activation")
        {
            ss << "Activation: " << l->layer_type() << endl;
        }


        /*




        Layer* layer=layers[i];

        if(layer->type()=="DenseNoBias")
        {
            LayerDenseNoBias* l=(LayerDenseNoBias*)layer;
            ss << "DenseNoBias:  InSize: " << l->in_size() << " OutSize: " << l->out_size() << endl;
            ss << "Weight:\n";
            ss << MatrixUtil::to_string(l->weight());
        }
        else if(layer->type()=="DenseAndBias")
        {
            LayerDenseAndBias* l=(LayerDenseAndBias*)layer;
            ss << "DenseAndBias:  InSize: " << l->in_size() << " OutSize: " << l->out_size() << endl;
            ss << "Weight:\n";
            ss << MatrixUtil::to_string(l->weight());
            ss << "Bias:\n";
            ss << MatrixUtil::to_string(l->bias());
        }
        else
        {
            ss << "Activation: " << layer->type() << endl;
        }
        */
        ss << "----------------------------------------------" << endl;
    }

    return ss.str();
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

    tiny_dnn::optimizer* opt=nullptr;

    if(dto.optimizer=="gradient_descent")
    {
        tiny_dnn::gradient_descent* op=new tiny_dnn::gradient_descent;
        op->alpha=dto.learningRate;
        opt=op;
    }

    if(dto.optimizer=="momentum")
    {
        tiny_dnn::momentum* op=new tiny_dnn::momentum;
        op->alpha=dto.learningRate;
        op->mu=dto.momentum;
        opt=op;
    }

    if(dto.optimizer=="nesterov_momentum")
    {
        tiny_dnn::nesterov_momentum* op=new tiny_dnn::nesterov_momentum;
        op->alpha=dto.learningRate;
        op->mu=dto.momentum;
        opt=op;
    }

    if(dto.optimizer=="adamax")
    {
        opt=new tiny_dnn::adamax;
    }

    if(dto.optimizer=="adagrad")
    {
        opt=new tiny_dnn::adagrad;
    }

    if(dto.optimizer=="adam")
    {
        opt=new tiny_dnn::adam;
    }

    if(dto.optimizer=="RMSprop")
    {
        opt=new tiny_dnn::RMSprop;
    }

    assert(opt!=nullptr);

    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;
    matrix_to_tinydnnmatrix(mSamples,vSamples);
    matrix_to_tinydnnmatrix(mTruth,vTruth);

    _pNet->fit<tiny_dnn::mse>(*opt, vSamples, vTruth, dto.batchSize, dto.epochs, []() {},[]() {});//  on_enumerate_epoch);

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
    delete opt;
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