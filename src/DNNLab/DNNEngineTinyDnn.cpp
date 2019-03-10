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
void matrix_to_intvector(const MatrixFloat & m1,std::vector<size_t>& _tinyMatrix)
{
    _tinyMatrix.clear();
    for(int i=0;i<m1.size();i++)
    {
        _tinyMatrix.push_back((size_t)m1(i));
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

    int iSize=(int)tinyMatrix.size();
    m1.resize(1,iSize);

    for(int i=0;i<iSize;i++)
        m1(0,i)=tinyMatrix[i];
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
    int iNbLayer=(int)_pNet->layer_size();
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

            auto w=l->weights();
            if(w.size()>0)
            {
                ss << "Weight:\n";
                MatrixFloat wmf;
                tinydnnmatrix_to_matrix(*(w[0]),wmf);
                wmf.resize(l->fan_in_size(),l->fan_out_size());
                ss << toString(wmf);

                if(w.size()>1)
                {
                    ss << "Bias:\n";
                    MatrixFloat wmb;
                    tinydnnmatrix_to_matrix(*(w[1]),wmb);
                    wmb.resize(1,l->fan_out_size());
                    ss << toString(wmb);
                }
            }
        }
        if(l->layer_type()=="dropout")
        {
            ss << "Dropout: rate=" << ((tiny_dnn::dropout_layer*)l)->dropout_rate() << endl;
        }
        else if(l->layer_type().find("activation")!=string::npos)
        {
            ss << "Activation: " << l->layer_type() << endl;
        }

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
void DNNEngineTinyDnn::add_dense_layer(int inSize, int outSize, bool bWithBias)
{
    *_pNet << tiny_dnn::fully_connected_layer((size_t)inSize, (size_t)outSize,bWithBias);
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::add_activation_layer(string sActivation)
{
    if(sActivation=="Tanh")
        *_pNet <<  tiny_dnn::tanh_layer();

    else if(sActivation=="TanhP1M2")
        *_pNet <<  tiny_dnn::tanh_p1m2_layer();

    else if(sActivation=="Sigmoid")
        *_pNet <<  tiny_dnn::sigmoid_layer();

    else if(sActivation=="Relu")
        *_pNet <<  tiny_dnn::relu_layer();

    else if(sActivation=="LeakyRelu")
        *_pNet <<  tiny_dnn::leaky_relu_layer();

    else if(sActivation=="Elu")
        *_pNet <<  tiny_dnn::elu_layer();

    else if(sActivation=="Selu")
        *_pNet <<  tiny_dnn::selu_layer();

    else if(sActivation=="Asinh")
        *_pNet <<  tiny_dnn::asinh_layer();

    else if(sActivation=="SoftMax")
        *_pNet <<  tiny_dnn::softmax_layer();

    else if(sActivation=="SoftPlus")
        *_pNet <<  tiny_dnn::softplus_layer();

    else if(sActivation=="SoftSign")
        *_pNet <<  tiny_dnn::softsign_layer();

    else
        ; // todo error activation does not exist
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::add_dropout_layer(int inSize,float fRatio)
{
    *_pNet << tiny_dnn::dropout_layer(inSize,fRatio);
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
int DNNEngineTinyDnn::classify(const MatrixFloat& mIn)
{
    tiny_dnn::vec_t vIn;
    vIn.assign(mIn.data(),mIn.data()+mIn.size());

    return (int)(_pNet->predict_label(vIn));
}
//////////////////////////////////////////////////////////////////////////////
void DNNEngineTinyDnn::learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)
{
    int iEpoch=0;
    float fLoss=0.f;
    assert(mSamples.rows()==mTruth.rows());

    tiny_dnn::optimizer* opt=nullptr;

    if(dto.optimizer=="SGD")
    {
        tiny_dnn::gradient_descent* op=new tiny_dnn::gradient_descent;
        op->alpha=dto.learningRate;
        opt=op;
    }

    if(dto.optimizer=="Momentum")
    {
        tiny_dnn::momentum* op=new tiny_dnn::momentum;
        op->alpha=dto.learningRate;
        op->mu=dto.momentum;
        opt=op;
    }

    if(dto.optimizer=="Nesterov")
    {
        tiny_dnn::nesterov_momentum* op=new tiny_dnn::nesterov_momentum;
        op->alpha=dto.learningRate;
        op->mu=dto.momentum;
        opt=op;
    }

    if(dto.optimizer=="Adamax")
    {
        tiny_dnn::adamax* op=new tiny_dnn::adamax;
        op->alpha=dto.learningRate;
        opt=op;
    }

    if(dto.optimizer=="Adagrad")
    {
        tiny_dnn::adagrad* op=new tiny_dnn::adagrad;
        op->alpha=dto.learningRate;
        opt=op;
    }

    if(dto.optimizer=="Adam")
    {
        tiny_dnn::adam* op=new tiny_dnn::adam;
        op->alpha=dto.learningRate;
        opt=op;
    }

    if(dto.optimizer=="RMSProp")
    {
        tiny_dnn::RMSprop *op=new tiny_dnn::RMSprop;
        op->alpha=dto.learningRate;
        op->mu=dto.decay; //?
        opt=op;
    }

    assert(opt!=nullptr);

    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;
    std::vector<size_t> vTruthi;
    matrix_to_tinydnnmatrix(mSamples,vSamples);
    matrix_to_tinydnnmatrix(mTruth,vTruth);
    matrix_to_intvector(mTruth,vTruthi);

    if(dto.lossFunction=="mse")
    {
        // this lambda function will be called after each epoch
        auto on_enumerate_epoch = [&]()
        {
            if(dto.testEveryEpochs!=-1)
                if( (iEpoch % dto.testEveryEpochs) == 0)
                    fLoss = _pNet->get_loss<tiny_dnn::mse>(vSamples, vTruth)/vSamples.size();

            _vdLoss.push_back(fLoss);
            iEpoch++;
        };

        if(_bClassification)
            _pNet->train<tiny_dnn::mse>(*opt, vSamples, vTruthi, dto.batchSize, dto.epochs, []() {},on_enumerate_epoch);//  on_enumerate_epoch);
        else
            _pNet->fit<tiny_dnn::mse>(*opt, vSamples, vTruth, dto.batchSize, dto.epochs, []() {},on_enumerate_epoch);//  on_enumerate_epoch);
    }

    delete opt;
}
//////////////////////////////////////////////////////////////////////////////
double DNNEngineTinyDnn::compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)
{
    std::vector<tiny_dnn::vec_t> vSamples;
    std::vector<tiny_dnn::vec_t> vTruth;
    matrix_to_tinydnnmatrix(mSamples,vSamples);
    matrix_to_tinydnnmatrix(mTruth,vTruth);

    return _pNet->get_loss<tiny_dnn::mse>(vSamples,vTruth)/vSamples.size();
}
//////////////////////////////////////////////////////////////////////////////
