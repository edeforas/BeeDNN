#include "FrameNetwork.h"
#include "ui_FrameNetwork.h"

#include "Activation.h"
#include "mainwindow.h"

#include <QComboBox>
#include "Net.h"
#include "Layer.h"

#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGaussianNoise.h"

#include <QObject>

FrameNetwork::FrameNetwork(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameNetwork)
{
    _pMainWindow=nullptr;
    _pNet=nullptr;

    _bLock=true;

    ui->setupUi(this);

    vector<string> vsActivations;
    list_activations_available( vsActivations);

    QStringList qsl;
    qsl+="LayerType";
    qsl+="InSize";
    qsl+="OutSize";
    qsl+="Arg1";

    ui->twNetwork->setHorizontalHeaderLabels(qsl);

    for(int i=0;i<10;i++)
    {
        QComboBox*  qcbType=new QComboBox;
        qcbType->addItem("");
        qcbType->addItem("DenseAndBias");
        qcbType->addItem("DenseNoBias");
        qcbType->addItem("Dropout");
        qcbType->addItem("GaussianNoise");
        qcbType->addItem("GlobalGain");
        qcbType->addItem("PoolAveraging1D");
        qcbType->addItem("Softmax");

        qcbType->insertSeparator(8);

        for(unsigned int a=0;a<vsActivations.size();a++)
            qcbType->addItem(vsActivations[a].c_str());

       // QObject::connect(qcbType,SIGN&QComboBox::currentIndexChanged, this,&FrameNetwork::on_type_changed);
        connect(qcbType,SIGNAL(currentIndexChanged(int)), this, SLOT(type_changed()));
        ui->twNetwork->setCellWidget(i,0,qcbType);
    }

    ui->twNetwork->setItem(0,1,new QTableWidgetItem("1")); //first input size is 1
    ui->twNetwork->adjustSize();

    _bLock=false;
}
//////////////////////////////////////////////////////////////
FrameNetwork::~FrameNetwork()
{
    delete ui;
}
//////////////////////////////////////////////////////////////
void FrameNetwork::init()
{
    _bLock=true;
    for(int i=0;i<10;i++)
    {
        ((QComboBox*)(ui->twNetwork->cellWidget(i,0)))->setCurrentIndex(0);
        ui->twNetwork->setItem(i,1,new QTableWidgetItem(""));
        ui->twNetwork->setItem(i,2,new QTableWidgetItem(""));
        ui->twNetwork->setItem(i,3,new QTableWidgetItem(""));
    }
    _bLock=false;
}
//////////////////////////////////////////////////////////////
void FrameNetwork::set_main_window(MainWindow* pMainWindow)
{
    _pMainWindow=pMainWindow;
}
//////////////////////////////////////////////////////////////
void FrameNetwork::set_net(Net* pNet)
{
    _pNet=pNet;
	_bLock = true;

    // write input size
    if(_pNet->layers().empty())
        ui->twNetwork->setItem(0,1,new QTableWidgetItem(to_string(_pNet->input_size()).data()));

    //todo redraw all
    auto layers= _pNet->layers();
    for(unsigned int i=0;i<layers.size();i++)
    {
        auto l=layers[i];
        string sType=l->type();
        if(sType=="Dense")
        {
            if(((LayerDense*)l)->has_bias())
                sType="DenseAndBias";
            else
                sType="DenseNoBias";
        }

        ((QComboBox*)ui->twNetwork->cellWidget(i,0))->setCurrentText(sType.c_str());

        if(sType=="GaussianNoise")
        {
            float fStd=((LayerGaussianNoise*)l)->get_std();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fStd).c_str()));
        }

        if(sType=="Dropout")
        {
            float fRate=((LayerDropout*)l)->get_rate();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fRate).c_str()));
        }

        if(sType=="GlobalGain")
        {
            float fGain=((LayerGlobalGain*)l)->gain();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fGain).c_str()));
        }

        if(l->in_size())
            ui->twNetwork->setItem(i,1,new QTableWidgetItem(to_string(l->in_size()).c_str()));

        if(l->out_size())
            ui->twNetwork->setItem(i,2,new QTableWidgetItem(to_string(l->out_size()).c_str()));
    }
	_bLock = false;
}
//////////////////////////////////////////////////////////////
void FrameNetwork::on_twNetwork_cellChanged(int row, int column)
{
    (void)row;
    (void)column;

    if(_bLock)
        return;

    //rescan all for now
    int iLastOut=_pNet->input_size();
    _pNet->clear();

    bool bOk;
    float fArg1=0.f;

    for(int iRow=0;iRow<10;iRow++) //todo dynamic size
    {
        QComboBox* pCombo=(QComboBox*)(ui->twNetwork->cellWidget(iRow,0));
        if(!pCombo)
            continue;
        string sType=pCombo->currentText().toStdString();

        QTableWidgetItem* pwiInSize=ui->twNetwork->item(iRow,1); //todo not used in activation
        int iInSize=0;
        if(!pwiInSize)
            iInSize=iLastOut; //use last out
        else
        {
            int iIn=pwiInSize->text().toInt(&bOk);
            if(bOk)
                iInSize=iIn;
            else
                iInSize=iLastOut;
        }

        QTableWidgetItem* pwiOutSize=ui->twNetwork->item(iRow,2); //todo not used in activation
        int iOutSize;
        if(!pwiOutSize)
            iOutSize=iInSize; //same size (i.e. activation case)
        else
        {
            int iOut=pwiOutSize->text().toInt(&bOk);
            if(bOk)
                iOutSize=iOut;
            else
                iOutSize=iInSize;
        }

        iLastOut=iOutSize;

        QTableWidgetItem* pwArg1=ui->twNetwork->item(iRow,3);
        if(pwArg1)
            fArg1=pwArg1->text().toFloat(&bOk);
        else
            bOk=false;

        if(!sType.empty())
        {
            if(sType=="Dropout")
            {
                float fRatio=0.2f; //by default
                if(bOk)
                    fRatio=fArg1;
                _pNet->add_dropout_layer(iInSize,fRatio);
            }
            else if(sType=="GaussianNoise")
            {
                float fStd=1.f; //by default
                if(bOk)
                    fStd=fArg1;
                _pNet->add_gaussian_noise_layer(iInSize,fStd);
            }
            else if(sType=="GlobalGain")
            {
                float fGain=0.f; //by default, 0.f mean learned
                if(bOk)
                    fGain=fArg1;
                _pNet->add_globalgain_layer(iInSize,fGain);
            }
            else if(sType=="PoolAveraging1D")
                _pNet->add_poolaveraging1D_layer(iInSize,iOutSize);
			else if (sType == "Softmax")
				_pNet->add_softmax_layer();
            else if(sType=="DenseAndBias")
                _pNet->add_dense_layer(iInSize,iOutSize,true);
            else if(sType=="DenseNoBias")
                _pNet->add_dense_layer(iInSize,iOutSize,false);
            else
                _pNet->add_activation_layer(sType);
        }
    }

    _pMainWindow->model_changed(this);
}
//////////////////////////////////////////////////////////////
void FrameNetwork::type_changed()
{
	on_twNetwork_cellChanged(0, 0);
}
//////////////////////////////////////////////////////////////
