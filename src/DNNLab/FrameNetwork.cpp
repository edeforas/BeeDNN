#include "FrameNetwork.h"
#include "ui_FrameNetwork.h"

#include "Activation.h"
#include "mainwindow.h"

#include <QComboBox>
#include "Net.h"
#include "Layer.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerGaussianDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGlobalBias.h"
#include "LayerBias.h"
#include "LayerGain.h"
#include "LayerUniformNoise.h"
#include "LayerGaussianNoise.h"
#include "LayerPRelu.h"
#include "LayerPoolMax2D.h"
#include "LayerSoftmax.h"

#include <QObject>


#include <sstream>
#include <vector>
using namespace std;

void split(string s,vector<string>& vsItems)
{
	vsItems.clear();
	
    istringstream f(s);
    string sitem;    
    while (getline(f, sitem, ','))
        vsItems.push_back(sitem);
}

FrameNetwork::FrameNetwork(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameNetwork)
{
    _pMainWindow=nullptr;
    _pNet=nullptr;

    _bLock=true;

    ui->setupUi(this);

    list_activations_available( _vsActivations);

    QStringList qsl;
    qsl+="LayerType";
    qsl+="InSize";
    qsl+="OutSize";
    qsl+="Arg1";

    ui->twNetwork->setHorizontalHeaderLabels(qsl);

    for(int i=0;i<10;i++)
		add_new_row();

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
void FrameNetwork::parse_cell(string sCell, float& fVal1, float& fVal2, float& fVal3)
{
	fVal1 = 0.f;
	fVal2 = 0.f;
	fVal3 = 0.f;
	vector<string> vsItems;
	split(sCell, vsItems);

	if (vsItems.size() == 0)
		return;

	fVal1 = stof( vsItems[0]);

	if (vsItems.size() <2)
		return;

	fVal2 = stof(vsItems[1]);

	if (vsItems.size() < 3)
		return;

	fVal2 = stof(vsItems[2]);
}
//////////////////////////////////////////////////////////////
void FrameNetwork::set_net(Net* pNet)
{
    _pNet=pNet;
	_bLock = true;

    // write input size, TODO 2D size
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
			LayerDense* ld = (LayerDense*)l;
			ui->twNetwork->setItem(i, 1, new QTableWidgetItem(to_string(ld->input_size()).c_str()));
			ui->twNetwork->setItem(i, 2, new QTableWidgetItem(to_string(ld->output_size()).c_str()));

            if(((LayerDense*)l)->has_bias())
                sType="DenseAndBias";
            else
                sType="DenseNoBias";
        }

        ((QComboBox*)ui->twNetwork->cellWidget(i,0))->setCurrentText(sType.c_str());

		if (sType == "UniformNoise")
		{
			float fNoise = ((LayerUniformNoise*)l)->get_noise();
			ui->twNetwork->setItem(i, 3, new QTableWidgetItem(to_string(fNoise).c_str()));
		}

        if(sType=="GaussianNoise")
        {
            float fStd=((LayerGaussianNoise*)l)->get_std();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fStd).c_str()));
        }

        if(sType=="GaussianDropout")
        {
            float fProba=((LayerGaussianDropout*)l)->get_proba();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fProba).c_str()));
        }

        if(sType=="Dropout")
        {
            float fRate=((LayerDropout*)l)->get_rate();
            ui->twNetwork->setItem(i,3,new QTableWidgetItem(to_string(fRate).c_str()));
        }
	
		// overwrite cells in case of 2d
		if (sType == "PoolMax2D")
		{
			LayerPoolMax2D* lpm=((LayerPoolMax2D*)l);
			Index iInRows, iInCols, iPlanes, iRowFactor, iColFactor;
			lpm->get_params(iInRows,iInCols,iPlanes, iRowFactor,iColFactor);
			string cell1 = to_string(iInRows) + "," + to_string(iInCols) + "," + to_string(iPlanes);
			string cell3 = to_string(iRowFactor) + "," + to_string(iColFactor);
			ui->twNetwork->setItem(i, 1, new QTableWidgetItem(cell1.c_str()));
			ui->twNetwork->setItem(i, 3, new QTableWidgetItem(cell3.c_str()));
		}
		// overwrite cells in case of 2d
		if (sType == "Convolution2D")
		{
			LayerConvolution2D* lpm = ((LayerConvolution2D*)l);
            Index iInRows, iInCols, iInChannels, iKernelRows, iKernelCols, iOutChannels, iRowStride,iColStride;
            lpm->get_params(iInRows, iInCols, iInChannels, iKernelRows, iKernelCols, iOutChannels,iRowStride,iColStride);

            ///TODO use iRowStride, iColStride

            string cell1 = to_string(iInRows) + "," + to_string(iInCols) + "," + to_string(iInChannels);
			string cell3 = to_string(iKernelRows) + "," + to_string(iKernelCols) + "," + to_string(iOutChannels);
			ui->twNetwork->setItem(i, 1, new QTableWidgetItem(cell1.c_str()));
			ui->twNetwork->setItem(i, 3, new QTableWidgetItem(cell3.c_str()));
		}
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

    //rescan all (for now)
    int iLastOut=_pNet->input_size();
    _pNet->clear();

    for(int iRow=0;iRow<10;iRow++) //todo dynamic size
    {
		bool bOk=false;
		QComboBox* pCombo=(QComboBox*)(ui->twNetwork->cellWidget(iRow,0));
        if(!pCombo)
            continue;
        string sType=pCombo->currentText().toStdString();

		//scan input size TODO 2D
		QTableWidgetItem* pwiInSize=ui->twNetwork->item(iRow,1); //todo not used in activation
        int iInSize=0, iInSize2=0, iInSize3 = 0;
		if (!pwiInSize)
		{
			iInSize = iLastOut; //use last out
		}
		else
        {
			float f1, f2,f3;
			parse_cell(pwiInSize->text().toStdString(), f1, f2,f3);
			if (f1 != 0.f)
			{
				iInSize = (int)f1;
				iInSize2 = (int)f2;
				iInSize3 = (int)f3;
			}
			else
				iInSize = iLastOut;
		}

		//scan output size TODO 2D
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

		//scan arguments TODO 2D
        float fArg1=0.f, fArg2=0.f, fArg3 = 0.f;
		QTableWidgetItem* pwArg1=ui->twNetwork->item(iRow,3);
        if(pwArg1)
			parse_cell(pwArg1->text().toStdString(), fArg1, fArg2, fArg3);
		else
			bOk=false;

        if(!sType.empty())
        {
            if(sType=="Dropout")
            {
                float fRatio=0.2f; //by default
                if(bOk)
                    fRatio=fArg1;
                _pNet->add(new LayerDropout(fRatio));
            }

			else if (sType == "UniformNoise")
			{
				float fNoise = 0.1f; //by default
				if (bOk)
					fNoise = fArg1;
				_pNet->add(new LayerUniformNoise(fNoise));
			}

            else if(sType=="GaussianNoise")
            {
                float fStd=1.f; //by default
                if(bOk)
                    fStd=fArg1;
                _pNet->add(new LayerGaussianNoise(fStd));
            }

            else if(sType=="GaussianDropout")
            {
                float fProba=1.f; //by default
                if(bOk)
                    fProba=fArg1;
                _pNet->add(new LayerGaussianDropout(fProba));
            }

            else if(sType=="GlobalGain")
                _pNet->add(new LayerGlobalGain());
			
			else if (sType == "GlobalBias")
				_pNet->add(new LayerGlobalBias());
			
            else if (sType == "Bias")
                _pNet->add(new LayerBias());

			else if (sType == "ChannelBias")
				_pNet->add(new LayerChannelBias(iInSize, iInSize2, max(iInSize3, 1)));

			else if (sType == "Gain")
				_pNet->add(new LayerGain());

			else if (sType == "PoolMax2D")
				_pNet->add(new LayerPoolMax2D(iInSize, iInSize2, max(iInSize3,1), max((int)fArg1,1), max((int)fArg2,1)));
			
			else if (sType == "Convolution2D")
				_pNet->add(new LayerConvolution2D(iInSize, iInSize2, max(iInSize3, 1), max((int)fArg1, 1), max((int)fArg2, 1), max((int)fArg3, 1)));

			else if (sType == "PRelu")
				_pNet->add(new LayerPRelu());

			else if (sType == "Softmax")
				_pNet->add(new LayerSoftmax());

            else if(sType=="DenseAndBias")
                _pNet->add(new LayerDense(iInSize,iOutSize,true));

            else if(sType=="DenseNoBias")
                _pNet->add(new LayerDense(iInSize,iOutSize,false));
            else
                _pNet->add(new LayerActivation(sType));
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
void FrameNetwork::on_btnNetworkInsert_clicked()
{
	int iRow = ui->twNetwork->currentRow();
	if (iRow == -1)
		return;

	add_new_row(iRow);
	on_twNetwork_cellChanged(0, 0);
}
//////////////////////////////////////////////////////////////
void FrameNetwork::on_btnNetworkRemove_clicked()
{
	int iRow = ui->twNetwork->currentRow();
	if (iRow == -1)
		return;

	ui->twNetwork->removeRow(iRow);
	add_new_row(); // to keep at least 1 free row at the end
	on_twNetwork_cellChanged(0, 0);
}
//////////////////////////////////////////////////////////////
void FrameNetwork::add_new_row(int iRow)
{
	if(iRow==-1)
		iRow = ui->twNetwork->rowCount(); //append at the end
	
	ui->twNetwork->insertRow(iRow);

	QComboBox*  qcbType = new QComboBox;
	qcbType->addItem("");
	qcbType->addItem("DenseAndBias");
	qcbType->addItem("DenseNoBias");
	qcbType->addItem("Dropout");
	qcbType->addItem("UniformNoise");
	qcbType->addItem("GaussianNoise");
	qcbType->addItem("GaussianDropout");
	qcbType->addItem("GlobalGain");
	qcbType->addItem("Gain");
	qcbType->addItem("GlobalBias");
    qcbType->addItem("Bias");
	qcbType->addItem("PRelu");
	qcbType->addItem("Softmax");
	qcbType->insertSeparator(qcbType->count());
	qcbType->addItem("ChannelBias");
	qcbType->addItem("Convolution2D");
	qcbType->addItem("PoolMax2D");
	qcbType->insertSeparator(qcbType->count());

	for (unsigned int a = 0; a < _vsActivations.size(); a++)
		qcbType->addItem(_vsActivations[a].c_str());

    qcbType->setMaxVisibleItems(1000);

	ui->twNetwork->setCellWidget(iRow, 0, qcbType);
	connect(qcbType, SIGNAL(currentIndexChanged(int)), this, SLOT(type_changed()));

    ui->twNetwork->resizeColumnToContents(0);
}
//////////////////////////////////////////////////////////////
