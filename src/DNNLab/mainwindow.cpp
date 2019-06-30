#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QColorDialog>
#include <QFileDialog>
#include <QClipboard>
#include <QComboBox>

#include <fstream>
using namespace std;

#include "SimpleCurveWidget.h"

#include "MLEngineBeeDnn.h"

#include "DataSource.h"

#include "ConfusionMatrix.h"
#include "NetUtil.h"

//////////////////////////////////////////////////////////////////////////
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
  , ui(new Ui::MainWindow)
{
    _pDataSource=nullptr;
    _pEngine=nullptr;

    ui->setupUi(this);

    ui->frameNotes->set_main_window(this);
    ui->frameGlobal->set_main_window(this);
    ui->frameLearning->set_main_window(this);
    ui->frameNetwork->set_main_window(this);

    resizeDocks({ui->dockWidget},{1},Qt::Horizontal);
    _qsRegression=new SimpleCurveWidget;
    _qsRegression->addXAxis();
    _qsRegression->addYAxis();
    ui->layoutRegression->addWidget(_qsRegression);

    _qsLoss=new SimpleCurveWidget;
    _qsLoss->addXAxis();
    _qsLoss->addYAxis();
    ui->layoutLossCurve->addWidget(_qsLoss);

    _qsAccuracy=new SimpleCurveWidget;
    _qsAccuracy->addXAxis();
    _qsAccuracy->addYAxis();
    ui->layoutAccuracyCurve->addWidget(_qsAccuracy);

    init_all();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::init_all()
{
    delete _pEngine;
    _pEngine=new MLEngineBeeDnn;

    delete _pDataSource;
    _pDataSource=new DataSource;
	_pDataSource->load("Sin");
	_pEngine->net().set_input_size(_pDataSource->data_size());

    ui->frameLearning->init();
    ui->frameGlobal->init();
    ui->frameNetwork->init();

    _qsAccuracy->clear();
    _qsLoss->clear();
    _qsRegression->clear();
    ui->twConfusionMatrixTrain->clear();
    ui->peDetails->clear();

    _bMustSave=false;
    //   _sFileName="";

    _curveColor=0xff0000; //red

    model_changed(0);

    updateTitle();
}
//////////////////////////////////////////////////////////////////////////
MainWindow::~MainWindow()
{
    delete ui;
    delete _pEngine;
    delete _pDataSource;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::train_and_test(bool bReset,bool bLearn)
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    if(bLearn)
        _bMustSave=true;

    if(bReset)
        _pEngine->init();

    _pEngine->set_problem(ui->frameGlobal->is_classification_problem());

    if(bLearn)
    {
        DNNTrainResult dtr =_pEngine->learn(_pDataSource->train_data(),_pDataSource->train_truth());

        ui->leComputedEpochs->setText(QString::number(dtr.computedEpochs));
        ui->leTimeByEpoch->setText(QString::number(dtr.epochDuration));

        drawLoss(dtr.loss);
        drawAccuracy(dtr.accuracy);
    }
    else
    {
        _qsRegression->clear();
        _qsLoss->clear();
        _qsAccuracy->clear();
    }

    float fLoss=_pEngine->compute_loss(_pDataSource->train_data(),_pDataSource->train_truth()); //final loss
    ui->leMSE->setText(QString::number((double)fLoss));

    updateTitle();
    drawRegression();
    update_classification_tab();
    update_details();

    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<float> vfLoss)
{
    if(!ui->cbHoldOn->isChecked())
        _qsLoss->clear();

    vector<float> x;
    for(unsigned int i=0;i<vfLoss.size();i++)
        x.push_back(i);

    _qsLoss->addHorizontalLine(0.);
    _qsLoss->addCurve(x,vfLoss,_curveColor);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawAccuracy(vector<float> vfAccuracy)
{
    if(!ui->cbHoldOn->isChecked())
        _qsAccuracy->clear();

    if(_pEngine->is_classification_problem())
    {
        ui->gbTrainAccuracy->setTitle("Train accuracy");

        // add 0%, 25% , 50%, 100%
        _qsAccuracy->addHorizontalLine(0.);
        _qsAccuracy->addHorizontalLine(25.);
        _qsAccuracy->addHorizontalLine(50.);
        _qsAccuracy->addHorizontalLine(75.);
        _qsAccuracy->addHorizontalLine(100.);

        //draw accuracy
        vector<float> x;
        for(unsigned int i=0;i<vfAccuracy.size();i++)
            x.push_back(i);

        _qsAccuracy->addCurve(x,vfAccuracy,_curveColor);
    }
    else
    {
        //draw euclidian distance
        ui->gbTrainAccuracy->setTitle("Train Euclidian distance");
    }
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawRegression()
{
    _qsRegression->clear();

    if(_pEngine->is_classification_problem()==true)
        return; //not a regression problem
    
	_qsRegression->addHorizontalLine(0.);

	bool bPlotTrainTruth = true, bPlotTestTruth = false, bPlotTrainPredicted = false, bPlotTestPredicted = true; //todo use checkbox
	bool bHasTrainData = _pDataSource->has_train_data();
	bool bHasTestData = _pDataSource->has_test_data();
	bPlotTrainTruth &= bHasTrainData;
	bPlotTestTruth &= bHasTrainData;
	bPlotTrainPredicted &= bHasTrainData;
	bPlotTestPredicted &= bHasTestData;
	const MatrixFloat& mTrainData = _pDataSource->train_data();
	const MatrixFloat& mTestData = _pDataSource->test_data();

	//plot train truth
	if (bPlotTrainTruth)
	{    
		vector<double> vSamples;
		vector<double> vTruth;
		const MatrixFloat& mTrainTruth = _pDataSource->train_truth();
		for (int i = 0; i < mTrainTruth.size(); i++)
		{
			vSamples.push_back(mTrainData(i));
			vTruth.push_back(mTrainTruth(i));
		}

		_qsRegression->addCurve(vSamples, vTruth, 0xFF0000);
	}

	//plot test truth
	if (bPlotTestTruth)
	{
		vector<double> vSamples;
		vector<double> vTruth;
		const MatrixFloat& mTestTruth = _pDataSource->test_truth();
		for (int i = 0; i < mTestTruth.size(); i++)
		{
			vSamples.push_back(mTestData(i));
			vTruth.push_back(mTestTruth(i));
		}

		_qsRegression->addCurve(vSamples, vTruth, 0x0000FF);
	}

	//plot predicted train
	if (bPlotTrainPredicted)
	{
		MatrixFloat mPredictedTrain;
		_pEngine->predict(mTrainData, mPredictedTrain);

		vector<double> vSamples;
		vector<double> vTruth;
		for (int i = 0; i < mPredictedTrain.size(); i++)
		{
			vSamples.push_back(mTrainData(i));
			vTruth.push_back(mPredictedTrain(i));
		}

		_qsRegression->addCurve(vSamples, vTruth, 0x7F0000);
	}


	//plot predicted test
	if (bPlotTestPredicted)
	{
		MatrixFloat mPredictedTest;
		_pEngine->predict(mTestData, mPredictedTest);

		vector<double> vSamples;
		vector<double> vTruth;
		for (int i = 0; i < mPredictedTest.size(); i++)
		{
			vSamples.push_back(mTestData(i));
			vTruth.push_back(mPredictedTest(i));
		}

		_qsRegression->addCurve(vSamples, vTruth, 0x00007F);
	}
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionQuit_triggered()
{
    if(ask_save())
        close();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionAbout_triggered()
{
    QMessageBox mb;
    QString qsText="DNNLab";
    qsText+= "\n";
    qsText+= "\n GitHub: https://github.com/edeforas/BeeDNN";
    qsText+= "\n by Etienne de Foras";
    qsText+="\n email: etienne.deforas@gmail.com";

    mb.setText(qsText);
    mb.exec();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::resizeEvent( QResizeEvent *e )
{
    (void)e;

    QMainWindow::resizeEvent(e);

    //todo resize curves

}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::update_details()
{
    if(_pEngine==nullptr)
    {
        ui->peDetails->clear();
        return;
    }

    string s;
    _pDataSource->write(s);
    s+="\n";
    _pEngine->write(s);

    ui->peDetails->setPlainText(s.c_str());
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbEngine_currentTextChanged(const QString &arg1)
{
    delete _pEngine;
    _pEngine=nullptr;

    if(arg1=="BeeDNN")
        _pEngine=new MLEngineBeeDnn;

    _bMustSave=true;
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_btnTrainMore_clicked()
{
    if(QGuiApplication::keyboardModifiers() & Qt::ControlModifier)
    {
        for(int i=0;i<9;i++)
            train_and_test(false,true);
    }

    train_and_test(false,true);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::net_to_ui()
{
    model_changed(0);

    //was updating twNet

    updateTitle();
    drawRegression();
    update_details();
    update_classification_tab();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbYLogAxis_stateChanged(int arg1)
{
    (void)arg1;
    _qsLoss->setYLogAxis(ui->cbYLogAxis->isChecked());
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_buttonColor_clicked()
{
    QColorDialog qcd;
    qcd.setCurrentColor(_curveColor);
    qcd.exec();
    _curveColor=qcd.currentColor().rgb();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_2_clicked() //clear
{
    _qsLoss->clear();
    _qsAccuracy->clear();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::set_input_size(int iSize)
{
    _iInputSize=iSize;
    _pEngine->net().set_input_size(_iInputSize);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::update_classification_tab()
{
    if(!_pEngine->is_classification_problem())
    { // not a classification problem
        ui->twConfusionMatrixTrain->clearContents();
        ui->leTrainAccuracy->setText("n/a");
        ui->leTestAccuracy->setText("n/a");
        return;
    }

    float fAccuracy=0.f;
    if(_pDataSource->has_test_data())
    {
        _pEngine->compute_confusion_matrix(_pDataSource->test_data(),_pDataSource->test_truth(),_mConfusionMatrix,fAccuracy);
        ui->leTestAccuracy->setText(QString::number((double)fAccuracy,'f',2));
    }
    else
        ui->leTestAccuracy->setText("n/a");

    _pEngine->compute_confusion_matrix(_pDataSource->train_data(),_pDataSource->train_truth(),_mConfusionMatrix,fAccuracy);
    ui->leTrainAccuracy->setText(QString::number((double)fAccuracy,'f',2));

    drawConfusionMatrix(); //for now on train data
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::drawConfusionMatrix()
{
    if(_pEngine->is_classification_problem()==false)
    { // not a classification problem
        ui->twConfusionMatrixTrain->clearContents();
        ui->leTrainAccuracy->setText("n/a");
        ui->leTestAccuracy->setText("n/a");
        return;
    }

    ui->twConfusionMatrixTrain->setColumnCount((int)_mConfusionMatrix.cols());
    ui->twConfusionMatrixTrain->setRowCount((int)_mConfusionMatrix.rows());

    if(ui->cbConfMatPercent->isChecked())
    {
        MatrixFloat mConfMatPercent;
        ConfusionMatrix::toPercent(_mConfusionMatrix,mConfMatPercent);

        for(int c=0;c<_mConfusionMatrix.cols();c++)
            for(int r=0;r<_mConfusionMatrix.rows();r++)
                ui->twConfusionMatrixTrain->setItem(r,c,new QTableWidgetItem(QString::number((double)mConfMatPercent(r,c),'f',1)));
    }
    else
    {
        for(int c=0;c<_mConfusionMatrix.cols();c++)
            for(int r=0;r<_mConfusionMatrix.rows();r++)
                ui->twConfusionMatrixTrain->setItem(r,c,new QTableWidgetItem(to_string( (int)(_mConfusionMatrix(r,c))).data() ));
    }

    //colorize in yellow the diagonal
    for(int c=0;c<_mConfusionMatrix.cols();c++)
        ui->twConfusionMatrixTrain->item(c,c)->setBackgroundColor(Qt::yellow);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbConfMatPercent_stateChanged(int arg1)
{
    (void)arg1;
    drawConfusionMatrix();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionSave_as_triggered()
{
    string sFileName = QFileDialog::getSaveFileName(this,tr("Save DNNLab File as"), ".", tr("DNNLab Files (*.dnnlab)")).toStdString();
    if(sFileName.empty())
        return;

    _sFileName=sFileName;

    save();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionNew_triggered()
{
    if(ask_save())
    {
        _sFileName="";
        init_all();
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionOpen_triggered()
{
    if(ask_save())
    {
        string sFileName = QFileDialog::getOpenFileName(this,tr("Open DNNLab File"), ".", tr("DNNLab Files (*.dnnlab)")).toStdString();
        if(sFileName.empty())
            return;

        _sFileName=sFileName;

        load();
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionSave_triggered()
{
    if(_sFileName.empty())
    {
        string sFileName = QFileDialog::getSaveFileName(this,tr("Save DNNLab File"), ".", tr("DNNLab Files (*.dnnlab)")).toStdString();
        if(sFileName.empty())
            return;

        _sFileName=sFileName;
    }

    save();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionClose_triggered()
{
    if(ask_save())
    {
        _sFileName="";
        init_all();
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionSave_with_Score_triggered()
{
    if(_sFileName.empty())
    {
        string sFileName = QFileDialog::getSaveFileName(this,tr("Save DNNLab File"), ".", tr("DNNLab Files (*.dnnlab)")).toStdString();
        if(sFileName.empty())
            return;

        _sFileName=sFileName;
    }

    //score=to_string() todo

    save();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_3_clicked() //copy to clipboard
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(ui->peDetails->toPlainText());
}
//////////////////////////////////////////////////////////////////////////////
bool MainWindow::save()
{
    string s;
    _pDataSource->write(s);
    _pEngine->write(s);
    s+="\nNotes=\n"+_sNotes+"\n";

    ofstream out(_sFileName,ios::binary); //todo test

    out << s;

    _bMustSave=false;
    updateTitle();
    return true; //for now
}
//////////////////////////////////////////////////////////////////////////////
bool MainWindow::load()
{
    //todo use a file I/O class, properties?

    QApplication::setOverrideCursor(Qt::WaitCursor);

    init_all();

    string s;
    ifstream fIn(_sFileName);
    string line;
    while(std::getline(fIn, line))
    {
        //concat lines without "="
        if(line.find("=")!=string::npos)
            s+="\n"+line;
        else
            s+=" "+line; //todo keep \n int multilines string
    }

    _pEngine->read(s);
    _pDataSource->read(s);
    _sNotes=NetUtil::find_key(s,"Notes");

    model_changed(nullptr); //update everything

    net_to_ui();

    QApplication::restoreOverrideCursor();

    //show intersting results from net
    if(_pEngine->is_classification_problem())
        ui->tabWidget->setCurrentIndex(2);
    else
        ui->tabWidget->setCurrentIndex(1);

    return true; //for now
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::load_file(string sFile)
{
    _sFileName=sFile;
    load();
}
//////////////////////////////////////////////////////////////////////////////
bool MainWindow::ask_save()
{
    if(!_bMustSave)
        return true;

    QMessageBox msgBox;
    msgBox.setText("The network has been modified.");
    msgBox.setInformativeText("Do you want to save your changes?");
    msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    msgBox.setDefaultButton(QMessageBox::Save);

    switch (msgBox.exec())
    {
    case QMessageBox::Save:
        save();
        return true;
    case QMessageBox::Discard:
        _bMustSave=false;
        return true;
    case QMessageBox::Cancel:
        return false;
    default:
        return false;
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::updateTitle()
{
    string sTitle="DNNLab ";

    if(!_sFileName.empty())
        sTitle+=_sFileName;

    if(_bMustSave)
        sTitle+=" *";

    setWindowTitle(sTitle.c_str());
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_btnTestOnly_clicked()
{
    train_and_test(false,false);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionReload_triggered()
{
    if(ask_save())
        load();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::closeEvent(QCloseEvent *event)
{
    if(ask_save())
        event->accept();
    else
        event->ignore();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::model_changed(void * pSender)
{
    if(pSender != (void*)(ui->frameNotes ) )
    {
        ui->frameNotes->setText(_sNotes);
    }
    else
    {
        _sNotes=ui->frameNotes->text();
        _bMustSave=true;
    }

    if(pSender != (void*)(ui->frameGlobal ) )
    {
        ui->frameGlobal->set_data_name(_pDataSource->name());
        ui->frameGlobal->set_problem(_pEngine->is_classification_problem());
        ui->frameGlobal->set_engine_name("BeeDNN");
    }
    else
    {
        _pDataSource->load(ui->frameGlobal->data_name());
        _pEngine->set_problem(ui->frameGlobal->is_classification_problem());
        set_input_size(_pDataSource->data_size());
        _bMustSave=true;
    }

    if(pSender!=(void*)(ui->frameLearning))
    {
        ui->frameLearning->set_nettrain(&_pEngine->netTrain());
    }
    else
    {
        _bMustSave=true;
    }

    if(pSender != (void*)(ui->frameNetwork ) )
    {
        ui->frameNetwork->set_net(&(_pEngine->net()));
    }
    else
    {
        //net modified
        _bMustSave=true;
    }

    updateTitle();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_btnTrainAndTest_clicked()
{
    if(QGuiApplication::keyboardModifiers() & Qt::ControlModifier)
    {
        for(int i=0;i<9;i++)
            train_and_test(true,true);
    }

    train_and_test(true,true);
}
//////////////////////////////////////////////////////////////////////////////
