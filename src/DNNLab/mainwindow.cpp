#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QColorDialog>
#include "SimpleCurveWidget.h"

#include "MLEngine.h"
#include "MLEngineBeeDnn.h"
#include "MNISTReader.h"

#ifdef USE_TINYDNN
#include "MLEngineTinyDnn.h"
#endif

#include "Activation.h"
#include "Optimizer.h"
#include "ConfusionMatrix.h"

//////////////////////////////////////////////////////////////////////////
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    vector<string> vsActivations;
    list_activations_available( vsActivations);

    ui->cbFunction->addItem("And");
    ui->cbFunction->addItem("Xor");
    ui->cbFunction->addItem("MNIST");
    ui->cbFunction->addItem("TextFiles");

    ui->cbFunction->insertSeparator(4);

    ui->cbFunction->addItem("Identity");
    ui->cbFunction->addItem("Sin");
    ui->cbFunction->addItem("Abs");
    ui->cbFunction->addItem("Parabolic");
    ui->cbFunction->addItem("Gamma");
    ui->cbFunction->addItem("Exp");
    ui->cbFunction->addItem("Sqrt");
    ui->cbFunction->addItem("Ln");
    ui->cbFunction->addItem("Gauss");
    ui->cbFunction->addItem("Inverse");
    ui->cbFunction->addItem("Rectangular");

    ui->cbFunction->setCurrentIndex(5);

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
        // qcbType->addItem("SoftMax");

        qcbType->insertSeparator(4);

        for(unsigned int a=0;a<vsActivations.size();a++)
            qcbType->addItem(vsActivations[a].c_str());

        ui->twNetwork->setCellWidget(i,0,qcbType);
    }

    ui->twNetwork->setItem(0,1,new QTableWidgetItem("1")); //first input size is 1
    ui->twNetwork->adjustSize();

#ifdef USE_TINYDNN
    ui->cbEngine->addItem("tiny-dnn");
#endif

    vector<string> vsOptimizers;
    list_optimizers_available( vsOptimizers);
    for(unsigned int i=0;i<vsOptimizers.size();i++)
        ui->cbOptimizer->addItem(vsOptimizers[i].data());

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

    _curveColor=0xff0000; //red

    _bHasTestData=false;
    set_input_size(1);
    _pEngine=new MLEngineBeeDnn;
}
//////////////////////////////////////////////////////////////////////////
MainWindow::~MainWindow()
{
    delete ui;
    delete _pEngine;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_clicked()
{
    train_and_test(true);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::train_and_test(bool bReset)
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    compute_truth();

    //LossObserver lossCB;

    if(bReset)
        parse_net();

    DNNTrainOption dto;
    dto.epochs=ui->leEpochs->text().toInt();
    dto.learningRate=ui->leLearningRate->text().toFloat();
    dto.batchSize=ui->leBatchSize->text().toInt();
    dto.keepBest=ui->cbKeepBest->isChecked();
    dto.decay=ui->leDecay->text().toFloat();
    dto.momentum=ui->leMomentum->text().toFloat();
    dto.optimizer=ui->cbOptimizer->currentText().toStdString();
    dto.lossFunction=ui->cbLossFunction->currentText().toStdString();
    dto.testEveryEpochs=ui->sbTestEveryEpochs->value();
    //dto.observer=nullptr;//&lossCB;

    if(bReset)
        _pEngine->init();

    _pEngine->set_problem(ui->cbProblem->currentText()=="Classification");
    DNNTrainResult dtr =_pEngine->learn(_mTrainData,_mTrainTruth,dto);

    float fLoss=_pEngine->compute_loss(_mTrainData,_mTrainTruth); //final loss
    ui->leMSE->setText(QString::number(fLoss));
    ui->leComputedEpochs->setText(QString::number(dtr.computedEpochs));
    ui->leTimeByEpoch->setText(QString::number(dtr.epochDuration));

    drawLoss(dtr.loss);
    drawAccuracy(dtr.accuracy);
    drawRegression();
    update_classification_tab();
    update_details();

    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<double> vdLoss)
{
    if(!ui->cbHoldOn->isChecked())
        _qsLoss->clear();

    vector<double> x;
    for(unsigned int i=0;i<vdLoss.size();i++)
        x.push_back(i);

    _qsLoss->addHorizontalLine(0.);
    _qsLoss->addCurve(x,vdLoss,_curveColor);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawAccuracy(vector<double> vdAccuracy)
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
        vector<double> x;
        for(unsigned int i=0;i<vdAccuracy.size();i++)
            x.push_back(i);

        _qsAccuracy->addCurve(x,vdAccuracy,_curveColor);
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
    {   //not a regression problem
        return;
    }

    //create ref sample hi-res and net output
    unsigned int iNbPoint=(unsigned int)(ui->leNbPointsTest->text().toInt());
    float fInputMin=ui->leInputMin->text().toFloat();
    float fInputMax=ui->leInputMax->text().toFloat();
    bool bExtrapole=ui->cbExtrapole->isChecked();
    vector<double> vTruth;
    vector<double> vSamples;
    vector<double> vRegression;
    MatrixFloat mIn(1,1),mOut;

    compute_truth();

    if(bExtrapole)
    {
        float fBorder=(fInputMax-fInputMin)/2.f;
        fInputMin-=fBorder;
        fInputMax+=fBorder;
        iNbPoint*=2;
    }

    float fVal=fInputMin;
    float fStep=(fInputMax-fInputMin)/(iNbPoint-1.f);

    for(unsigned int i=0;i<iNbPoint;i++)
    {
        mIn(0,0)=fVal;
        vTruth.push_back((double)(_mTrainTruth(i,0)));
        vSamples.push_back((double)(fVal));
        _pEngine->predict(mIn,mOut);

        if(mOut.size()==0)
            return; //todo

        vRegression.push_back((double)(mOut(0)));
        fVal+=fStep;
    }

    _qsRegression->addCurve(vSamples,vTruth,0xFF0000);
    _qsRegression->addCurve(vSamples,vRegression,0xFF);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionQuit_triggered()
{
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
void MainWindow::compute_truth()
{
    //function not optimized but not mandatory for now

    string sFunction=ui->cbFunction->currentText().toStdString();
    _bHasTestData=false;

    if(sFunction=="And")
    {
        _mTrainData.resize(4,2);
        _mTrainData(0,0)=0; _mTrainData(0,1)=0;
        _mTrainData(1,0)=1; _mTrainData(1,1)=0;
        _mTrainData(2,0)=0; _mTrainData(2,1)=1;
        _mTrainData(3,0)=1; _mTrainData(3,1)=1;

        _mTrainTruth.resize(4,1);
        _mTrainTruth(0,0)=0;
        _mTrainTruth(1,0)=0;
        _mTrainTruth(2,0)=0;
        _mTrainTruth(3,0)=1;

        _mTestData=_mTrainData;
        _mTestTruth=_mTrainTruth;
        set_input_size(2);
        return;
    }

    if(sFunction=="Xor")
    {
        _mTrainData.resize(4,2);
        _mTrainData(0,0)=0; _mTrainData(0,1)=0;
        _mTrainData(1,0)=1; _mTrainData(1,1)=0;
        _mTrainData(2,0)=0; _mTrainData(2,1)=1;
        _mTrainData(3,0)=1; _mTrainData(3,1)=1;

        _mTrainTruth.resize(4,1);
        _mTrainTruth(0,0)=0;
        _mTrainTruth(1,0)=1;
        _mTrainTruth(2,0)=1;
        _mTrainTruth(3,0)=0;

        _mTestData=_mTrainData;
        _mTestTruth=_mTrainTruth;
        set_input_size(2);
        return;
    }

    if(sFunction=="MNIST")
    {
        if( (_mTrainData.cols()!=784) || (_mTrainData.rows()!=60000))
        {
            MNISTReader r;
            r.read_from_folder(".",_mTrainData,_mTrainTruth,_mTestData,_mTestTruth);
            _mTrainData/=256.f;
            _mTestData/=256.f;
        }

        _bHasTestData=true;
        set_input_size(_mTrainData.cols());
        return;
    }

    if(sFunction=="TextFiles")
    {
        _mTrainData=fromFile("train_data.txt");
        _mTrainTruth=fromFile("train_truth.txt");
        _mTrainData/=256.f;

        _mTestData=fromFile("test_data.txt");
        _mTestTruth=fromFile("test_truth.txt");
        _mTestData/=256.f;

        _bHasTestData=_mTestData.size()!=0;
        if(!_bHasTestData) //if invalid/noexistent test_data, use train_data
        {
            _mTestData=_mTrainData;
            _mTestTruth=_mTrainTruth;
        }

        set_input_size((int)_mTrainData.cols());
        return;
    }

    //simple function to interpolate
    int iNbPoint=ui->leNbPointsLearn->text().toInt();
    float dInputMin=ui->leInputMin->text().toFloat();
    float dInputMax=ui->leInputMax->text().toFloat();
    float dStep=(dInputMax-dInputMin)/(iNbPoint-1.f);

    _mTrainTruth.resize(iNbPoint,1);
    _mTrainData.resize(iNbPoint,1);
    float dVal=dInputMin,dOut=0.f;

    for( int i=0;i<iNbPoint;i++)
    {
        if(sFunction=="Identity")
            dOut=dVal;

        if(sFunction=="Sin")
            dOut=sinf(dVal);

        if(sFunction=="Abs")
            dOut=fabs(dVal);

        if(sFunction=="Parabolic")
            dOut=dVal*dVal;

        if(sFunction=="Gamma")
            dOut=tgammaf(dVal);

        if(sFunction=="Exp")
            dOut=expf(dVal);

        if(sFunction=="Sqrt")
            dOut=sqrtf(dVal);

        if(sFunction=="Ln")
            dOut=logf(dVal);

        if(sFunction=="Gauss")
            dOut=expf(-dVal*dVal);

        if(sFunction=="Inverse")
            dOut=1.f/dVal;

        if(sFunction=="Rectangular")
            dOut= ((((int)dVal)+(dVal<0.f))+1) & 1 ;

        _mTrainData(i,0)=dVal;
        _mTrainTruth(i,0)=dOut;
        dVal+=dStep;
    }
    _mTestData=_mTrainData; //for now
    _mTestTruth=_mTrainTruth; //for now
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::resizeEvent( QResizeEvent *e )
{
    (void)e;

    QMainWindow::resizeEvent(e);

    //toto resize curves

}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::update_details()
{
    if(_pEngine==nullptr)
    {
        ui->peDetails->clear();
        return;
    }

    ui->peDetails->setPlainText(_pEngine->to_string().c_str());
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbEngine_currentTextChanged(const QString &arg1)
{
    delete _pEngine;
    _pEngine=nullptr;

    if(arg1=="BeeDNN")
        _pEngine=new MLEngineBeeDnn;

#ifdef USE_TINYDNN
    if(arg1=="tiny-dnn")
        _pEngine=new DNNEngineTinyDnn;
#endif
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_btnTrainMore_clicked()
{
    train_and_test(false);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::parse_net()
{
    _pEngine->clear();
    int iLastOut=_iInputSize;
    for(int iRow=0;iRow<10;iRow++) //todo dynamic size
    {
        QComboBox* pCombo=(QComboBox*)(ui->twNetwork->cellWidget(iRow,0));
        if(!pCombo)
            continue;
        string sType=pCombo->currentText().toStdString();

        QTableWidgetItem* pwiInSize=ui->twNetwork->item(iRow,1); //todo not used in activation
        int iInSize;
        if(!pwiInSize)
            iInSize=iLastOut; //use last out
        else
            iInSize=pwiInSize->text().toInt();

        QTableWidgetItem* pwiOutSize=ui->twNetwork->item(iRow,2); //todo not used in activation
        int iOutSize;
        if(!pwiOutSize)
        {
            iOutSize=iInSize; //same size (i.e. activation case)
        }
        else
            iOutSize=pwiOutSize->text().toInt();

        iLastOut=iOutSize;

        if(!sType.empty())
        {
            if(sType=="Dropout")
            {
                float fRatio=0.2f; //by default

                QTableWidgetItem* pwArg1=ui->twNetwork->item(iRow,3);
                if(pwArg1)
                    fRatio=pwArg1->text().toFloat();

                _pEngine->add_dropout_layer(iInSize,fRatio);
            }
            else if(sType=="DenseAndBias")
                _pEngine->add_dense_layer(iInSize,iOutSize,true);
            else if(sType=="DenseNoBias")
                _pEngine->add_dense_layer(iInSize,iOutSize,false);
            else
                _pEngine->add_activation_layer(sType);
        }
    }

    _pEngine->init();
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
void MainWindow::on_pushButton_2_clicked()
{
    _qsLoss->clear();
    _qsAccuracy->clear();
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::set_input_size(int iSize)
{
    _iInputSize=iSize;
    //ui->twNetwork->item(0,1)->setText(to_string(_iInputSize).data());
    ui->twNetwork->setItem(0,1,new QTableWidgetItem(to_string(_iInputSize).data())); //first input size is 1
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
    if(_bHasTestData)
    {
        _pEngine->compute_confusion_matrix(_mTestData,_mTestTruth,_mConfusionMatrix,fAccuracy);
        ui->leTestAccuracy->setText(QString::number(fAccuracy,'f',2));
    }
    else
        ui->leTestAccuracy->setText("n/a");

    _pEngine->compute_confusion_matrix(_mTrainData,_mTrainTruth,_mConfusionMatrix,fAccuracy);
    ui->leTrainAccuracy->setText(QString::number(fAccuracy,'f',2));

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
void MainWindow::on_cbFunction_currentIndexChanged(int index)
{
    (void)index;
    string sFunction=ui->cbFunction->currentText().toStdString();

    if(sFunction=="And")
    {
        set_input_size(2);
        return;
    }

    if(sFunction=="Xor")
    {
        set_input_size(2);
        return;
    }

    if(sFunction=="MNIST")
    {
        set_input_size(784);
        return;
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbConfMatPercent_stateChanged(int arg1)
{
    (void)arg1;
    drawConfusionMatrix();
}
//////////////////////////////////////////////////////////////////////////////
