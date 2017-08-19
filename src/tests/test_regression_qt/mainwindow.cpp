#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include "SimpleCurve.h"

#include "Net.h"
#include "DenseLayer.h"

//////////////////////////////////////////////////////////////////////////
// callback class to observe loss evolution
class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)
    {
        vdLoss.push_back(tr.loss);
        vdMaxError.push_back(tr.maxError);
    }

    vector<double> vdLoss,vdMaxError;
};
//////////////////////////////////////////////////////////////////////////
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    vector<string> vsActivations;
    _activ.list_all(vsActivations);

    for(unsigned int i=0;i<vsActivations.size();i++)
    {
        ui->cbActivationLayer1->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer2->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer3->addItem(vsActivations[i].c_str());
    }
    ui->cbActivationLayer1->setCurrentText("Gauss");
    ui->cbActivationLayer2->setCurrentText("Gauss");
    ui->cbActivationLayer3->setCurrentText("Linear");

    ui->cbFunction->addItem("Sin");
    ui->cbFunction->addItem("Abs");
    ui->cbFunction->addItem("Parabolic");
    ui->cbFunction->addItem("Gamma");
    ui->cbFunction->addItem("Exp");
    ui->cbFunction->addItem("Sqrt");
    ui->cbFunction->addItem("Ln");
    ui->cbFunction->addItem("Gauss");
    ui->cbFunction->addItem("Rectangular");

    resizeDocks({ui->dockWidget},{1},Qt::Horizontal);
}
//////////////////////////////////////////////////////////////////////////
MainWindow::~MainWindow()
{
    delete ui;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    LossObserver lossCB;

    Net n;

    int iNbHiddenNeurons2=ui->sbNbNeurons2->value();
    int iNbHiddenNeurons3=ui->sbNbNeurons3->value();
    Activation* pActivLayer1=_activ.get_activation(ui->cbActivationLayer1->currentText().toStdString());
    Activation* pActivLayer2=_activ.get_activation(ui->cbActivationLayer2->currentText().toStdString());
    Activation* pActivLayer3=_activ.get_activation(ui->cbActivationLayer3->currentText().toStdString());
    DenseLayer l1(1,iNbHiddenNeurons2,pActivLayer1);
    DenseLayer l2(iNbHiddenNeurons2,iNbHiddenNeurons3,pActivLayer2);
    DenseLayer l3(iNbHiddenNeurons3,1,pActivLayer3);

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    int iNbPoint=ui->leNbPointsLearn->text().toInt();
    double dInputMin=ui->leInputMin->text().toDouble();
    double dInputMax=ui->leInputMax->text().toDouble();
    double dStep=(dInputMax-dInputMin)/(double)(iNbPoint-1);

    //create ref sample
    MatrixFloat mTruth(iNbPoint);
    MatrixFloat mSamples(iNbPoint);
    double dVal=dInputMin;

    for( int i=0;i<iNbPoint;i++)
    {
        mTruth(i)=compute_truth(dVal);
        mSamples(i)=dVal;
        dVal+=dStep;
    }

    TrainOption tOpt;
    tOpt.epochs=ui->leEpochs->text().toInt();
    tOpt.earlyAbortMaxError=ui->leEarlyAbortMaxError->text().toDouble();
    tOpt.earlyAbortMeanError=ui->leEarlyAbortMeanError->text().toDouble(); //same as loss?
    tOpt.learningRate=ui->leLearningRate->text().toDouble();;
    tOpt.batchSize=ui->leBatchSize->text().toInt();
    tOpt.momentum=ui->leMomentum->text().toDouble();
    tOpt.observer=&lossCB;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);

    ui->leMSE->setText(QString::number(tr.loss));
    ui->leMaxError->setText(QString::number(tr.maxError));
    ui->leComputedEpochs->setText(QString::number(tr.computedEpochs));

    drawLoss(lossCB.vdLoss,lossCB.vdMaxError);
    drawRegression(n);

    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<double> vdLoss,vector<double> vdMaxError)
{
    SimpleCurve* qs=new SimpleCurve;

    vector<double> x,loss,maxError;

    for(unsigned int i=0;i<vdLoss.size();i++)
    {
        x.push_back(i);
        loss.push_back(-vdLoss[i]*1000.); // //up side down *1000
        maxError.push_back(-vdMaxError[i]); //up side down
    }

    qs->addCurve(x,loss,Qt::red);
    qs->addCurve(x,maxError,Qt::black);

    QPen penBlack(Qt::black);
    penBlack.setCosmetic(true);
    qs->addLine(0,0,vdLoss.size()-1,0,penBlack);

    ui->gvLearningCurve->setScene(qs); //take ownership
    ui->gvLearningCurve->fitInView(qs->itemsBoundingRect());
    ui->gvLearningCurve->scale(0.8,0.8);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawRegression(const Net& n)
{
    SimpleCurve* qs=new SimpleCurve;

    //create ref sample hi-res and net output
    unsigned int iNbPoint=(unsigned int)(ui->leNbPointsTest->text().toInt());
    float dInputMin=ui->leInputMin->text().toDouble();
    float dInputMax=ui->leInputMax->text().toDouble();
    float dStep=(dInputMax-dInputMin)/(double)(iNbPoint-1);
    vector<double> vTruth(iNbPoint);
    vector<double> vSamples(iNbPoint);
    vector<double> vRegression(iNbPoint);
    MatrixFloat mIn(1),mOut;
    float dVal=dInputMin;

    for(unsigned int i=0;i<iNbPoint;i++)
    {
        mIn(0)=dVal;
        vTruth[i]=-compute_truth(dVal);
        vSamples[i]=dVal;
        n.forward(mIn,mOut);
        vRegression[i]=-mOut(0);
        dVal+=dStep;
    }

    qs->addCurve(vSamples,vTruth,Qt::red);
    qs->addCurve(vSamples,vRegression,Qt::blue);

    QPen penBlack(Qt::black);
    penBlack.setCosmetic(true);
    qs->addLine(dInputMin,0,dInputMax,0,penBlack);

    ui->gvRegression->setScene(qs); //take ownership
    ui->gvRegression->fitInView(qs->itemsBoundingRect());
    ui->gvRegression->scale(0.8,0.8);
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
    QString qsText="Regression Net Demo";
    qsText+= "\n";
    qsText+= "\n GitHub: https://github.com/edeforas/test_DNN";
    qsText+= "\n by Etienne de Foras";
    qsText+="\n email: etienne.deforas@gmail.com";

    mb.setText(qsText);
    mb.exec();
}
//////////////////////////////////////////////////////////////////////////
double MainWindow::compute_truth(double x)
{
    //function not optimized but not mandatory

    string sFunction=ui->cbFunction->currentText().toStdString();

    if(sFunction=="Sin")
        return sin(x);

    if(sFunction=="Abs")
        return fabs(x);

    if(sFunction=="Parabolic")
        return x*x;

    if(sFunction=="Gamma")
        return tgamma(x);

    if(sFunction=="Exp")
        return exp(x);

    if(sFunction=="Sqrt")
        return sqrt(x);

    if(sFunction=="Ln")
        return log(x);

    if(sFunction=="Gauss")
        return exp(-x*x);

    if(sFunction=="Rectangular")
        return ((((int)x)+(x<0.))+1) & 1 ;

    return 0.;
}
//////////////////////////////////////////////////////////////////////////

