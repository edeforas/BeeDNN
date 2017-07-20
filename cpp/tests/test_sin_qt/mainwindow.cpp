#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include <QGraphicsPolygonItem>

#include "Net.h"
#include "DenseLayer.h"

//////////////////////////////////////////////////////////////////////////
// call back class to observe loss evolution
class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)
    {
        vdLoss.push_back(tr.maxError);
    }

    vector<double> vdLoss;
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
    ui->cbActivationLayer1->setCurrentText("Tanh");
    ui->cbActivationLayer2->setCurrentText("Tanh");
    ui->cbActivationLayer3->setCurrentText("Tanh");

    ui->gvLearningCurve->setScene(new QGraphicsScene);
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
    ui->gvLearningCurve->scene()->clear();

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

    //create ref sample
    Matrix mTruth(64);
    Matrix mSamples(64);
    for( int i=0;i<64;i++)
    {
        double x=(double)i/10.;
        mTruth(i)=sin(x);
        mSamples(i)=x;
    }

    TrainOption tOpt;
    tOpt.epochs=ui->leEpochs->text().toInt();
    tOpt.earlyAbortMaxError=ui->leEarlyAbortMaxError->text().toDouble();
    tOpt.learningRate=ui->leLearningRate->text().toDouble();;
    tOpt.batchSize=ui->leBatchSize->text().toInt();
    tOpt.momentum=ui->leMomentum->text().toDouble();
    tOpt.observer=&lossCB;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);

    ui->leMSE->setText(QString::number(0)); // todo
    ui->leMaxError->setText(QString::number(tr.maxError));
    ui->leComputedEpochs->setText(QString::number(tr.computedEpochs));

    drawLoss(lossCB.vdLoss);

    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<double> vdLoss)
{
    QGraphicsScene* qs= ui->gvLearningCurve->scene();

    QPolygonF poly;
    QPolygonF polyZero;

    for(int i=0;i<vdLoss.size();i++)
    {
        poly.append(QPointF(i,-vdLoss[i])); //up side down
        polyZero.append(QPointF(i,0));
    }

    QPen qp;
    qp.setCosmetic(true);

    qs->addPolygon(poly,qp);
    qs->addPolygon(polyZero,qp,Qt::black);

    ui->gvLearningCurve->fitInView(qs->itemsBoundingRect());
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
    QString qsText="Sin Net Demo";
    qsText+= "\n";
    qsText+= "\n GitHub: https://github.com/edeforas/test_DNN";
    qsText+= "\n by Etienne de Foras";
    qsText+="\n email: etienne.deforas@gmail.com";

    mb.setText(qsText);
    mb.exec();
}
//////////////////////////////////////////////////////////////////////////
