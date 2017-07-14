#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "Net.h"
#include "ActivationSigmoid.h"
#include "DenseLayer.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    Net n;

    ActivationSigmoid ac;
    DenseLayer l1(2,3,ac);
    DenseLayer l2(3,1,ac);

    n.add(&l1);
    n.add(&l2);

    double dSamples[]={ 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1};
    double dTruths[]={ 0 , 1 , 1, 0 };

    const Matrix mSamples(dSamples,4,2);
    const Matrix mTruth(dTruths,4,1);

    TrainOption tOpt;
    tOpt.learningRate=ui->leLearningRate->text().toDouble();
    tOpt.batchSize=ui->leBatchSize->text().toInt();
    tOpt.momentum=ui->leMomentum->text().toDouble();
    tOpt.epochs=ui->leEpochs->text().toInt();

    TrainResult tr=n.train(mSamples,mTruth,tOpt); //todo
    (void)tr; //todo
    //cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " MaxEpoch=" << tr.maxEpoch << endl;

    Matrix m00,m01,m10,m11;

    n.forward(mSamples.row(0),m00);
    n.forward(mSamples.row(1),m01);
    n.forward(mSamples.row(2),m10);
    n.forward(mSamples.row(3),m11);

    ui->leXOR00->setText(QString::number(m00(0)));
    ui->leXOR10->setText(QString::number(m10(0)));
    ui->leXOR01->setText(QString::number(m01(0)));
    ui->leXOR11->setText(QString::number(m11(0)));
    double loss = 0; // todo net.get_loss<mse>(input_data, desired_out);
    ui->leMSE->setText(QString::number(loss));

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Close_clicked()
{
    close();
}
