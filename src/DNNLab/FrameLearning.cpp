#include "FrameLearning.h"
#include "ui_FrameLearning.h"

#include "mainwindow.h"
#include "Loss.h"
#include "Optimizer.h"

//////////////////////////////////////////////////////////////
FrameLearning::FrameLearning(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameLearning)
{
    _pMainWindow=nullptr;
_pTrain=nullptr;
    _bLock=true;
    ui->setupUi(this);

    //setup loss
    vector<string> vsloss;
    list_loss_available(vsloss);
    for(unsigned int i=0;i<vsloss.size();i++)
        ui->cbLossFunction->addItem(vsloss[i].data());

    //setup optimizer
    vector<string> vsOptimizers;
    list_optimizers_available( vsOptimizers);
    for(unsigned int i=0;i<vsOptimizers.size();i++)
        ui->cbOptimizer->addItem(vsOptimizers[i].data());

    //default init
    ui->leBatchSize->setText("16");
    ui->leLearningRate->setText("-1");
    ui->leDecay->setText("-1");
    ui->leMomentum->setText("-1");
    ui->leReboost->setText("-1");
    ui->cbKeepBest->setChecked(true);
    ui->cbOptimizer->setCurrentText("Adam");
    ui->cbLossFunction->setCurrentIndex(0);

    _bLock=false;
}
//////////////////////////////////////////////////////////////
FrameLearning::~FrameLearning()
{
    delete ui;
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_cbKeepBest_stateChanged(int arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setKeepBest(bool bKeepBest)
{
    _bLock=true;
    ui->cbKeepBest->setChecked(bKeepBest);
    _bLock=false;
}
bool FrameLearning::keepBest()
{
    return ui->cbKeepBest->isChecked();
}
////////////////////////////////////////////////////////////////////
void FrameLearning::on_cbLossFunction_currentTextChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setLoss(string sLoss)
{
    _bLock=true;
    ui->cbLossFunction->setCurrentText(sLoss.c_str());
    _bLock=false;
}
string FrameLearning::loss()
{
    return ui->cbLossFunction->currentText().toStdString();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_cbOptimizer_currentTextChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setOptimizer(string sOptimizer)
{
    _bLock=true;
    ui->cbOptimizer->setCurrentText(sOptimizer.c_str());
    _bLock=false;
}
string FrameLearning::optimizer()
{
    return ui->cbOptimizer->currentText().toStdString();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leLearningRate_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setLearningRate(float fLearningRate)
{
    _bLock=true;
    ui->leLearningRate->setText(to_string(fLearningRate).c_str());
    _bLock=false;
}
float FrameLearning::learningRate()
{
    return ui->leLearningRate->text().toFloat();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leDecay_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setDecay(float fDecay)
{
    _bLock=true;
    ui->leDecay->setText(to_string(fDecay).c_str());
    _bLock=false;
}
float FrameLearning::decay()
{
    return ui->leDecay->text().toFloat();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leMomentum_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setMomentum(float fMomentum)
{
    _bLock=true;
    ui->leMomentum->setText(to_string(fMomentum).c_str());
    _bLock=false;
}
float FrameLearning::momentum()
{
    return ui->leMomentum->text().toFloat();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leEpochs_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setEpochs(int iEpochs)
{
    _bLock=true;
    ui->leEpochs->setText(to_string(iEpochs).c_str());
    _bLock=false;
}
int FrameLearning::epochs()
{
    return ui->leEpochs->text().toInt();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leBatchSize_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setBatchSize(int iBatchSize)
{
    _bLock=true;
    ui->leBatchSize->setText(to_string(iBatchSize).c_str());
    _bLock=false;
}
int FrameLearning::batchSize()
{
    return ui->leBatchSize->text().toInt();
}
//////////////////////////////////////////////////////////////
void FrameLearning::on_leReboost_textChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
void FrameLearning::setReboost(int iReboost)
{
    _bLock=true;
    ui->leReboost->setText(to_string(iReboost).c_str());
    _bLock=false;
}
int FrameLearning::reboost()
{
    return ui->leReboost->text().toInt();
}
//////////////////////////////////////////////////////////////
void FrameLearning::set_main_window(MainWindow* pMainWindow)
{
    _pMainWindow=pMainWindow;
}
//////////////////////////////////////////////////////////////
    void FrameLearning::set_nettrain(NetTrain* pTrain)
    {
        _pTrain=pTrain;
    }


    //////////////////////////////////////////////////////////////
