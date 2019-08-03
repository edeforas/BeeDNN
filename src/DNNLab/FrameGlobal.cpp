#include "FrameGlobal.h"
#include "ui_FrameGlobal.h"

#include "mainwindow.h"
#include <QFileDialog>

FrameGlobal::FrameGlobal(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameGlobal)
{
    _bLock=true;

    _pMainWindow=nullptr;

    ui->setupUi(this);

    ui->cbData->addItem("Identity");
    ui->cbData->addItem("File...");
    ui->cbData->addItem("And");
    ui->cbData->addItem("Xor");
    ui->cbData->addItem("MNIST");

    ui->cbData->insertSeparator(5);

    ui->cbData->addItem("Sin");
	ui->cbData->addItem("Sin4Period");
    ui->cbData->addItem("Abs");
    ui->cbData->addItem("Parabolic");
    ui->cbData->addItem("Exp");
    ui->cbData->addItem("Gauss");
    ui->cbData->addItem("Rectangular");

    init();

    _bLock=false;
}

FrameGlobal::~FrameGlobal()
{
    delete ui;
}

void FrameGlobal::init()
{
    ui->cbData->setCurrentIndex(6);
    ui->cbEngine->setCurrentIndex(0);
    ui->cbProblem->setCurrentIndex(0);
}

void FrameGlobal::set_main_window(MainWindow* pMainWindow)
{
    _pMainWindow=pMainWindow;
}

void FrameGlobal::on_cbEngine_currentTextChanged(const QString &arg1)
{
    (void)arg1;

    if(_bLock)
        return;
    _pMainWindow->model_changed(this);
}

void FrameGlobal::on_cbProblem_currentTextChanged(const QString &arg1)
{
    (void)arg1;
    if(_bLock)
        return;
    _pMainWindow->model_changed(this);
}

string FrameGlobal::data_name() const
{
    return ui->cbData->currentText().toStdString();
}

void FrameGlobal::set_data_name(string sDataName)
{
    _bLock=true;
    ui->cbData->setCurrentText(sDataName.c_str());
    ui->cbData->setToolTip(sDataName.c_str());
    _bLock=false;
}

string FrameGlobal::engine_name() const
{
    return ui->cbEngine->currentText().toStdString();
}

void FrameGlobal::set_engine_name(string sEngineName)
{
    _bLock=true;
    ui->cbEngine->setCurrentText(sEngineName.c_str());
    _bLock=false;
}

bool FrameGlobal::is_classification_problem() const
{
    return ui->cbProblem->currentText().toStdString()=="Classification";
}

void FrameGlobal::set_problem(bool bClassificationProblem)
{
    _bLock=true;
	string sProblem = bClassificationProblem ? "Classification" : "Regression";

    ui->cbProblem->setCurrentText(sProblem.c_str());
    _bLock=false;
}

void FrameGlobal::on_cbData_currentIndexChanged(int index)
{
    if(_bLock)
        return;

    if(index==1)
    {
        //custom file
        string sFileName = QFileDialog::getOpenFileName(this,tr("Open data, truth, test or train file"), ".", tr("All files (*.*)")).toStdString();
        if(sFileName.empty())
            return;

        ui->cbData->insertItem(2,sFileName.c_str());
        ui->cbData->setCurrentIndex(2);
    }

    _pMainWindow->model_changed(this);
}

