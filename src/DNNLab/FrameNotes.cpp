#include "FrameNotes.h"
#include "ui_FrameNotes.h"

#include "mainwindow.h"

FrameNotes::FrameNotes(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameNotes)
{
    _bLock=true;
    _pMainWindow=nullptr;
    ui->setupUi(this);
    _bLock=false;
}

FrameNotes::~FrameNotes()
{
    delete ui;
}

string FrameNotes::text() const
{
    return ui->leNotes->toPlainText().toStdString();
}

void FrameNotes::setText(const string &sText)
{
    _bLock=true;
    ui->leNotes->setPlainText(sText.c_str());
    _bLock=false;
}

void FrameNotes::set_main_window(MainWindow* pMainWindow)
{
    _pMainWindow=pMainWindow;
}

void FrameNotes::on_leNotes_textChanged()
{
    if(_bLock)
        return;

    _pMainWindow->model_changed(this);
}
