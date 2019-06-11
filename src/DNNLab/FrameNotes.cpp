#include "FrameNotes.h"
#include "ui_FrameNotes.h"

#include "mainwindow.h"

FrameNotes::FrameNotes(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameNotes)
{
    ui->setupUi(this);
    _pMainWindow=nullptr;
}

FrameNotes::~FrameNotes()
{
    delete ui;
}


string FrameNotes::text()
{
    return ui->leNotes->toPlainText().toStdString();
}

void FrameNotes::setText(string sText)
{
      ui->leNotes->setPlainText(sText.c_str());
}

void FrameNotes::set_main_window(MainWindow* pMainWindow)
{
    _pMainWindow=pMainWindow;
}

void FrameNotes::on_leNotes_textChanged()
{
    _pMainWindow->model_changed(this);

    //todo
}
