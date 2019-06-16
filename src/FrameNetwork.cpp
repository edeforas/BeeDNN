#include "FrameNetwork.h"
#include "ui_FrameNetwork.h"

FrameNetwork::FrameNetwork(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::FrameNetwork)
{
    ui->setupUi(this);
}

FrameNetwork::~FrameNetwork()
{
    delete ui;
}
