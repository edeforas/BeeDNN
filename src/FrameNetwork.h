#ifndef FRAMENETWORK_H
#define FRAMENETWORK_H

#include <QFrame>

namespace Ui {
class FrameNetwork;
}

class FrameNetwork : public QFrame
{
    Q_OBJECT

public:
    explicit FrameNetwork(QWidget *parent = 0);
    ~FrameNetwork();

private:
    Ui::FrameNetwork *ui;
};

#endif // FRAMENETWORK_H
