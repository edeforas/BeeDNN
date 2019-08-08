#ifndef FrameNetwork_
#define FrameNetwork_

#include <QFrame>

namespace Ui {
class FrameNetwork;
}

class MainWindow;
class Net;
class QTableWidgetItem;

class FrameNetwork : public QFrame
{
    Q_OBJECT

public:
    explicit FrameNetwork(QWidget *parent = nullptr);
    ~FrameNetwork();

    void init();
    void set_main_window(MainWindow* pMainWindow);
    void set_net(Net* pNet);

private slots:
    void on_twNetwork_cellChanged(int row, int column);
    void type_changed();

    void on_btnNetworkInsert_clicked();
    void on_btnNetworkRemove_clicked();

private:
    Ui::FrameNetwork *ui;

    MainWindow*_pMainWindow;
    Net* _pNet;
    bool _bLock;
};

#endif
