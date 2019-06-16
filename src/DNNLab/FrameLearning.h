#ifndef FrameLearning_
#define FrameLearning_

#include <QFrame>
#include <string>
using namespace std;

namespace Ui {
class FrameLearning;
}

class MainWindow;
class NetTrain;

class FrameLearning : public QFrame
{
    Q_OBJECT

public:
    explicit FrameLearning(QWidget *parent = 0);
    ~FrameLearning();

    void set_main_window(MainWindow* pMainWindow);
    void set_nettrain(NetTrain* pTrain);

private slots:
    void on_cbKeepBest_stateChanged(int arg1);
    void on_cbLossFunction_currentTextChanged(const QString &arg1);
    void on_cbOptimizer_currentTextChanged(const QString &arg1);
    void on_leLearningRate_textChanged(const QString &arg1);
    void on_leDecay_textChanged(const QString &arg1);
    void on_leMomentum_textChanged(const QString &arg1);
    void on_leEpochs_textChanged(const QString &arg1);
    void on_leBatchSize_textChanged(const QString &arg1);
    void on_leReboost_textChanged(const QString &arg1);

private:
    void update_optimizer();

    Ui::FrameLearning *ui;
    MainWindow*_pMainWindow;
    NetTrain* _pTrain;
    bool _bLock;
};

#endif
