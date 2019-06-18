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
    explicit FrameLearning(QWidget *parent = nullptr);
    ~FrameLearning();

    void set_main_window(MainWindow* pMainWindow);
    void set_nettrain(NetTrain* pTrain);

private slots:
    void on_cbKeepBest_stateChanged(int arg1);
    void on_cbLossFunction_currentTextChanged(const QString &arg1);
    void on_cbOptimizer_currentTextChanged(const QString &arg1);
    void on_leLearningRate_editingFinished();
    void on_leReboost_editingFinished();
    void on_leBatchSize_editingFinished();
    void on_leEpochs_editingFinished();
    void on_leMomentum_editingFinished();
    void on_leDecay_editingFinished();

private:
    void update_optimizer();

    Ui::FrameLearning *ui;
    MainWindow*_pMainWindow;
    NetTrain* _pTrain;
    bool _bLock;
};

#endif
