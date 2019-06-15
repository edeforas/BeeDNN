#ifndef FRAMELEARNING_H
#define FRAMELEARNING_H

#include <QFrame>
#include <string>
using namespace std;

namespace Ui {
class FrameLearning;
}

class MainWindow;

class FrameLearning : public QFrame
{
    Q_OBJECT

public:
    explicit FrameLearning(QWidget *parent = 0);
    ~FrameLearning();

    void set_main_window(MainWindow* pMainWindow);

    void setOptimizer(string sOptimizer);
    string optimizer();

    void setLoss(string sLoss);
    string loss();

    void setKeepBest(bool bKeepBest);
    bool keepBest();

    void setEpochs(int iEpochs);
    int epochs();

    void setReboost(int iEpochs);
    int reboost();

    void setBatchSize(int iBatchSize);
    int batchSize();

    void setLearningRate(float fLearningRate);
    float learningRate();

    void setDecay(float fDecay);
    float decay();

    void setMomentum(float fMomentum);
    float momentum();

private slots:
    void on_btnTestOnly_clicked();

    void on_btnTrainAndTest_clicked();

    void on_btnTrainMore_clicked();

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
    Ui::FrameLearning *ui;
    MainWindow*_pMainWindow;
    bool _bLock;
};

#endif // FRAMELEARNING_H
