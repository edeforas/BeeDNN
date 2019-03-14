#ifndef MAINWINDOW_H
#define MAINWINDOW_H

class DNNEngine;


#include "Matrix.h"

#include <QMainWindow>

class SimpleCurveWidget;

namespace Ui {
class MainWindow;
}

class Net;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    virtual void resizeEvent( QResizeEvent *e );

private slots:
    void on_pushButton_clicked();
    void on_actionQuit_triggered();
    void on_actionAbout_triggered();
    void on_cbEngine_currentTextChanged(const QString &arg1);
    void on_btnTrainMore_clicked();   
    void on_cbYLogAxis_stateChanged(int arg1);

    void on_buttonColor_clicked();

    void on_pushButton_2_clicked();



    void on_cbFunction_currentIndexChanged(int index);

    void on_cbConfMatPercent_stateChanged(int arg1);

private:
    void drawLoss(vector<double> vdLoss);
    void drawRegression();
    void update_classification_tab();
    void drawConfusionMatrix();
    void compute_truth();
    void train_and_test(bool bReset);
    void update_details();
    void parse_net();
    void set_input_size(int iSize);

    Ui::MainWindow *ui;

    DNNEngine* _pEngine;
    SimpleCurveWidget* _qsRegression;
    SimpleCurveWidget* _qsLoss;
    unsigned int _curveColor;
    int _iInputSize;

    MatrixFloat _mInputData;
    MatrixFloat _mTruth;

    MatrixFloat _mTestInputData;
    MatrixFloat _mTestTruth;
    MatrixFloat _mConfusionMatrix;
};

#endif // MAINWINDOW_H
