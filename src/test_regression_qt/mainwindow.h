#ifndef MAINWINDOW_H
#define MAINWINDOW_H

class DNNEngine;

#include "Activation.h"

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class Net;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    virtual void resizeEvent( QResizeEvent *e );

private slots:
    void on_pushButton_clicked();
    void on_actionQuit_triggered();
    void on_actionAbout_triggered();
    void on_cbEngine_currentTextChanged(const QString &arg1);
    void on_btnTrainMore_clicked();

private:
    void drawLoss(vector<double> vdLoss,vector<double> vdMaxError);
    void drawRegression();
    double compute_truth(double x);
    void train_and_test(bool bReset);
    void update_details();

    Ui::MainWindow *ui;

    DNNEngine* _pEngine;
};

#endif // MAINWINDOW_H
