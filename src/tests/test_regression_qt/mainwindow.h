#ifndef MAINWINDOW_H
#define MAINWINDOW_H

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

private slots:
    void on_pushButton_clicked();
    void on_actionQuit_triggered();

    void on_actionAbout_triggered();

private:
    void drawLoss(vector<double> vdLoss,vector<double> vdMaxError);
    void drawRegression(const Net& n);
    double compute_truth(double x);
    Ui::MainWindow *ui;

    ActivationManager _activ;
};

#endif // MAINWINDOW_H
