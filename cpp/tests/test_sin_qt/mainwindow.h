#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ActivationManager.h"

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

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
    Ui::MainWindow *ui;

    ActivationManager _activ;
};

#endif // MAINWINDOW_H
