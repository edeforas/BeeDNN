#ifndef MainWindow_
#define MainWindow_

class MLEngine;

#include "Matrix.h"

#include <QMainWindow>

class SimpleCurveWidget;

namespace Ui {
class MainWindow;
}

class Net;
class DataSource;
class MLEngineBeeDnn;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    void model_changed(void * pSender);
	void load_file(string sFile);
	
protected:
    virtual void resizeEvent( QResizeEvent *e ) override;
    virtual void closeEvent(QCloseEvent *event) override;

private slots:
    void on_actionQuit_triggered();
    void on_actionAbout_triggered();
    void on_cbEngine_currentTextChanged(const QString &arg1);
    void on_btnTrainMore_clicked();
    void on_cbYLogAxis_stateChanged(int arg1);
    void on_buttonColor_clicked();
    void on_pushButton_2_clicked();
    void on_cbConfMatPercent_stateChanged(int arg1);
    void on_actionSave_as_triggered();
    void on_actionNew_triggered();
    void on_actionOpen_triggered();
    void on_actionSave_triggered();
    void on_actionClose_triggered();
    void on_actionSave_with_Score_triggered();
    void on_pushButton_3_clicked();
    void on_btnTestOnly_clicked();
    void on_actionReload_triggered();

    void on_btnTrainAndTest_clicked();

private:
    void init_all();
    bool ask_save(); //return true if saved/ready to overwrite

    void drawLoss(vector<float> vfLoss);
    void drawAccuracy(vector<float> vfAccuracy);
    void updateTitle();

    void drawRegression();
    void update_classification_tab();
    void drawConfusionMatrix();
    void train_and_test(bool bReset, bool bLearn);
    void update_details();
    void net_to_ui();
    void set_input_size(int iSize);

    bool save();
    bool load();

    Ui::MainWindow *ui;

    SimpleCurveWidget* _qsRegression;
    SimpleCurveWidget* _qsLoss;
    SimpleCurveWidget* _qsAccuracy;
    unsigned int _curveColor;
    int _iInputSize;

    bool _bMustSave;
    string _sFileName;

    //state data, need to be saved
    MLEngineBeeDnn* _pEngine;
    MatrixFloat _mConfusionMatrix;
    DataSource* _pDataSource;
    string _sNotes;
};

#endif
