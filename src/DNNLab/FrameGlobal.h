#ifndef FrameGlobal_
#define FrameGlobal_

#include <QFrame>

namespace Ui {
class FrameGlobal;
}

#include <string>
using namespace std;

class MainWindow;

class FrameGlobal : public QFrame
{
    Q_OBJECT

public:
    explicit FrameGlobal(QWidget *parent = nullptr);
    ~FrameGlobal();

    void init();
    void set_main_window(MainWindow* _pMainWindow);

    string data_name() const;
    void set_data_name(string sDataName);

    string engine_name() const;
    void set_engine_name(string sEngineName);

    string problem_name() const;
    void set_problem_name(string sProblemName);

private slots:
    void on_cbEngine_currentTextChanged(const QString &arg1);
    void on_cbProblem_currentTextChanged(const QString &arg1);
    void on_cbData_currentIndexChanged(int index);

private:
    MainWindow* _pMainWindow;
    bool _bLock;
    Ui::FrameGlobal *ui;
};

#endif
