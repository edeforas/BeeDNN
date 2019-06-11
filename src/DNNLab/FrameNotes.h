#ifndef FRAMENOTES_H
#define FRAMENOTES_H

#include <QFrame>

#include <string>
using namespace std;

namespace Ui {
class FrameNotes;
}

class MainWindow;

class FrameNotes : public QFrame
{
    Q_OBJECT

public:
    explicit FrameNotes(QWidget *parent = nullptr);
    ~FrameNotes();

    void set_main_window(MainWindow* _pMainWindow);

    string text();
    void setText(string sText);

private slots:
    void on_leNotes_textChanged();

private:
    MainWindow* _pMainWindow;
    Ui::FrameNotes *ui;
};

#endif // FRAMENOTES_H
