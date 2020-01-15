#ifndef FrameNotes_
#define FrameNotes_

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

    void set_main_window(MainWindow* pMainWindow);
    string text() const;
    void setText(const string& sText);

private slots:
    void on_leNotes_textChanged();

private:
    bool _bLock;
    MainWindow* _pMainWindow;
    Ui::FrameNotes *ui;
};

#endif
