#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.showMaximized();

    if (argc==2)
    {
		w.load_file(argv[1]);
    }

    return a.exec();
}
