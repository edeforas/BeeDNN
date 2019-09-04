#include "mainwindow.h"
#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setStyle(QStyleFactory::create("fusion"));

    MainWindow w;
    w.showMaximized();

    if (argc==2)
    {
		w.load_file(argv[1]);
    }

    return a.exec();
}
