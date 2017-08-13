QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14

SOURCES += main.cpp\
        mainwindow.cpp \
    ../../Activation.cpp \
    ../../DenseLayer.cpp \
    ../../Layer.cpp \
    ../../Net.cpp \
    SimpleCurve.cpp \
    ../../MNISTReader.cpp

HEADERS  += mainwindow.h \
    ../../Activation.h \
    ../../DenseLayer.h \
    ../../Layer.h \
    ../../Matrix.h \
    ../../Net.h \
    SimpleCurve.h \
    ../../MNISTReader.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\..\
