QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14

SOURCES += main.cpp\
        mainwindow.cpp \
    ../../Activation.cpp \
    ../../Layer.cpp \
    ../../Net.cpp \
    SimpleCurve.cpp \
    ../../MatrixUtil.cpp \
    ../../MNISTReader.cpp \
    ../../ActivationLayer.cpp \
    ../../NNEngineTestDnn.cpp \
    ../../NNEngineTinyDnn.cpp \
    ../../NNEngine.cpp

HEADERS  += mainwindow.h \
    ../../Activation.h \
    ../../Layer.h \
    ../../Matrix.h \
    ../../MatrixUtil.h \
    ../../Net.h \
    SimpleCurve.h \
    ../../MNISTReader.h \
    ../../ActivationLayer.h \
    ../../NNEngine.h \
    ../../NNEngineTestDnn.h \
    ../../NNEngineTinyDnn.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\..\

#TinyDNN
INCLUDEPATH += $$(TINY_DNN_PATH)