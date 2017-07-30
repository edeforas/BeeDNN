QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14

SOURCES += main.cpp\
        mainwindow.cpp \
    ../../Activation.cpp \
    ../../ActivationRelu.cpp \
    ../../ActivationSelu.cpp \
    ../../ActivationSigmoid.cpp \
    ../../ActivationTanh.cpp \
    ../../DenseLayer.cpp \
    ../../Layer.cpp \
    ../../Net.cpp \
    ../../ActivationLinear.cpp \
    ../../ActivationAtan.cpp \
    ../../ActivationElliot.cpp \
    ../../ActivationGauss.cpp \
    ../../ActivationSoftPlus.cpp \
    ../../ActivationSoftSign.cpp \
    ../../ActivationManager.cpp \
    SimpleCurve.cpp \
    ../../ActivationElu.cpp \
    ../../ActivationLeakyRelu.cpp

HEADERS  += mainwindow.h \
    ../../Activation.h \
    ../../ActivationRelu.h \
    ../../ActivationSelu.h \
    ../../ActivationSigmoid.h \
    ../../ActivationTanh.h \
    ../../DenseLayer.h \
    ../../Layer.h \
    ../../Matrix.h \
    ../../Net.h \
    ../../ActivationLinear.h \
    ../../ActivationAtan.h \
    ../../ActivationElliot.h \
    ../../ActivationGauss.h \
    ../../ActivationSoftPlus.h \
    ../../ActivationSoftSign.h \
    ../../ActivationManager.h \
    SimpleCurve.h \
    ../../ActivationElu.h \
    ../../ActivationLeakyRelu.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\..\
