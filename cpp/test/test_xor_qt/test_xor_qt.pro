QT       += core gui widgets

TARGET = test_xor_qt
TEMPLATE = app

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
    ../../ActivationManager.cpp

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
    ../../ActivationManager.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\..\
