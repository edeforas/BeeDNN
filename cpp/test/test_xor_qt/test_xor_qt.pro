QT       += core gui

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
    ../../Net.cpp

HEADERS  += mainwindow.h \
    ../../Activation.h \
    ../../ActivationRelu.h \
    ../../ActivationSelu.h \
    ../../ActivationSigmoid.h \
    ../../ActivationTanh.h \
    ../../DenseLayer.h \
    ../../Layer.h \
    ../../Matrix.h \
    ../../Net.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\..\
