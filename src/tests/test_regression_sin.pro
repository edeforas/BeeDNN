TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN_NO"

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

SOURCES += \
    ../Net.cpp \
    ../NetTrainMomentum.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../ActivationLayer.cpp \
    test_regression_sin.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainMomentum.h \
    ../MatrixUtil.h \
    ../ActivationLayer.h
