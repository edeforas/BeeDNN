TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN_NO"

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../NetTrainMomentum.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../ActivationLayer.cpp

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainMomentum.h \
    ../MatrixUtil.h \
    ../ActivationLayer.h
