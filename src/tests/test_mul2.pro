TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN"

SOURCES += test_mul2.cpp \
    ../Net.cpp \
    ../NetTrainLearningRate.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseWithoutBias.cpp \
    ../LayerDenseWithBias.cpp

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseWithoutBias.h \
    ../LayerDenseWithBias.h
