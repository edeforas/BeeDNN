TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

#eigen use (optionnal)
DEFINES+= "USE_EIGEN"
INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)


SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../NetTrain.cpp \
    ../NetTrainLearningRate.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseAndBias.cpp \
    ../LayerDenseNoBias.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseAndBias.h \
    ../LayerDenseNoBias.h
