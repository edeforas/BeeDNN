TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

#eigen use (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH+=$$(EIGEN_PATH)

INCLUDEPATH+=..

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../Optimizer.cpp \
    ../NetTrain.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../Optimizer.h \
    ../NetTrain.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h
