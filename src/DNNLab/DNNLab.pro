QT       += core gui widgets
TARGET = DNNLab
TEMPLATE = app
CONFIG += c++17


#qwt (optional)
#DEFINES+= "USE_QWT"
#INCLUDEPATH += $$(QWT_PATH)/include


#eigen (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH += $$(EIGEN_PATH)


#tiny-dnn (optional)
DEFINES+= "USE_TINYDNN"
INCLUDEPATH += $$(TINY_DNN_PATH)
SOURCES += MLEngineTinyDnn.cpp
HEADERS +=MLEngineTinyDnn.h


#net library
INCLUDEPATH += ..
SOURCES += \
    ../Net.cpp \
    ../NetUtil.cpp \
    ../NetTrain.cpp \
    ../Optimizer.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../ConfusionMatrix.cpp \
    ../MNISTReader.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerGlobalGain.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../ConfusionMatrix.h \
    ../MNISTReader.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrain.h \
    ../Optimizer.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h \
    ../LayerGlobalGain.h

#ui
SOURCES += \
    main.cpp\
    mainwindow.cpp \
    SimpleCurveWidget.cpp \
    MLEngineBeeDnn.cpp \
    MLEngine.cpp
HEADERS  += \
    mainwindow.h \
    SimpleCurveWidget.h \
    MLEngine.h \
    MLEngineBeeDnn.h
FORMS    += mainwindow.ui

