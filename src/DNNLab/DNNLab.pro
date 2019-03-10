QT       += core gui widgets
TARGET = DNNLab
TEMPLATE = app
CONFIG += c++14


#qwt (optional)
#DEFINES+= "USE_QWT"
#INCLUDEPATH += $$(QWT_PATH)/include


#eigen (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH += $$(EIGEN_PATH)


#tiny-dnn (optional)
DEFINES+= "USE_TINYDNN"
INCLUDEPATH += $$(TINY_DNN_PATH)
SOURCES += DNNEngineTinyDnn.cpp
HEADERS +=DNNEngineTinyDnn.h


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
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../ConfusionMatrix.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrain.h \
    ../Optimizer.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h


#ui
SOURCES += \
    main.cpp\
    mainwindow.cpp \
    SimpleCurveWidget.cpp \
    DNNEngineTestDnn.cpp \
    DNNEngine.cpp
HEADERS  += \
    mainwindow.h \
    SimpleCurveWidget.h \
    DNNEngine.h \
    DNNEngineTestDnn.h
FORMS    += mainwindow.ui

