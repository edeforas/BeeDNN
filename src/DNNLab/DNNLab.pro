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
    ../NetTrainSGD.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseNoBias.cpp \
    ../LayerDenseAndBias.cpp \
    ../LayerDropout.cpp
	HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h \
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

