QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14

DEFINES+= "USE_EIGEN_NO"

SOURCES += \
	main.cpp\
	mainwindow.cpp \
    SimpleCurve.cpp \
    ../DNNEngineTestDnn.cpp \
    ../DNNEngineTinyDnn.cpp \
    ../DNNEngine.cpp

SOURCES += \
    ../Net.cpp \
    ../NetUtil.cpp \
    ../NetTrainLearningRate.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseWithoutBias.cpp \
    ../LayerDenseWithBias.cpp



HEADERS  += \
	mainwindow.h \
    SimpleCurve.h \
    ../DNNEngine.h \
    ../DNNEngineTestDnn.h \
    ../DNNEngineTinyDnn.h


HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseWithoutBias.h \
    ../LayerDenseWithBias.h




FORMS    += mainwindow.ui

INCLUDEPATH += ..
INCLUDEPATH += $$(EIGEN_PATH)

#tiny-dnn
INCLUDEPATH += $$(TINY_DNN_PATH)
