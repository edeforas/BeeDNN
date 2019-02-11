QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14


#eigen (optional)
DEFINES+= "USE_EIGEN_NO"
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
    ../NetTrainLearningRate.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseNoBias.cpp \
    ../LayerDenseAndBias.cpp
HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrain.h \
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h


#ui
SOURCES += \
    main.cpp\
    mainwindow.cpp \
    SimpleCurve.cpp \
    DNNEngineTestDnn.cpp \
    DNNEngine.cpp
HEADERS  += \
    mainwindow.h \
    SimpleCurve.h \
    DNNEngine.h \
    DNNEngineTestDnn.h
FORMS    += mainwindow.ui

