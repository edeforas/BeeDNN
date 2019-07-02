/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <FrameNetwork.h>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "FrameGlobal.h"
#include "FrameLearning.h"
#include "FrameNotes.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionQuit;
    QAction *actionAbout;
    QAction *actionNew;
    QAction *actionOpen;
    QAction *actionSave;
    QAction *actionSave_as;
    QAction *actionClose;
    QAction *actionSave_with_Score;
    QAction *actionReload;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGridLayout *gridLayout_3;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_10;
    QVBoxLayout *layoutLossCurve;
    QGroupBox *gbTrainAccuracy;
    QGridLayout *gridLayout_2;
    QVBoxLayout *layoutAccuracyCurve;
    QHBoxLayout *horizontalLayout_7;
    QCheckBox *cbHoldOn;
    QPushButton *buttonColor;
    QPushButton *pushButton_2;
    QCheckBox *cbYLogAxis;
    QSpacerItem *horizontalSpacer_2;
    QWidget *tab_2;
    QGridLayout *gridLayout_5;
    QGroupBox *gbRegression;
    QGridLayout *gridLayout_7;
    QVBoxLayout *layoutRegression;
    QWidget *tab_5;
    QVBoxLayout *verticalLayout_5;
    QVBoxLayout *verticalLayout_3;
    QTableWidget *twConfusionMatrixTrain;
    QHBoxLayout *horizontalLayout_8;
    QCheckBox *cbConfMatPercent;
    QSpacerItem *horizontalSpacer_5;
    QWidget *tab_3;
    QGridLayout *gridLayout_9;
    QGraphicsView *gvTopology;
    QWidget *tab_4;
    QGridLayout *gridLayout_8;
    QPlainTextEdit *peDetails;
    QHBoxLayout *horizontalLayout_11;
    QPushButton *pushButton_3;
    QSpacerItem *horizontalSpacer_11;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuHelp;
    QDockWidget *dockWidget_7;
    QWidget *dockWidgetContents_7;
    QVBoxLayout *verticalLayout;
    FrameGlobal *frameGlobal;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_4;
    FrameNetwork *frameNetwork;
    QDockWidget *dockWidget_3;
    QWidget *dockWidgetContents_3;
    QGridLayout *gridLayout_4;
    FrameLearning *frameLearning;
    QHBoxLayout *horizontalLayout;
    QPushButton *btnTestOnly;
    QPushButton *btnTrainAndTest;
    QPushButton *btnTrainMore;
    QSpacerItem *horizontalSpacer;
    QDockWidget *dockWidget_4;
    QWidget *dockWidgetContents_4;
    QGridLayout *gridLayout_6;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_5;
    QLineEdit *leMSE;
    QSpacerItem *horizontalSpacer_4;
    QLabel *label_26;
    QLineEdit *leTrainAccuracy;
    QLabel *label_27;
    QSpacerItem *horizontalSpacer_8;
    QLabel *label_17;
    QLineEdit *leTestAccuracy;
    QLabel *label_25;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_13;
    QLineEdit *leComputedEpochs;
    QLabel *label_23;
    QLineEdit *leTimeByEpoch;
    QLabel *label_24;
    QSpacerItem *horizontalSpacer_7;
    QDockWidget *dockNotes;
    QWidget *dockWidgetContents_2;
    QGridLayout *gridLayout_11;
    FrameNotes *frameNotes;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1141, 737);
        MainWindow->setDocumentMode(false);
        actionQuit = new QAction(MainWindow);
        actionQuit->setObjectName(QStringLiteral("actionQuit"));
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QStringLiteral("actionAbout"));
        actionNew = new QAction(MainWindow);
        actionNew->setObjectName(QStringLiteral("actionNew"));
        actionNew->setEnabled(true);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionOpen->setEnabled(true);
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QStringLiteral("actionSave"));
        actionSave->setEnabled(true);
        actionSave_as = new QAction(MainWindow);
        actionSave_as->setObjectName(QStringLiteral("actionSave_as"));
        actionSave_as->setEnabled(true);
        actionClose = new QAction(MainWindow);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        actionClose->setEnabled(true);
        actionSave_with_Score = new QAction(MainWindow);
        actionSave_with_Score->setObjectName(QStringLiteral("actionSave_with_Score"));
        actionSave_with_Score->setEnabled(true);
        actionReload = new QAction(MainWindow);
        actionReload->setObjectName(QStringLiteral("actionReload"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        gridLayout_3 = new QGridLayout(tab);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        groupBox_2 = new QGroupBox(tab);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_10 = new QGridLayout(groupBox_2);
        gridLayout_10->setSpacing(6);
        gridLayout_10->setContentsMargins(11, 11, 11, 11);
        gridLayout_10->setObjectName(QStringLiteral("gridLayout_10"));
        layoutLossCurve = new QVBoxLayout();
        layoutLossCurve->setSpacing(6);
        layoutLossCurve->setObjectName(QStringLiteral("layoutLossCurve"));

        gridLayout_10->addLayout(layoutLossCurve, 0, 0, 1, 1);


        gridLayout_3->addWidget(groupBox_2, 1, 0, 1, 1);

        gbTrainAccuracy = new QGroupBox(tab);
        gbTrainAccuracy->setObjectName(QStringLiteral("gbTrainAccuracy"));
        gridLayout_2 = new QGridLayout(gbTrainAccuracy);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        layoutAccuracyCurve = new QVBoxLayout();
        layoutAccuracyCurve->setSpacing(6);
        layoutAccuracyCurve->setObjectName(QStringLiteral("layoutAccuracyCurve"));

        gridLayout_2->addLayout(layoutAccuracyCurve, 0, 0, 1, 1);


        gridLayout_3->addWidget(gbTrainAccuracy, 0, 0, 1, 1);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        cbHoldOn = new QCheckBox(tab);
        cbHoldOn->setObjectName(QStringLiteral("cbHoldOn"));

        horizontalLayout_7->addWidget(cbHoldOn);

        buttonColor = new QPushButton(tab);
        buttonColor->setObjectName(QStringLiteral("buttonColor"));

        horizontalLayout_7->addWidget(buttonColor);

        pushButton_2 = new QPushButton(tab);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));

        horizontalLayout_7->addWidget(pushButton_2);

        cbYLogAxis = new QCheckBox(tab);
        cbYLogAxis->setObjectName(QStringLiteral("cbYLogAxis"));

        horizontalLayout_7->addWidget(cbYLogAxis);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_2);


        gridLayout_3->addLayout(horizontalLayout_7, 2, 0, 1, 1);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        gridLayout_5 = new QGridLayout(tab_2);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gbRegression = new QGroupBox(tab_2);
        gbRegression->setObjectName(QStringLiteral("gbRegression"));
        gridLayout_7 = new QGridLayout(gbRegression);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        layoutRegression = new QVBoxLayout();
        layoutRegression->setSpacing(6);
        layoutRegression->setObjectName(QStringLiteral("layoutRegression"));

        gridLayout_7->addLayout(layoutRegression, 0, 0, 1, 1);


        gridLayout_5->addWidget(gbRegression, 0, 0, 1, 1);

        tabWidget->addTab(tab_2, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QStringLiteral("tab_5"));
        verticalLayout_5 = new QVBoxLayout(tab_5);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        twConfusionMatrixTrain = new QTableWidget(tab_5);
        twConfusionMatrixTrain->setObjectName(QStringLiteral("twConfusionMatrixTrain"));
        twConfusionMatrixTrain->verticalHeader()->setCascadingSectionResizes(true);

        verticalLayout_3->addWidget(twConfusionMatrixTrain);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        cbConfMatPercent = new QCheckBox(tab_5);
        cbConfMatPercent->setObjectName(QStringLiteral("cbConfMatPercent"));

        horizontalLayout_8->addWidget(cbConfMatPercent);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_5);


        verticalLayout_3->addLayout(horizontalLayout_8);


        verticalLayout_5->addLayout(verticalLayout_3);

        tabWidget->addTab(tab_5, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        gridLayout_9 = new QGridLayout(tab_3);
        gridLayout_9->setSpacing(6);
        gridLayout_9->setContentsMargins(11, 11, 11, 11);
        gridLayout_9->setObjectName(QStringLiteral("gridLayout_9"));
        gvTopology = new QGraphicsView(tab_3);
        gvTopology->setObjectName(QStringLiteral("gvTopology"));

        gridLayout_9->addWidget(gvTopology, 0, 0, 1, 1);

        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QStringLiteral("tab_4"));
        gridLayout_8 = new QGridLayout(tab_4);
        gridLayout_8->setSpacing(6);
        gridLayout_8->setContentsMargins(11, 11, 11, 11);
        gridLayout_8->setObjectName(QStringLiteral("gridLayout_8"));
        peDetails = new QPlainTextEdit(tab_4);
        peDetails->setObjectName(QStringLiteral("peDetails"));

        gridLayout_8->addWidget(peDetails, 0, 0, 1, 1);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        pushButton_3 = new QPushButton(tab_4);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));

        horizontalLayout_11->addWidget(pushButton_3);

        horizontalSpacer_11 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_11);


        gridLayout_8->addLayout(horizontalLayout_11, 1, 0, 1, 1);

        tabWidget->addTab(tab_4, QString());

        gridLayout->addWidget(tabWidget, 0, 1, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1141, 26));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        MainWindow->setMenuBar(menuBar);
        dockWidget_7 = new QDockWidget(MainWindow);
        dockWidget_7->setObjectName(QStringLiteral("dockWidget_7"));
        dockWidgetContents_7 = new QWidget();
        dockWidgetContents_7->setObjectName(QStringLiteral("dockWidgetContents_7"));
        verticalLayout = new QVBoxLayout(dockWidgetContents_7);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        frameGlobal = new FrameGlobal(dockWidgetContents_7);
        frameGlobal->setObjectName(QStringLiteral("frameGlobal"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(frameGlobal->sizePolicy().hasHeightForWidth());
        frameGlobal->setSizePolicy(sizePolicy1);
        frameGlobal->setFrameShape(QFrame::StyledPanel);
        frameGlobal->setFrameShadow(QFrame::Raised);

        verticalLayout->addWidget(frameGlobal);

        dockWidget_7->setWidget(dockWidgetContents_7);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_7);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(dockWidget->sizePolicy().hasHeightForWidth());
        dockWidget->setSizePolicy(sizePolicy2);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        verticalLayout_4 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        frameNetwork = new FrameNetwork(dockWidgetContents);
        frameNetwork->setObjectName(QStringLiteral("frameNetwork"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(frameNetwork->sizePolicy().hasHeightForWidth());
        frameNetwork->setSizePolicy(sizePolicy3);
        frameNetwork->setFrameShape(QFrame::StyledPanel);
        frameNetwork->setFrameShadow(QFrame::Raised);

        verticalLayout_4->addWidget(frameNetwork);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        dockWidget_3 = new QDockWidget(MainWindow);
        dockWidget_3->setObjectName(QStringLiteral("dockWidget_3"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Minimum);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(dockWidget_3->sizePolicy().hasHeightForWidth());
        dockWidget_3->setSizePolicy(sizePolicy4);
        dockWidgetContents_3 = new QWidget();
        dockWidgetContents_3->setObjectName(QStringLiteral("dockWidgetContents_3"));
        gridLayout_4 = new QGridLayout(dockWidgetContents_3);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        frameLearning = new FrameLearning(dockWidgetContents_3);
        frameLearning->setObjectName(QStringLiteral("frameLearning"));
        frameLearning->setFrameShape(QFrame::StyledPanel);
        frameLearning->setFrameShadow(QFrame::Raised);

        gridLayout_4->addWidget(frameLearning, 0, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        btnTestOnly = new QPushButton(dockWidgetContents_3);
        btnTestOnly->setObjectName(QStringLiteral("btnTestOnly"));

        horizontalLayout->addWidget(btnTestOnly);

        btnTrainAndTest = new QPushButton(dockWidgetContents_3);
        btnTrainAndTest->setObjectName(QStringLiteral("btnTrainAndTest"));

        horizontalLayout->addWidget(btnTrainAndTest);

        btnTrainMore = new QPushButton(dockWidgetContents_3);
        btnTrainMore->setObjectName(QStringLiteral("btnTrainMore"));

        horizontalLayout->addWidget(btnTrainMore);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        gridLayout_4->addLayout(horizontalLayout, 1, 0, 1, 1);

        dockWidget_3->setWidget(dockWidgetContents_3);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_3);
        dockWidget_4 = new QDockWidget(MainWindow);
        dockWidget_4->setObjectName(QStringLiteral("dockWidget_4"));
        sizePolicy4.setHeightForWidth(dockWidget_4->sizePolicy().hasHeightForWidth());
        dockWidget_4->setSizePolicy(sizePolicy4);
        dockWidgetContents_4 = new QWidget();
        dockWidgetContents_4->setObjectName(QStringLiteral("dockWidgetContents_4"));
        gridLayout_6 = new QGridLayout(dockWidgetContents_4);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        label_5 = new QLabel(dockWidgetContents_4);
        label_5->setObjectName(QStringLiteral("label_5"));

        horizontalLayout_2->addWidget(label_5);

        leMSE = new QLineEdit(dockWidgetContents_4);
        leMSE->setObjectName(QStringLiteral("leMSE"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(leMSE->sizePolicy().hasHeightForWidth());
        leMSE->setSizePolicy(sizePolicy5);
        leMSE->setReadOnly(true);

        horizontalLayout_2->addWidget(leMSE);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_4);

        label_26 = new QLabel(dockWidgetContents_4);
        label_26->setObjectName(QStringLiteral("label_26"));

        horizontalLayout_2->addWidget(label_26);

        leTrainAccuracy = new QLineEdit(dockWidgetContents_4);
        leTrainAccuracy->setObjectName(QStringLiteral("leTrainAccuracy"));
        sizePolicy5.setHeightForWidth(leTrainAccuracy->sizePolicy().hasHeightForWidth());
        leTrainAccuracy->setSizePolicy(sizePolicy5);

        horizontalLayout_2->addWidget(leTrainAccuracy);

        label_27 = new QLabel(dockWidgetContents_4);
        label_27->setObjectName(QStringLiteral("label_27"));

        horizontalLayout_2->addWidget(label_27);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_8);

        label_17 = new QLabel(dockWidgetContents_4);
        label_17->setObjectName(QStringLiteral("label_17"));

        horizontalLayout_2->addWidget(label_17);

        leTestAccuracy = new QLineEdit(dockWidgetContents_4);
        leTestAccuracy->setObjectName(QStringLiteral("leTestAccuracy"));
        sizePolicy5.setHeightForWidth(leTestAccuracy->sizePolicy().hasHeightForWidth());
        leTestAccuracy->setSizePolicy(sizePolicy5);

        horizontalLayout_2->addWidget(leTestAccuracy);

        label_25 = new QLabel(dockWidgetContents_4);
        label_25->setObjectName(QStringLiteral("label_25"));

        horizontalLayout_2->addWidget(label_25);


        gridLayout_6->addLayout(horizontalLayout_2, 1, 0, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        label_13 = new QLabel(dockWidgetContents_4);
        label_13->setObjectName(QStringLiteral("label_13"));

        horizontalLayout_5->addWidget(label_13);

        leComputedEpochs = new QLineEdit(dockWidgetContents_4);
        leComputedEpochs->setObjectName(QStringLiteral("leComputedEpochs"));
        sizePolicy5.setHeightForWidth(leComputedEpochs->sizePolicy().hasHeightForWidth());
        leComputedEpochs->setSizePolicy(sizePolicy5);

        horizontalLayout_5->addWidget(leComputedEpochs);

        label_23 = new QLabel(dockWidgetContents_4);
        label_23->setObjectName(QStringLiteral("label_23"));

        horizontalLayout_5->addWidget(label_23);

        leTimeByEpoch = new QLineEdit(dockWidgetContents_4);
        leTimeByEpoch->setObjectName(QStringLiteral("leTimeByEpoch"));

        horizontalLayout_5->addWidget(leTimeByEpoch);

        label_24 = new QLabel(dockWidgetContents_4);
        label_24->setObjectName(QStringLiteral("label_24"));

        horizontalLayout_5->addWidget(label_24);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_7);


        gridLayout_6->addLayout(horizontalLayout_5, 0, 0, 1, 1);

        dockWidget_4->setWidget(dockWidgetContents_4);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget_4);
        dockNotes = new QDockWidget(MainWindow);
        dockNotes->setObjectName(QStringLiteral("dockNotes"));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QStringLiteral("dockWidgetContents_2"));
        gridLayout_11 = new QGridLayout(dockWidgetContents_2);
        gridLayout_11->setSpacing(6);
        gridLayout_11->setContentsMargins(11, 11, 11, 11);
        gridLayout_11->setObjectName(QStringLiteral("gridLayout_11"));
        frameNotes = new FrameNotes(dockWidgetContents_2);
        frameNotes->setObjectName(QStringLiteral("frameNotes"));
        frameNotes->setFrameShape(QFrame::StyledPanel);
        frameNotes->setFrameShadow(QFrame::Raised);

        gridLayout_11->addWidget(frameNotes, 0, 0, 1, 1);

        dockNotes->setWidget(dockWidgetContents_2);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockNotes);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionNew);
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave);
        menuFile->addAction(actionSave_with_Score);
        menuFile->addAction(actionSave_as);
        menuFile->addAction(actionReload);
        menuFile->addAction(actionClose);
        menuFile->addSeparator();
        menuFile->addAction(actionQuit);
        menuHelp->addAction(actionAbout);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "DNNLab", nullptr));
        actionQuit->setText(QApplication::translate("MainWindow", "Exit", nullptr));
        actionAbout->setText(QApplication::translate("MainWindow", "About", nullptr));
        actionNew->setText(QApplication::translate("MainWindow", "New", nullptr));
        actionOpen->setText(QApplication::translate("MainWindow", "Open...", nullptr));
        actionSave->setText(QApplication::translate("MainWindow", "Save", nullptr));
        actionSave_as->setText(QApplication::translate("MainWindow", "Save as...", nullptr));
        actionClose->setText(QApplication::translate("MainWindow", "Close", nullptr));
        actionSave_with_Score->setText(QApplication::translate("MainWindow", "Save with Score", nullptr));
        actionReload->setText(QApplication::translate("MainWindow", "Reload", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Loss", nullptr));
        gbTrainAccuracy->setTitle(QApplication::translate("MainWindow", "Train Accuracy", nullptr));
        cbHoldOn->setText(QApplication::translate("MainWindow", "HoldOn", nullptr));
        buttonColor->setText(QApplication::translate("MainWindow", "Color...", nullptr));
        pushButton_2->setText(QApplication::translate("MainWindow", "Clear", nullptr));
        cbYLogAxis->setText(QApplication::translate("MainWindow", "YLogAxis", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Train Curves", nullptr));
        gbRegression->setTitle(QApplication::translate("MainWindow", "Red=truth, Blue=net output", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Regression", nullptr));
        cbConfMatPercent->setText(QApplication::translate("MainWindow", "Disp Percent", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QApplication::translate("MainWindow", "Classification", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("MainWindow", "Topology", nullptr));
        pushButton_3->setText(QApplication::translate("MainWindow", "Copy to ClipBoard", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("MainWindow", "Details", nullptr));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", nullptr));
        menuHelp->setTitle(QApplication::translate("MainWindow", "Help", nullptr));
        dockWidget_7->setWindowTitle(QApplication::translate("MainWindow", "Global", nullptr));
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Network", nullptr));
        dockWidget_3->setWindowTitle(QApplication::translate("MainWindow", "Learning", nullptr));
        btnTestOnly->setText(QApplication::translate("MainWindow", "Test Only", nullptr));
        btnTrainAndTest->setText(QApplication::translate("MainWindow", "Train && Test", nullptr));
        btnTrainMore->setText(QApplication::translate("MainWindow", "Train more", nullptr));
        dockWidget_4->setWindowTitle(QApplication::translate("MainWindow", "Results", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "Loss", nullptr));
        label_26->setText(QApplication::translate("MainWindow", "TrainAccuracy", nullptr));
        label_27->setText(QApplication::translate("MainWindow", "%", nullptr));
        label_17->setText(QApplication::translate("MainWindow", "TestAccuracy", nullptr));
        label_25->setText(QApplication::translate("MainWindow", "%", nullptr));
        label_13->setText(QApplication::translate("MainWindow", "ComputedEpochs", nullptr));
        label_23->setText(QApplication::translate("MainWindow", "Time/Epochs", nullptr));
        label_24->setText(QApplication::translate("MainWindow", "s", nullptr));
        dockNotes->setWindowTitle(QApplication::translate("MainWindow", "Notes", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
