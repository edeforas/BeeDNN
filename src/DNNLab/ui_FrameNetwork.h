/********************************************************************************
** Form generated from reading UI file 'FrameNetwork.ui'
**
** Created by: Qt User Interface Compiler version 5.11.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FRAMENETWORK_H
#define UI_FRAMENETWORK_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTableWidget>

QT_BEGIN_NAMESPACE

class Ui_FrameNetwork
{
public:
    QGridLayout *gridLayout;
    QTableWidget *twNetwork;

    void setupUi(QFrame *FrameNetwork)
    {
        if (FrameNetwork->objectName().isEmpty())
            FrameNetwork->setObjectName(QStringLiteral("FrameNetwork"));
        FrameNetwork->resize(630, 443);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(FrameNetwork->sizePolicy().hasHeightForWidth());
        FrameNetwork->setSizePolicy(sizePolicy);
        gridLayout = new QGridLayout(FrameNetwork);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        twNetwork = new QTableWidget(FrameNetwork);
        if (twNetwork->columnCount() < 4)
            twNetwork->setColumnCount(4);
        if (twNetwork->rowCount() < 10)
            twNetwork->setRowCount(10);
        twNetwork->setObjectName(QStringLiteral("twNetwork"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(60);
        sizePolicy1.setHeightForWidth(twNetwork->sizePolicy().hasHeightForWidth());
        twNetwork->setSizePolicy(sizePolicy1);
        twNetwork->setMinimumSize(QSize(600, 200));
        twNetwork->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        twNetwork->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
        twNetwork->setAutoScroll(false);
        twNetwork->setAlternatingRowColors(true);
        twNetwork->setGridStyle(Qt::SolidLine);
        twNetwork->setCornerButtonEnabled(false);
        twNetwork->setRowCount(10);
        twNetwork->setColumnCount(4);
        twNetwork->horizontalHeader()->setDefaultSectionSize(150);
        twNetwork->horizontalHeader()->setMinimumSectionSize(130);
        twNetwork->horizontalHeader()->setStretchLastSection(false);

        gridLayout->addWidget(twNetwork, 0, 0, 1, 1);


        retranslateUi(FrameNetwork);

        QMetaObject::connectSlotsByName(FrameNetwork);
    } // setupUi

    void retranslateUi(QFrame *FrameNetwork)
    {
        FrameNetwork->setWindowTitle(QApplication::translate("FrameNetwork", "Frame", nullptr));
    } // retranslateUi

};

namespace Ui {
    class FrameNetwork: public Ui_FrameNetwork {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FRAMENETWORK_H
