/********************************************************************************
** Form generated from reading UI file 'FrameGlobal.ui'
**
** Created by: Qt User Interface Compiler version 5.11.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FRAMEGLOBAL_H
#define UI_FRAMEGLOBAL_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_FrameGlobal
{
public:
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label_22;
    QComboBox *cbEngine;
    QSpacerItem *horizontalSpacer;
    QLabel *label_16;
    QComboBox *cbProblem;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label_6;
    QComboBox *cbData;

    void setupUi(QFrame *FrameGlobal)
    {
        if (FrameGlobal->objectName().isEmpty())
            FrameGlobal->setObjectName(QStringLiteral("FrameGlobal"));
        FrameGlobal->resize(500, 46);
        gridLayout = new QGridLayout(FrameGlobal);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label_22 = new QLabel(FrameGlobal);
        label_22->setObjectName(QStringLiteral("label_22"));

        horizontalLayout->addWidget(label_22);

        cbEngine = new QComboBox(FrameGlobal);
        cbEngine->addItem(QString());
        cbEngine->setObjectName(QStringLiteral("cbEngine"));

        horizontalLayout->addWidget(cbEngine);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        label_16 = new QLabel(FrameGlobal);
        label_16->setObjectName(QStringLiteral("label_16"));

        horizontalLayout->addWidget(label_16);

        cbProblem = new QComboBox(FrameGlobal);
        cbProblem->addItem(QString());
        cbProblem->addItem(QString());
        cbProblem->setObjectName(QStringLiteral("cbProblem"));

        horizontalLayout->addWidget(cbProblem);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        label_6 = new QLabel(FrameGlobal);
        label_6->setObjectName(QStringLiteral("label_6"));

        horizontalLayout->addWidget(label_6);

        cbData = new QComboBox(FrameGlobal);
        cbData->setObjectName(QStringLiteral("cbData"));
        cbData->setEditable(true);
        cbData->setSizeAdjustPolicy(QComboBox::AdjustToContentsOnFirstShow);
        cbData->setFrame(true);

        horizontalLayout->addWidget(cbData);


        gridLayout->addLayout(horizontalLayout, 0, 0, 1, 1);


        retranslateUi(FrameGlobal);

        QMetaObject::connectSlotsByName(FrameGlobal);
    } // setupUi

    void retranslateUi(QFrame *FrameGlobal)
    {
        FrameGlobal->setWindowTitle(QApplication::translate("FrameGlobal", "Frame", nullptr));
        label_22->setText(QApplication::translate("FrameGlobal", "Engine", nullptr));
        cbEngine->setItemText(0, QApplication::translate("FrameGlobal", "BeeDNN", nullptr));

        label_16->setText(QApplication::translate("FrameGlobal", "Problem", nullptr));
        cbProblem->setItemText(0, QApplication::translate("FrameGlobal", "Regression", nullptr));
        cbProblem->setItemText(1, QApplication::translate("FrameGlobal", "Classification", nullptr));

        label_6->setText(QApplication::translate("FrameGlobal", "Data", nullptr));
    } // retranslateUi

};

namespace Ui {
    class FrameGlobal: public Ui_FrameGlobal {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FRAMEGLOBAL_H
