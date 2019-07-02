/********************************************************************************
** Form generated from reading UI file 'FrameLearning.ui'
**
** Created by: Qt User Interface Compiler version 5.11.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FRAMELEARNING_H
#define UI_FRAMELEARNING_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_FrameLearning
{
public:
    QGridLayout *gridLayout;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_4;
    QLineEdit *leReboost;
    QLabel *label_14;
    QSpacerItem *horizontalSpacer_12;
    QCheckBox *cbKeepBest;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_8;
    QLineEdit *leEpochs;
    QSpacerItem *horizontalSpacer;
    QLabel *label_7;
    QLineEdit *leBatchSize;
    QSpacerItem *horizontalSpacer_9;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QComboBox *cbOptimizer;
    QSpacerItem *horizontalSpacer_10;
    QLabel *label_2;
    QComboBox *cbLossFunction;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_9;
    QLineEdit *leLearningRate;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label_11;
    QLineEdit *leDecay;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_10;
    QLineEdit *leMomentum;

    void setupUi(QFrame *FrameLearning)
    {
        if (FrameLearning->objectName().isEmpty())
            FrameLearning->setObjectName(QStringLiteral("FrameLearning"));
        FrameLearning->resize(490, 168);
        gridLayout = new QGridLayout(FrameLearning);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        label_4 = new QLabel(FrameLearning);
        label_4->setObjectName(QStringLiteral("label_4"));

        horizontalLayout_6->addWidget(label_4);

        leReboost = new QLineEdit(FrameLearning);
        leReboost->setObjectName(QStringLiteral("leReboost"));
        leReboost->setEnabled(true);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(leReboost->sizePolicy().hasHeightForWidth());
        leReboost->setSizePolicy(sizePolicy);

        horizontalLayout_6->addWidget(leReboost);

        label_14 = new QLabel(FrameLearning);
        label_14->setObjectName(QStringLiteral("label_14"));

        horizontalLayout_6->addWidget(label_14);

        horizontalSpacer_12 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_12);

        cbKeepBest = new QCheckBox(FrameLearning);
        cbKeepBest->setObjectName(QStringLiteral("cbKeepBest"));
        cbKeepBest->setChecked(true);

        horizontalLayout_6->addWidget(cbKeepBest);


        gridLayout->addLayout(horizontalLayout_6, 3, 0, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        label_8 = new QLabel(FrameLearning);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_3->addWidget(label_8);

        leEpochs = new QLineEdit(FrameLearning);
        leEpochs->setObjectName(QStringLiteral("leEpochs"));
        sizePolicy.setHeightForWidth(leEpochs->sizePolicy().hasHeightForWidth());
        leEpochs->setSizePolicy(sizePolicy);

        horizontalLayout_3->addWidget(leEpochs);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer);

        label_7 = new QLabel(FrameLearning);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout_3->addWidget(label_7);

        leBatchSize = new QLineEdit(FrameLearning);
        leBatchSize->setObjectName(QStringLiteral("leBatchSize"));
        leBatchSize->setEnabled(true);
        sizePolicy.setHeightForWidth(leBatchSize->sizePolicy().hasHeightForWidth());
        leBatchSize->setSizePolicy(sizePolicy);

        horizontalLayout_3->addWidget(leBatchSize);

        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_9);


        gridLayout->addLayout(horizontalLayout_3, 2, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(FrameLearning);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        cbOptimizer = new QComboBox(FrameLearning);
        cbOptimizer->setObjectName(QStringLiteral("cbOptimizer"));

        horizontalLayout->addWidget(cbOptimizer);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_10);

        label_2 = new QLabel(FrameLearning);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout->addWidget(label_2);

        cbLossFunction = new QComboBox(FrameLearning);
        cbLossFunction->setObjectName(QStringLiteral("cbLossFunction"));
        cbLossFunction->setEnabled(true);

        horizontalLayout->addWidget(cbLossFunction);


        gridLayout->addLayout(horizontalLayout, 0, 0, 1, 1);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        label_9 = new QLabel(FrameLearning);
        label_9->setObjectName(QStringLiteral("label_9"));

        horizontalLayout_4->addWidget(label_9);

        leLearningRate = new QLineEdit(FrameLearning);
        leLearningRate->setObjectName(QStringLiteral("leLearningRate"));

        horizontalLayout_4->addWidget(leLearningRate);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_2);

        label_11 = new QLabel(FrameLearning);
        label_11->setObjectName(QStringLiteral("label_11"));

        horizontalLayout_4->addWidget(label_11);

        leDecay = new QLineEdit(FrameLearning);
        leDecay->setObjectName(QStringLiteral("leDecay"));
        leDecay->setEnabled(true);

        horizontalLayout_4->addWidget(leDecay);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_3);

        label_10 = new QLabel(FrameLearning);
        label_10->setObjectName(QStringLiteral("label_10"));

        horizontalLayout_4->addWidget(label_10);

        leMomentum = new QLineEdit(FrameLearning);
        leMomentum->setObjectName(QStringLiteral("leMomentum"));
        leMomentum->setEnabled(true);
        leMomentum->setReadOnly(false);

        horizontalLayout_4->addWidget(leMomentum);


        gridLayout->addLayout(horizontalLayout_4, 1, 0, 1, 1);


        retranslateUi(FrameLearning);

        cbOptimizer->setCurrentIndex(-1);


        QMetaObject::connectSlotsByName(FrameLearning);
    } // setupUi

    void retranslateUi(QFrame *FrameLearning)
    {
        FrameLearning->setWindowTitle(QApplication::translate("FrameLearning", "Frame", nullptr));
        label_4->setText(QApplication::translate("FrameLearning", "Reboost every", nullptr));
        leReboost->setText(QApplication::translate("FrameLearning", "-1", nullptr));
        label_14->setText(QApplication::translate("FrameLearning", "Epochs", nullptr));
        cbKeepBest->setText(QApplication::translate("FrameLearning", "KeepBest", nullptr));
        label_8->setText(QApplication::translate("FrameLearning", "Epochs", nullptr));
        leEpochs->setText(QApplication::translate("FrameLearning", "100", nullptr));
        label_7->setText(QApplication::translate("FrameLearning", "BatchSize", nullptr));
        leBatchSize->setText(QApplication::translate("FrameLearning", "16", nullptr));
        label->setText(QApplication::translate("FrameLearning", "Optimizer", nullptr));
        label_2->setText(QApplication::translate("FrameLearning", "LossFunction", nullptr));
        label_9->setText(QApplication::translate("FrameLearning", "Learning Rate", nullptr));
        leLearningRate->setText(QApplication::translate("FrameLearning", "-1", nullptr));
        label_11->setText(QApplication::translate("FrameLearning", "Decay", nullptr));
        leDecay->setText(QApplication::translate("FrameLearning", "-1", nullptr));
        label_10->setText(QApplication::translate("FrameLearning", "Momentum", nullptr));
        leMomentum->setText(QApplication::translate("FrameLearning", "-1", nullptr));
    } // retranslateUi

};

namespace Ui {
    class FrameLearning: public Ui_FrameLearning {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FRAMELEARNING_H
