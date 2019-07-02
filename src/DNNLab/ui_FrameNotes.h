/********************************************************************************
** Form generated from reading UI file 'FrameNotes.ui'
**
** Created by: Qt User Interface Compiler version 5.11.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FRAMENOTES_H
#define UI_FRAMENOTES_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QTextEdit>

QT_BEGIN_NAMESPACE

class Ui_FrameNotes
{
public:
    QGridLayout *gridLayout;
    QTextEdit *leNotes;

    void setupUi(QFrame *FrameNotes)
    {
        if (FrameNotes->objectName().isEmpty())
            FrameNotes->setObjectName(QStringLiteral("FrameNotes"));
        FrameNotes->resize(400, 300);
        gridLayout = new QGridLayout(FrameNotes);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        leNotes = new QTextEdit(FrameNotes);
        leNotes->setObjectName(QStringLiteral("leNotes"));
        QPalette palette;
        QBrush brush(QColor(255, 255, 255, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Light, brush);
        QBrush brush1(QColor(255, 254, 248, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Light, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Light, brush);
        QBrush brush2(QColor(240, 240, 240, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush2);
        leNotes->setPalette(palette);
        leNotes->setTextInteractionFlags(Qt::LinksAccessibleByKeyboard|Qt::LinksAccessibleByMouse|Qt::TextBrowserInteraction|Qt::TextEditable|Qt::TextEditorInteraction|Qt::TextSelectableByKeyboard|Qt::TextSelectableByMouse);

        gridLayout->addWidget(leNotes, 0, 0, 1, 1);


        retranslateUi(FrameNotes);

        QMetaObject::connectSlotsByName(FrameNotes);
    } // setupUi

    void retranslateUi(QFrame *FrameNotes)
    {
        FrameNotes->setWindowTitle(QApplication::translate("FrameNotes", "Frame", nullptr));
    } // retranslateUi

};

namespace Ui {
    class FrameNotes: public Ui_FrameNotes {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FRAMENOTES_H
