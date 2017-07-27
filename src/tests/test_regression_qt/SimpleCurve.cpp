#include "SimpleCurve.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsSceneWheelEvent>
#include <cassert>

//////////////////////////////////////////////////////////////////////////
SimpleCurve::SimpleCurve(): QGraphicsScene()
{ }
//////////////////////////////////////////////////////////////////////////
SimpleCurve::~SimpleCurve()
{ }
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::addCurve(const vector<double>& vdX, const vector<double>& vdY,Qt::GlobalColor color)
{
    QPainterPath painter;

	assert(vdX.size()==vdY.size());
	
	if(vdX.size()==0)
		return;
	
    painter.moveTo(QPointF(vdX[0],vdY[0]));
    for(unsigned int i=1;i<vdX.size();i++)
    {
        painter.lineTo(QPointF(vdX[i],vdY[i]));
    }

    QPen penBlack(color);
    penBlack.setCosmetic(true);

    addPath(painter,penBlack);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::wheelEvent(QGraphicsSceneWheelEvent* wheelEvent)
{
    QGraphicsView* v=views()[0]; //todo check
    v->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    if (wheelEvent->delta()>0)
        v->scale(1.25,1.25);
    else
        v->scale(0.8,0.8);

    wheelEvent->accept();
}
//////////////////////////////////////////////////////////////////////////
