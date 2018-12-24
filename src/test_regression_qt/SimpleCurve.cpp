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
	assert(vdX.size()==vdY.size());

    QPainterPath painter;

	if(vdX.size()==0)
        return;
	
    //save and draw curve data
    CurveData cd;
    cd.vdX=vdX;
    cd.vdY=vdY;
    cd._iColor=color;

    painter.moveTo(QPointF(vdX[0],vdY[0]));
    for(unsigned int i=1;i<vdX.size();i++)
    {
        painter.lineTo(QPointF(vdX[i],vdY[i]));
    }

    QRectF qr=painter.boundingRect();
    cd.xMin=qr.left();
    cd.xMax=qr.right();
    cd.yMin=qr.top();
    cd.yMax=qr.bottom();
    _vCurves.push_back(cd);

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
void SimpleCurve::addXAxis()
{   
    compute_bounding_box();

    QPen penBlack(Qt::black);
    penBlack.setCosmetic(true);
    addLine(xMin,0.,xMax,0.,penBlack);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::addYAxis()
{
    compute_bounding_box();
    QPen penBlack(Qt::black);
    penBlack.setCosmetic(true);
    addLine(0.,yMin,0.,yMax,penBlack);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::compute_bounding_box()
{
    if(_vCurves.size()==0)
    {
        xMin=xMax=0.;
        yMin=yMax=0.;

        return;
    }

    //compute x range
    xMin=_vCurves[0].xMin;
    xMax=_vCurves[0].xMax;
    yMin=_vCurves[0].yMin;
    yMax=_vCurves[0].yMax;

    for(unsigned int i=1;i<_vCurves.size();i++)
    {
        if(_vCurves[i].xMin<xMin)
            xMin=_vCurves[i].xMin;

        if(_vCurves[i].xMax>xMax)
            xMax=_vCurves[i].xMax;

        if(_vCurves[i].yMin<yMin)
            yMin=_vCurves[i].yMin;

        if(_vCurves[i].yMax>yMax)
            yMax=_vCurves[i].yMax;
    }
}
//////////////////////////////////////////////////////////////////////////