#include "SimpleCurve.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsSceneWheelEvent>

#include <cassert>
#include <cmath>

//////////////////////////////////////////////////////////////////////////
SimpleCurve::SimpleCurve(): QGraphicsScene()
{
    _bYLogAxis=false;
}
//////////////////////////////////////////////////////////////////////////
SimpleCurve::~SimpleCurve()
{ }
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB)
{ 
    assert(vdX.size()==vdY.size());

    if(vdX.size()==0)
        return;

    //save and draw curve data
    CurveData cd;
    cd.vdX=vdX;
    cd.vdY=vdY;
    cd._iColorRGB=iColorRGB;

    // compute bounding box
    double xMin,yMin,xMax,yMax;
    xMin=vdX[0];
    xMax=vdX[0];
    yMin=vdY[0];
    yMax=vdY[0];
    for(int i=1;i<vdX.size();i++)
    {
        if(vdX[i]<xMin)
            xMin=vdX[i];

        if(vdX[i]>xMax)
            xMax=vdX[i];

        if(vdY[i]<yMin)
            yMin=vdY[i];

        if(vdY[i]>yMax)
            yMax=vdY[i];
    }

    cd.xMin=xMin;
    cd.xMax=xMax;
    cd.yMin=yMin;
    cd.yMax=yMax;

    _vCurves.push_back(cd);
    compute_bounding_box();
    replot_curve(_vCurves.size()-1);

    replot_axis(); //bug! must erase axis before.
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
    _bDrawXaxis=true;
    replot_axis();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::addYAxis()
{
    _bDrawYaxis=true;
    replot_axis();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::compute_bounding_box()
{
    if(_vCurves.size()==0)
    {
        xMin=xMax=0.;
        yMin=yMax=0.;

        if(_bYLogAxis)
        {
            yMin=yMax=-8.;
        }

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

    if(_bYLogAxis)
    {
        yMin=log10(max<double>(yMin,1.e-8));
        yMax=log10(max<double>(yMax,1.e-8));
    }
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::setYLogAxis(bool bSetLogAxis)
{
    _bYLogAxis=bSetLogAxis;
    replot_all();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::clear()
{
    QGraphicsScene::clear();
    _vCurves.clear();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::replot_all()
{
    QGraphicsScene::clear();

    compute_bounding_box();
    replot_axis();

    if(_bDrawXaxis)
        addXAxis();

    if(_bDrawYaxis)
        addYAxis();

    for(int i=0;i<_vCurves.size();i++)
        replot_curve(i);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::replot_curve(int iCurve)
{
    const CurveData& curve=_vCurves[iCurve];

    QPainterPath painter;

    if(!_bYLogAxis)
    {
        painter.moveTo(QPointF(curve.vdX[0],curve.vdY[0]));
        for(unsigned int i=1;i<curve.vdX.size();i++)
            painter.lineTo(QPointF(curve.vdX[i],curve.vdY[i]));
    }
    else
    {
        painter.moveTo(QPointF(curve.vdX[0],log10(max<double>(curve.vdY[0],1.e-8))));
        for(unsigned int i=1;i<curve.vdX.size();i++)
            painter.lineTo(QPointF(curve.vdX[i],log10(max<double>(curve.vdY[i],1.e-8))));
    }

    QPen penBlack(QRgb(curve._iColorRGB));
    penBlack.setCosmetic(true);

    addPath(painter,penBlack);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurve::replot_axis()
{
    compute_bounding_box();

    if(_bDrawXaxis)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        addLine(xMin,0.,xMax,0.,penBlack);
    }

    if(_bDrawYaxis)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        addLine(0.,yMin,0.,yMax,penBlack);
    }
}
//////////////////////////////////////////////////////////////////////////
