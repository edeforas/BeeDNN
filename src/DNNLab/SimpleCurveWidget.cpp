#include "SimpleCurveWidget.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QWheelEvent>
#include <QGraphicsSceneWheelEvent>

#include <cassert>
#include <cmath>

//////////////////////////////////////////////////////////////////////////
SimpleCurveWidget::SimpleCurveWidget(): QGraphicsView()
{
    setScene(&_qs);
    setDragMode(ScrollHandDrag);
    //    setMouseTracking(true);
    //    setInteractive(true);
    //    setTabletTracking(true);

    xMin=0;
    xMax=0;

    yMin=0;
    yMax=0;

    _bDrawXaxis=true;
    _bDrawYaxis=true;
    _bYLogAxis=false;
}
//////////////////////////////////////////////////////////////////////////
SimpleCurveWidget::~SimpleCurveWidget()
{ }
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB)
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
    for(unsigned int i=1;i<vdX.size();i++)
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

    replot_all();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::wheelEvent(QWheelEvent* event)
{
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    if (event->delta()>0)
        scale(1.25,1.25);
    else
        scale(0.8,0.8);

    event->accept();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addXAxis()
{   
    _bDrawXaxis=true;
    replot_axis();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addYAxis()
{
    _bDrawYaxis=true;
    replot_axis();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::compute_bounding_box()
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

    yMinL=log10(max(yMin,1.e-8));
    yMaxL=log10(max(yMax,1.e-8));
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::setYLogAxis(bool bSetLogAxis)
{
    _bYLogAxis=bSetLogAxis;
    replot_all();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::clear()
{
    _qs.clear();
    _vCurves.clear();
    compute_bounding_box();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::replot_all()
{  
    _qs.clear();
    compute_bounding_box();
    replot_axis();

    for(unsigned int i=0;i<_vCurves.size();i++)
        replot_curve((int)i);

    setSceneRect(_qs.itemsBoundingRect());//QRectF(xMin,yMin,xMax-xMin,yMax-yMin)); //for now
    fitInView(_qs.itemsBoundingRect());
    scale(0.9,0.9);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::replot_curve(int iCurve)
{
    const CurveData& curve=_vCurves[(unsigned int)iCurve];

    QPainterPath painter;

    if(!_bYLogAxis)
    {
        painter.moveTo(QPointF(curve.vdX[0],-curve.vdY[0]));
        for(unsigned int i=1;i<curve.vdX.size();i++)
            painter.lineTo(QPointF(curve.vdX[i],-curve.vdY[i]));
    }
    else
    {
        painter.moveTo(QPointF(curve.vdX[0], -log10(max(curve.vdY[0],1.e-8))));
        for(unsigned int i=1;i<curve.vdX.size();i++)
            painter.lineTo(QPointF(curve.vdX[i], -log10(max(curve.vdY[i],1.e-8)) ));
    }

    QPen penBlack(QRgb(curve._iColorRGB));
    penBlack.setCosmetic(true);

    _qs.addPath(painter,penBlack);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::replot_axis()
{
    if(_bDrawXaxis)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        _qs.addLine(xMin,0.,xMax,0.,penBlack);
    }

    if(_bDrawYaxis)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        if(_bYLogAxis)
            _qs.addLine(0.,-yMaxL,0.,yMinL ,penBlack);
        else
            _qs.addLine(0.,-yMax ,0.,yMin,penBlack);
    }
}
//////////////////////////////////////////////////////////////////////////
