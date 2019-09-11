#include "SimpleCurveWidget.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#if USE_QWT

#include <qwt_plot_curve.h>
#include <qwt_plot_grid.h>
#include <qwt_scale_engine.h>

SimpleCurveWidget::SimpleCurveWidget(): QwtPlot()
{
    //    QwtPlotGrid *grid = new QwtPlotGrid();
    //   grid->attach( this );

    _zoomer = new QwtPlotZoomer(canvas() );
}
//////////////////////////////////////////////////////////////////////////
SimpleCurveWidget::~SimpleCurveWidget()
{ }
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::clear()
{
    detachItems();
    replot();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addXAxis()
{
    //TODO
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addYAxis()
{
    //TODO
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addHorizontalLine(double dY)
{
    (void)dY;
    //TODO
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::setYLogAxis(bool bSetLogAxis)
{
    if(bSetLogAxis)
        setAxisScaleEngine(0, new QwtLogScaleEngine());
    else
        setAxisScaleEngine(0,new QwtLinearScaleEngine());

    replot();
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addCurve(const vector<float>& vfX, const vector<float>& vfY,unsigned int iColorRGB)
{
    vector<double> vdX,vdY;
    for(unsigned int i=0;i<vfX.size();i++)
    {
        vdX.push_back((double)vfX[i]);
        vdY.push_back((double)vfY[i]);
    }

    addCurve(vdX,vdY,iColorRGB);
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB)
{
    QwtPlotCurve *curve = new QwtPlotCurve();
    curve->setRenderHint( QwtPlotItem::RenderAntialiased, true );

    QPolygonF points;

    for(int i=0;i<vdX.size();i++)
        points << QPointF(vdX[i], vdY[i]);

    curve->setSamples( points );
    curve->setPen(QColor(iColorRGB));

    curve->attach( this );
    setAxisAutoScale(0);
    _zoomer->setZoomBase();
}
//////////////////////////////////////////////////////////////////////////
#else

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QWheelEvent>
#include <QGraphicsSceneWheelEvent>
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

    yMinL=-8.;
    yMaxL=-8.;

    //_bDrawXaxis=true;
    _bDrawYaxis=true;
    _bYLogAxis=false;
}
//////////////////////////////////////////////////////////////////////////
SimpleCurveWidget::~SimpleCurveWidget()
{ }
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addCurve(const vector<float>& vfX, const vector<float>& vfY,unsigned int iColorRGB)
{
    vector<double> vdX,vdY;
    for(unsigned int i=0;i<vfX.size();i++)
    {
        vdX.push_back((double)vfX[i]);
        vdY.push_back((double)vfY[i]);
    }

    addCurve(vdX,vdY,iColorRGB);
}
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
    _horizontalLines.push_back(0.);
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

    //compute range with horizonal lines
    for(unsigned int i=0;i<_horizontalLines.size();i++)
    {
        if(_horizontalLines[i]<yMin)
            yMin=_horizontalLines[i];

        if(_horizontalLines[i]>yMax)
            yMax=_horizontalLines[i];
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
    _horizontalLines.clear();
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
    for(unsigned int i=0;i<_horizontalLines.size();i++)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        double dY=_horizontalLines[i];
        _qs.addLine(xMin,-dY,xMax,-dY,penBlack);
    }

    if(_bDrawYaxis)
    {
        QPen penBlack(Qt::black);
        penBlack.setCosmetic(true);
        if(_bYLogAxis)
            _qs.addLine(0.,-yMaxL,0.,-yMinL ,penBlack);
        else
            _qs.addLine(0.,-yMax ,0.,-yMin,penBlack);
    }
}
//////////////////////////////////////////////////////////////////////////
void SimpleCurveWidget::addHorizontalLine(double dY)
{
    if(std::find(_horizontalLines.begin(),_horizontalLines.end(),dY)!=_horizontalLines.end())
        return;

    _horizontalLines.push_back(dY);
    compute_bounding_box();
}
//////////////////////////////////////////////////////////////////////////
#endif
