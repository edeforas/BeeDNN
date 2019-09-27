#ifndef SimpleCurve_
#define SimpleCurve_

#include <vector>
using namespace std;

class CurveData
{
public:
    vector<double> vdX;
    vector<double> vdY;

    double xMin,xMax,yMin,yMax;

    unsigned int _iColorRGB;
};

#ifdef USE_QWT
#include <qwt_plot.h>
#include <qwt_plot_zoomer.h>

class SimpleCurveWidget : public QwtPlot
{
Q_OBJECT
public:

    SimpleCurveWidget();
    virtual ~SimpleCurveWidget();
    void clear();

    void addXAxis();
    void addYAxis();
    void addHorizontalLine(double dY);

    void setYLogAxis(bool bSetLogAxis);

    void addCurve(const vector<float>& vfX, const vector<float>& vfY,unsigned int iColorRGB=0xFFFFFF);
    void addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB=0xFFFFFF);

private:
    QwtPlotZoomer* _zoomer;

};


#else

#include <QGraphicsScene>
#include <QGraphicsView>

class SimpleCurveWidget: public QGraphicsView
{
Q_OBJECT
public:


    SimpleCurveWidget();
    virtual ~SimpleCurveWidget();
    void clear();

    void addXAxis();
    void addYAxis();
    void addHorizontalLine(double dY);

    void addCurve(const vector<float>& vfX, const vector<float>& vfY,unsigned int iColorRGB=0xFFFFFF);
    void addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB=0xFFFFFF);

    void setYLogAxis(bool bSetLogAxis);

public slots:
    void wheelEvent(QWheelEvent* event);

private:
    void compute_bounding_box();
    void replot_curve(int iCurve);
    void replot_axis();
    void replot_all();

    vector<CurveData> _vCurves;
    double xMin,xMax,yMin,yMax;

    double yMinL,yMaxL;

    bool _bYLogAxis;
    //bool _bDrawXaxis;
    bool _bDrawYaxis;

    vector<double> _horizontalLines;

    QGraphicsScene _qs;
};

#endif
#endif
