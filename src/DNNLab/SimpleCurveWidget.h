#ifndef SimpleCurve_
#define SimpleCurve_

#include <QGraphicsScene>
#include <QGraphicsView>
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

class SimpleCurveWidget: public QGraphicsView
{
Q_OBJECT
public:
    SimpleCurveWidget();
    virtual ~SimpleCurveWidget();

    void addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB=0xFFFFFF);
    void clear();

    void addXAxis();
    void addYAxis();
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
    bool _bDrawXaxis,_bDrawYaxis;
    QGraphicsScene _qs;
};

#endif
