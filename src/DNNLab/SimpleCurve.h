#ifndef SimpleCurve_
#define SimpleCurve_

#include <QGraphicsScene>
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

class SimpleCurve: public QGraphicsScene
{
    //	QOBJECT

public:
    SimpleCurve();
    virtual ~SimpleCurve();

    void addCurve(const vector<double>& vdX, const vector<double>& vdY,unsigned int iColorRGB=0xFFFFFF);
    void clear();

    void addXAxis();
    void addYAxis();
    void setYLogAxis(bool bSetLogAxis);

    virtual void wheelEvent(QGraphicsSceneWheelEvent* wheelEvent);
private:
    void compute_bounding_box();
    void replot_curve(int iCurve);
    void replot_axis();
    void replot_all();

    vector<CurveData> _vCurves;
    double xMin,xMax,yMin,yMax;
    bool _bYLogAxis;
    bool _bDrawXaxis,_bDrawYaxis;
};

#endif
