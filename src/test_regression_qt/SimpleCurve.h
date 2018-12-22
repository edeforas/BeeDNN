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

    int xMin,xMax,yMin,yMax;

    int _iColor;
};

class SimpleCurve: public QGraphicsScene
{
    //	QOBJECT

public:
    SimpleCurve();
    virtual ~SimpleCurve();

    void addCurve(const vector<double>& vdX, const vector<double>& vdY,Qt::GlobalColor=Qt::black);

    void addXAxis();
    void addYAxis();

    virtual void wheelEvent(QGraphicsSceneWheelEvent* wheelEvent);
private:
    vector<CurveData> _vCurves;
};

#endif
