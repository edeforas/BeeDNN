#ifndef SimpleCurve_
#define SimpleCurve_

#include <QGraphicsScene>
#include <vector>
using namespace std;

class SimpleCurve: public QGraphicsScene
{
//	QOBJECT
	
public:
	SimpleCurve();
	virtual ~SimpleCurve();
	
    void addCurve(const vector<double>& vdX, const vector<double>& vdY,Qt::GlobalColor=Qt::black);

    virtual void wheelEvent(QGraphicsSceneWheelEvent* wheelEvent);

};

#endif
