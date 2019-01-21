#ifndef Net_
#define Net_

#include <vector>
using namespace std;

class Layer;
#include "Matrix.h"

class Net
{
public:
    Net();
    virtual ~Net();

	void clear();
    void add(Layer *l); //take ownership of layer todo
    const vector<Layer*> layers() const;
    Layer* layer(size_t iLayer);

    void forward(const MatrixFloat& mIn,MatrixFloat& mOut) const;
    void classify(const MatrixFloat& mIn,MatrixFloat& mOut) const;

private:
    vector<Layer*> _layers;
};

#endif
