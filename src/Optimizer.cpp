#include "Optimizer.h"
#include "Layer.h"

// Network optimizer as in:
// http://cs231n.github.io/neural-networks-3/#update

//////////////////////////////////////////////////////////
Optimizer::Optimizer()
{}
//////////////////////////////////////////////////////////
Optimizer::~Optimizer()
{}
//////////////////////////////////////////////////////////
class OptimizerSGD : public Optimizer
{
public:	
    OptimizerSGD()
    {}

    ~OptimizerSGD() override
    {}

    virtual void init(const Layer& l) override
    { (void)l; }

    virtual void optimize(MatrixFloat& weight,const MatrixFloat& mDx) override
    {
        // Vanilla update
        //	x += -learning_rate * dx
        weight -=  mDx * fLearningRate ;
    }
};
//////////////////////////////////////////////////////////
class OptimizerMomentum : public Optimizer
{
public:
    OptimizerMomentum()
    {}

    ~OptimizerMomentum() override
    {}

    virtual void init(const Layer& l) override
    {
        (void)l;
        _v.resize(0,0);
    }

    virtual void optimize(MatrixFloat& weight, const MatrixFloat& mDx) override
    {
        // init _V if needed
        if (_v.size() == 0)
        {
            _v.resize(weight.rows(), weight.cols());
            _v.setZero();
        }

        // v = mu * v - learning_rate * dx // integrate velocity
        // x += v // integrate position
        _v = _v*fMomentum - mDx*fLearningRate;

        weight += _v;
    }
private:
    MatrixFloat _v;
};
//////////////////////////////////////////////////////////
class OptimizerNesterov : public Optimizer
{
public:
    OptimizerNesterov()
    {}

    ~OptimizerNesterov() override
    {}

    virtual void init(const Layer& l) override
    {
        (void)l;
        _v.resize(0, 0);
    }

    virtual void optimize(MatrixFloat& weight, const MatrixFloat& mDx) override
    {
        // init _V if needed
        if (_v.size() == 0)
        {
            _v.resize(weight.rows(), weight.cols());
            _v.setZero();
        }

        // v_prev = v # back this up
        // v = mu * v - learning_rate * dx # velocity update stays the same
        // x += -mu * v_prev + (1 + mu) * v # position update changes form
        _v_prev = _v;
        _v = _v*fMomentum - mDx*fLearningRate ;
        weight += _v_prev*(-fMomentum) + _v*(1.f + fMomentum) ;
    }
private:
    MatrixFloat _v, _v_prev;
};
//////////////////////////////////////////////////////////////////////////////
class OptimizerAdagrad : public Optimizer
{
public:
    OptimizerAdagrad()
    {}

    ~OptimizerAdagrad() override
    {}

    virtual void init(const Layer& l) override
    {
        (void)l;
        _cache.resize(0,0);
    }

    virtual void optimize(MatrixFloat& weight, const MatrixFloat& mDx) override
    {
        // init _cache if needed
        if (_cache.size() == 0)
        {
            _cache.resize(mDx.rows(), mDx.cols());
            _cache.setZero();
        }

        // cache += dx**2
        // x += - learning_rate * dx / (np.sqrt(cache) + eps)

        _cache +=mDx.cwiseAbs2();
        weight += mDx.cwiseQuotient(_cache.cwiseSqrt().cwiseMax(1.e-8f))*(-fLearningRate);
    }
private:
    MatrixFloat _cache;
};
//////////////////////////////////////////////////////////
Optimizer* get_optimizer(const string& sOptimizer)
{
    if (sOptimizer == "SGD")
        return new OptimizerSGD;

    if (sOptimizer == "Momentum")
        return new OptimizerMomentum;

    if (sOptimizer == "Nesterov")
        return new OptimizerNesterov;

    if (sOptimizer == "Adagrad")
        return new OptimizerAdagrad;

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_optimizers_available(vector<string>& vsOptimizers)
{
    vsOptimizers.clear();

    vsOptimizers.push_back("SGD");
    vsOptimizers.push_back("Momentum");
    vsOptimizers.push_back("Nesterov");
}
//////////////////////////////////////////////////////////////////////////////
