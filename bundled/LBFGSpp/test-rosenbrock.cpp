#include "itensor/all.h"
#include <iostream>
#include "include/LBFGS.h"

using namespace LBFGSpp;
    
typedef itensor::Vec<itensor::Real> VectorXd;

class Rosenbrock
{
private:
    int n;
public:
    Rosenbrock(int n_) : n(n_) {}
    float operator()(const VectorXd& x, VectorXd& grad)
    {
        float fx = 0.0;
        for(int i = 0; i < n; i += 2)
        {
            float t1 = 1.0 - x[i];
            float t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        assert( ! std::isnan(fx) );
        return fx;
    }
};

int main()
{
    LBFGSParam<double> param;
    LBFGSSolver<double, itensor::Real> solver(param);

    for( int n=2; n <= 16; n += 2 )
    {
        std::cout << "n = " << n << std::endl;
        Rosenbrock fun(n);
        for( int test=0; test < 1024; test++ )
        {
            VectorXd x = itensor::randomVec(n);
            double fx;
            int niter = solver.minimize(fun, x, fx);

            assert( ( (x.array() - 1.0).abs() < 1e-4 ).all() );
        }
    }

    return 0;
}
