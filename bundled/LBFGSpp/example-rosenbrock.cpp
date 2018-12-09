#include "itensor/all.h"
#include <iostream>
#include "include/LBFGS.h"

// using Eigen::VectorXf;
// using Eigen::MatrixXf;
typedef itensor::Vec<itensor::Real> VectorXf;
typedef itensor::Mat<itensor::Real> MatrixXf;
using namespace LBFGSpp;

class Rosenbrock
{
private:
    int n;
public:
    Rosenbrock(int n_) : n(n_) {}
    double operator()(const VectorXf& x, VectorXf& grad)
    {
        double fx = 0.0;
        for(int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
        }
        return fx;
    }
};

int main()
{
    const int n = 100;
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    LBFGSSolver<double,itensor::Real> solver(param);
    Rosenbrock fun(n);

    VectorXf x = itensor::randomVec(n);
    x *= 0;

    double fx;
    int niter = solver.minimize(fun, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    return 0;
}
