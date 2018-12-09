// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef _ITENSOR_PORT_LBFGS_H
#define _ITENSOR_PORT_LBFGS_H

#include "itensor/all.h"
#include "LBFGS/Param.h"
#include "LBFGS/LineSearch.h"


namespace LBFGSpp {


///
/// LBFGS solver for unconstrained numerical optimization
///
template <typename Scalar, typename Field>
class LBFGSSolver
{
private:
    // typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    // typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    // typedef Eigen::Map<Vector> MapVec;
    typedef itensor::Vec<Field> Vector;
    typedef itensor::Mat<Field> Matrix;
    typedef itensor::Vec<Field> MapVec;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    Matrix                    m_s;      // History of the s vectors
    Matrix                    m_y;      // History of the y vectors
    Vector                    m_ys;     // History of the s'y values
    Vector                    m_alpha;  // History of the step lengths
    Vector                    m_fx;     // History of the objective function values
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // New gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction

    inline void reset(int n)
    {
        const int m = m_param.m;
        resize(m_s, n, m);
        resize(m_y, n, m);
        resize(m_ys, m);
        resize(m_alpha, m);
        resize(m_xp, n);
        resize(m_grad, n);
        resize(m_gradp, n);
        resize(m_drt, n);
        if(m_param.past > 0)
            resize(m_fx, m_param.past);
    }

public:
    ///
    /// Constructor for LBFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using LBFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx)
    {
        const int n = x.size();
        const int fpast = m_param.past;
        reset(n);

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        Scalar xnorm = norm(x);
        Scalar gnorm = norm(m_grad);
        if(fpast > 0)
            m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer
        if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
        {
            return 1;
        }

        // Initial direction
        m_drt = (-1.0) * m_grad;
        // Initial step
        Scalar step = Scalar(1.0) / norm(m_drt);

        int k = 1;
        int end = 0;
        for( ; ; )
        {
            std::cout<<"----- LineSearch "<< k <<" -----"<< std::endl;
            // Save the curent x and gradient
            m_xp = x;
            m_gradp = m_grad;

            // Line search to update x, fx and gradient
            LineSearch<Scalar, Field>::Backtracking(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

            // New x norm and gradient norm
            xnorm = norm(x);
            gnorm = norm(m_grad);

            // Convergence test -- gradient
            if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
            {
                return k;
            }
            // Convergence test -- objective function value
            if(fpast > 0)
            {
                if(k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
                    return k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            MapVec svec(column(m_s,end)); // MapVec svec(&m_s(0, end), n);
            MapVec yvec(column(m_y,end)); // MapVec yvec(&m_y(0, end), n);
            svec = x - m_xp;
            yvec = m_grad - m_gradp;

            // ys = y's = 1/rho
            // yy = y'y
            Scalar ys = yvec * svec;
            Scalar yy = yvec * yvec; // norm^2
            m_ys[end] = ys;

            // Recursive formula to compute d = -H * g
            m_drt = (-1.0)*m_grad;
            int bound = std::min(m_param.m, k);
            end = (end + 1) % m_param.m;
            int j = end;
            for(int i = 0; i < bound; i++)
            {
                j = (j + m_param.m - 1) % m_param.m;
                MapVec sj(column(m_s,j)); // MapVec sj(&m_s(0, j), n);
                MapVec yj(column(m_y,j)); // MapVec yj(&m_y(0, j), n);
                m_alpha[j] = sj * m_drt / m_ys[j];
                m_drt -= m_alpha[j] * yj;
            }

            m_drt *= (ys / yy);

            for(int i = 0; i < bound; i++)
            {
                MapVec sj(column(m_s,j)); // MapVec sj(&m_s(0, j), n);
                MapVec yj(column(m_y,j)); // MapVec yj(&m_y(0, j), n);
                Scalar beta = yj * m_drt / m_ys[j];
                m_drt += (m_alpha[j] - beta) * sj;
                j = (j + 1) % m_param.m;
            }

            // step = 1.0 as initial guess
            step = Scalar(1.0);
            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGS_H
