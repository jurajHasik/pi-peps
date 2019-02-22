#ifndef _ITENSOR_LAPACK_LINSYS_SOLVERS_H
#define _ITENSOR_LAPACK_LINSYS_SOLVERS_H

#include "p-ipeps/itensor-linsys-solvers.h"


namespace itensor {

//
//
// Forward declarations of fortran lapack routines
//
//
#ifdef LAPACK_REQUIRE_EXTERN
extern "C" {

// SOLVE LINEAR SYSTEM Ax = b for x (Direct Solver by LU Decomposition for General A)
void F77NAME(zgesv)(LAPACK_INT *n, LAPACK_INT *nrhs, LAPACK_COMPLEX *a, LAPACK_INT *lda,
                    LAPACK_INT *ipiv, LAPACK_COMPLEX *b, LAPACK_INT *ldb, 
                    LAPACK_INT *info);

void F77NAME(dgesv)(LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda,
                    LAPACK_INT *ipiv, double *b, LAPACK_INT *ldb, 
                    LAPACK_INT *info);

// SOLVE LINEAR SYSTEM Ax = b for x (Direct Solver by Cholesky Decomposition for Symmetric-Positive-Definite A)
void F77NAME(zposv)(char *uplo, LAPACK_INT *n, LAPACK_INT *nrhs, LAPACK_COMPLEX *a, LAPACK_INT *lda, 
                    LAPACK_COMPLEX *b, LAPACK_INT *ldb, LAPACK_INT *info);

void F77NAME(dposv)(char *uplo, LAPACK_INT *n, LAPACK_INT *nrhs, double * a, LAPACK_INT *lda, 
                    double * b, LAPACK_INT *ldb, LAPACK_INT *info);
} //extern "C"
#endif

//
// (d/z)gesv
//
// The routine solves the system of linear equations for X:
// A*X = B
// by LU decomposition
// where
//         A is a square matrix.
//         The columns of matrix B are individual right-hand sides.
//         The columns of X are the corresponding solutions.
// The matrix B is overwritten by X.
//
LAPACK_INT 
zgesv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              Cplx const * A,    // matrix A
              Cplx * b);        // matrix b

LAPACK_INT 
dgesv_wrapper(LAPACK_INT n,
              LAPACK_INT nrhs,
              LAPACK_REAL const * A,
              LAPACK_REAL * b);


LAPACK_INT 
zgesv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              Cplx const * A,    // matrix A
              Cplx * b)         // matrix b
{
    std::vector<LAPACK_COMPLEX> cpA;
    std::vector<LAPACK_COMPLEX> cpb;
    std::vector<LAPACK_INT> ipiv;

    //Copy A data into cpA
    cpA.resize(n*n);
    auto pA = reinterpret_cast<LAPACK_COMPLEX const*>(A);
    std::copy(pA,pA+n*n,cpA.data());

    // cpb.resize(n*nrhs);
    // auto pb = reinterpret_cast<LAPACK_COMPLEX const*>(b);
    // std::copy(pb,pb+n*nrhs,cpb.data());

    static_assert(sizeof(LAPACK_COMPLEX)==sizeof(Cplx),"LAPACK_COMPLEX and itensor::Cplx have different size");
    auto pb = reinterpret_cast<LAPACK_COMPLEX*>(b);

    ipiv.resize(n);

    // The leading dimension of the array b; 
    // ldb≥ max(1, n) for column major layout and ldb≥nrhs for row major layout.

    LAPACK_INT info = 0;
    F77NAME(zgesv)(&n,&nrhs,cpA.data(),&n,ipiv.data(), pb, &n,&info); // cpb.data()
    return info;
}

LAPACK_INT 
dgesv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              LAPACK_REAL const * A,    // matrix A
              LAPACK_REAL * b)         // matrix b
{
    std::vector<LAPACK_REAL> cpA;
    std::vector<LAPACK_REAL> cpb;
    std::vector<LAPACK_INT> ipiv;

    //Copy A data into cpA
    cpA.resize(n*n);
    auto pA = reinterpret_cast<LAPACK_REAL const*>(A);
    std::copy(pA,pA+n*n,cpA.data());

    // cpb.resize(n*nrhs);
    // auto pb = reinterpret_cast<LAPACK_REAL const*>(b);
    // std::copy(pb,pb+n*nrhs,cpb.data());

    ipiv.resize(n);

    // The leading dimension of the array b; 
    // ldb≥ max(1, n) for column major layout and ldb≥nrhs for row major layout.

    // LAPACK_REAL const * ptrA = A;
    // for (int i=0; i<n*n; i++) {
    //   std::cout<<"A["<< i <<"]"<< *ptrA << std::endl;
    //   ptrA++;
    // }
    // for (int i=0; i<n*n; i++) {
    //   std::cout<<"cpA["<< i <<"]"<< cpA[i] << std::endl;
    // }

    // LAPACK_REAL * ptrb = b;
    // for (int i=0; i<n*nrhs; i++) {
    //   std::cout<<"b["<< i <<"]"<< *ptrb << std::endl;
    //   ptrb++;
    // }

    LAPACK_INT info = 0;
    F77NAME(dgesv)(&n,&nrhs,cpA.data(),&n,ipiv.data(),b,&n,&info); // cpb.data()
    
    // std::cout<<"INFO: "<< info << std::endl;

    // ptrb = b;
    // for (int i=0; i<n*nrhs; i++) {
    //   std::cout<<"b["<< i <<"]"<< *ptrb << std::endl;
    //   ptrb++;
    // }

    return info;
}


struct CholeskySolver : LinSysSolver {
    
    void 
    solve(
      MatRefc<Real> const& A,
      VecRef<Real>  const& B, 
      VecRef<Real>  const& X,
      Args const& args) const override
    {  
        auto dbg    = args.getBool("dbg",false);
        if(dbg) std::cout<<"[CholeskySolver::solve<Real>] called"<<std::endl;

        auto ncA = const_cast<Real*>(A.data());
        auto pA = reinterpret_cast<LAPACK_REAL*>(ncA);

        LAPACK_INT Mr   = nrows(A);
        LAPACK_INT nrhs = 1;
        char uplo = 'U';
        LAPACK_INT info;
        F77NAME(dposv)(&uplo,&Mr,&nrhs,pA,&Mr,B.data(),&Mr,&info);

        if (info == 0) std::copy(B.data(),B.data()+Mr,X.data());
        else throw std::runtime_error("CholeskySolver: error info: "+std::to_string(info));
    }

    void 
    solve(
      MatRefc<Cplx>  const& A,
      VecRef<Cplx>  const& B, 
      VecRef<Cplx>  const& X, 
      Args const& args) const override
    {  
        auto dbg    = args.getBool("dbg",false);
        if(dbg) std::cout<<"[CholeskySolver::solve<Cplx>] called"<<std::endl;

        LAPACK_INT Mr   = nrows(A);
        LAPACK_INT nrhs = 1;
        char uplo = 'U';
        LAPACK_INT info;

        auto ncA = const_cast<Cplx*>(A.data());

        auto pA = reinterpret_cast<LAPACK_COMPLEX*>(ncA);
        auto pB = reinterpret_cast<LAPACK_COMPLEX*>(B.data());
        
        F77NAME(zposv)(&uplo,&Mr,&nrhs,pA,&Mr,pB,&Mr,&info);

        if(info == 0) std::copy(B.data(),B.data()+Mr,X.data());
        else throw std::runtime_error("CholeskySolver: error info: "+std::to_string(info));
    }
};

    
} //namespace detail

#endif