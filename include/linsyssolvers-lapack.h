#ifndef _ITENSOR_LAPACK_LINSYS_SOLVERS_H
#define _ITENSOR_LAPACK_LINSYS_SOLVERS_H

#include "itensor-linsys-solvers.h"

//
// Headers and typedefs
//

// //
// //
// // Generic Linux LAPACK
// //
// //
// #ifdef PLATFORM_lapack

// #define LAPACK_REQUIRE_EXTERN

// namespace itensor {
//     using LAPACK_INT = int;
//     using LAPACK_REAL = double;
//     typedef struct
//     {
//     LAPACK_REAL real, imag;
//     } LAPACK_COMPLEX;
// }
// #elif defined PLATFORM_openblas

// #define ITENSOR_USE_CBLAS

// #include "cblas.h"
// #include "lapacke.h"
// #undef I //lapacke.h includes complex.h which defined an `I` macro
//          //that can cause problems, so best to undefine it

// namespace itensor {
// using LAPACK_INT = lapack_int;
// using LAPACK_REAL = double;
// using LAPACK_COMPLEX = lapack_complex_double;

// inline LAPACK_REAL& 
// realRef(LAPACK_COMPLEX & z) 
//     { 
//     auto* p = reinterpret_cast<double*>(&z);
//     return p[0];
//     }

// inline LAPACK_REAL& 
// imagRef(LAPACK_COMPLEX & z) 
//     { 
//     auto* p = reinterpret_cast<double*>(&z);
//     return p[1];
//     }
// }

// //
// //
// // Apple Accelerate/vecLib
// //
// //
// #elif defined PLATFORM_macos

// #define ITENSOR_USE_CBLAS
// //#define ITENSOR_USE_ZGEMM

// #include <Accelerate/Accelerate.h>
//     namespace itensor {
//     using LAPACK_INT = __CLPK_integer;
//     using LAPACK_REAL = __CLPK_doublereal;
//     using LAPACK_COMPLEX = __CLPK_doublecomplex;

//     inline LAPACK_REAL& 
//     realRef(LAPACK_COMPLEX & z) { return z.r; }

//     inline LAPACK_REAL& 
//     imagRef(LAPACK_COMPLEX & z) { return z.i; }
//     }

// //
// //
// // Intel MKL
// //
// //
// #elif defined PLATFORM_mkl

// #define ITENSOR_USE_CBLAS
// #define ITENSOR_USE_ZGEMM

// #include "mkl_cblas.h"
// #include "mkl_lapack.h"
//     namespace itensor {
//     using LAPACK_INT = MKL_INT;
//     using LAPACK_REAL = double;
//     using LAPACK_COMPLEX = MKL_Complex16;

//     inline LAPACK_REAL& 
//     realRef(LAPACK_COMPLEX & z) { return z.real; }

//     inline LAPACK_REAL& 
//     imagRef(LAPACK_COMPLEX & z) { return z.imag; }
//     }

// //
// //
// // AMD ACML
// //
// //
// #elif defined PLATFORM_acml

// #define LAPACK_REQUIRE_EXTERN
// //#include "acml.h"
//     namespace itensor {
//     using LAPACK_INT = int;
//     using LAPACK_REAL = double;
//     typedef struct
//     {
//     LAPACK_REAL real, imag;
//     } LAPACK_COMPLEX;

//     inline LAPACK_REAL& 
//     realRef(LAPACK_COMPLEX & z) { return z.real; }

//     inline LAPACK_REAL& 
//     imagRef(LAPACK_COMPLEX & z) { return z.imag; }
//     }

// #endif // different PLATFORM types



// #ifdef FORTRAN_NO_TRAILING_UNDERSCORE
// #define F77NAME(x) x
// #else
// #define F77NAME(x) x##_
// #endif


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

//
// (d/z)posv
//
// The routine solves the system of linear equations for X:
// A*X = B
// by Cholesky decomposition
// where
//         A is a square matrix Symmetric Positive-Definite
//         The columns of matrix B are individual right-hand sides.
//         The columns of X are the corresponding solutions.
// The matrix B is overwritten by X.
//
LAPACK_INT 
zposv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              Cplx const * A,   // matrix A
              Cplx * b);        // matrix b

LAPACK_INT 
dposv_wrapper(LAPACK_INT n,
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

LAPACK_INT 
zposv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              Cplx const * A,    // matrix A
              Cplx * b)         // matrix b
{
    std::vector<LAPACK_COMPLEX> cpA;
    std::vector<LAPACK_COMPLEX> cpb;
    std::vector<LAPACK_INT> ipiv;

    char uplo = 'U';
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
    F77NAME(zposv)(&uplo,&n,&nrhs,cpA.data(),&n,pb,&n,&info); // cpb.data()
    return info;
}

LAPACK_INT 
dposv_wrapper(LAPACK_INT n,     // rank of square matrix A    
              LAPACK_INT nrhs,  // number of right hand sides b = matrix n x nrhs
              LAPACK_REAL const * A,   // matrix A
              LAPACK_REAL * b)         // matrix b
{
    std::vector<LAPACK_REAL> cpA;
    std::vector<LAPACK_REAL> cpb;
    std::vector<LAPACK_INT> ipiv;

    char uplo = 'U';
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
    F77NAME(dposv)(&uplo,&n,&nrhs,cpA.data(),&n,b,&n,&info); // cpb.data()
    
    // std::cout<<"INFO: "<< info << std::endl;

    // ptrb = b;
    // for (int i=0; i<n*nrhs; i++) {
    //   std::cout<<"b["<< i <<"]"<< *ptrb << std::endl;
    //   ptrb++;
    // }

    return info;
}

struct CholeskySolver : LinSysSolver {
    
    virtual void 
    solve(
      MatRefc<Real> const& A,
      VecRef<Real>  const& B, 
      VecRef<Real>  const& X,
      Args const& args) 
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

    virtual void 
    solve(
      MatRefc<Cplx>  const& A,
      VecRef<Cplx>  const& B, 
      VecRef<Cplx>  const& X, 
      Args const& args) 
    {  
        auto dbg    = args.getBool("dbg",false);
        if(dbg) std::cout<<"[CholeskySolver::solve<Cplx>] called"<<std::endl;

        LAPACK_INT Mr   = nrows(A);
        LAPACK_INT nrhs = 1;
        char uplo = 'U';
        LAPACK_INT info;

        auto ncA = const_cast<Cplx*>(A.data());
        // A is in row major order, hence if read in column-major order it
        // represents A**T. Performing conjugate we recover A**T**C = A
        auto t_ncA = ncA;
        for (int i=0; i<Mr*Mr; i++, t_ncA++) *t_ncA = std::conj(*t_ncA);

        auto pA = reinterpret_cast<LAPACK_COMPLEX*>(ncA);
        auto pB = reinterpret_cast<LAPACK_COMPLEX*>(B.data());
        
        F77NAME(zposv)(&uplo,&Mr,&nrhs,pA,&Mr,pB,&Mr,&info);

        if(info == 0) std::copy(B.data(),B.data()+Mr,X.data());
        else throw std::runtime_error("CholeskySolver: error info: "+std::to_string(info));
    }
};

    
} //namespace detail

#endif