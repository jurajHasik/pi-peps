#ifndef _PEPS_LAPACK_SVD_SOLVERS_H
#define _PEPS_LAPACK_SVD_SOLVERS_H

#include "itensor-svd-solvers.h"

namespace itensor {

//
//
// Forward declarations of fortran lapack routines
//
//
#ifdef LAPACK_REQUIRE_EXTERN
extern "C" {

#ifdef PLATFORM_acml
void F77NAME(dgesdd)(char *jobz, LAPACK_INT *m, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *s, 
             double *u, LAPACK_INT *ldu, double *vt, LAPACK_INT *ldvt, 
             double *work, LAPACK_INT *lwork, LAPACK_INT *iwork, LAPACK_INT *info, int jobz_len);
#else
void F77NAME(dgesdd)(char *jobz, LAPACK_INT *m, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *s, 
             double *u, LAPACK_INT *ldu, double *vt, LAPACK_INT *ldvt, 
             double *work, LAPACK_INT *lwork, LAPACK_INT *iwork, LAPACK_INT *info);
#endif

} //extern "C"
#endif


void 
dgesdd_wrapper(char jobz,           //char* specifying how much of U, V to compute
                                    //choosing *jobz=='S' computes min(m,n) cols of U, V
               LAPACK_INT m,        //number of rows of input matrix *A
               LAPACK_INT n,        //number of cols of input matrix *A
               LAPACK_REAL *A,      //contents of input matrix A
               LAPACK_REAL *s,      //on return, singular values of A
               LAPACK_REAL *u,           //on return, unitary matrix U
               LAPACK_REAL *vt,          //on return, unitary matrix V transpose
               LAPACK_INT *info)
    {
    auto pA = reinterpret_cast<LAPACK_REAL*>(A);
    std::vector<LAPACK_REAL> work;
    std::vector<LAPACK_INT> iwork;
    LAPACK_INT l = std::min(m,n),
               g = std::max(m,n);
    LAPACK_INT lwork = l*(6 + 4*l) + g;
    work.resize(lwork);
    iwork.resize(8*l);
#ifdef PLATFORM_acml
    LAPACK_INT jobz_len = 1;
    F77NAME(dgesdd)(&jobz,&m,&n,pA,&m,s,u,&m,vt,&l,work.data(),&lwork,iwork.data(),info,jobz_len);
#else
    F77NAME(dgesdd)(&jobz,&m,&n,pA,&m,s,u,&m,vt,&l,work.data(),&lwork,iwork.data(),info);
#endif
    }


void
zgesdd_wrapper(char jobz,           //char* specifying how much of U, V to compute
               LAPACK_INT m,        //number of rows of input matrix *A
               LAPACK_INT n,        //number of cols of input matrix *A
               Cplx *A,    //contents of input matrix A
               LAPACK_REAL *s,
               Cplx *u,    //on return, unitary matrix U
               Cplx *vt,   //on return, unitary matrix V transpose
               LAPACK_INT *info)
{
    // auto ncA = const_cast<Cplx*>(A); 
    auto pA = reinterpret_cast<LAPACK_COMPLEX*>(A);
    auto pU = reinterpret_cast<LAPACK_COMPLEX*>(u);
    auto pVt = reinterpret_cast<LAPACK_COMPLEX*>(vt);
    std::vector<LAPACK_COMPLEX> work;
    std::vector<LAPACK_REAL> rwork;
    std::vector<LAPACK_INT> iwork;
    LAPACK_INT l = std::min(m,n),
               g = std::max(m,n);
 	LAPACK_INT lwork = l*l+2*l+g+100;
    work.resize(lwork);
    rwork.resize(5*l*(1+l));
    iwork.resize(8*l);
#ifdef PLATFORM_acml
    LAPACK_INT jobz_len = 1;
    F77NAME(zgesdd)(&jobz,&m,&n,pA,&m,s,pU,&m,pVt,&l,work.data(),&lwork,rwork.data(),iwork.data(),info,jobz_len);
#else
    F77NAME(zgesdd)(&jobz,&m,&n,pA,&m,s,pU,&m,pVt,&l,work.data(),&lwork,rwork.data(),iwork.data(),info);
#endif
    }



struct GESDDSolver : SvdSolver {
	
  	void 
  	solve(
      MatRefc<Real> const& M,
      MatRef<Real>  const& U, 
      VectorRef     const& D, 
      MatRef<Real>  const& V,
      Args const& args) 
  	{
  		auto Mr = nrows(M);
        auto Mc = ncols(M);
    if(Mr > Mc)
        {
        solve(transpose(M),V,D,U,args);
        conjugate(V);
        conjugate(U);
#ifdef CHKSVD
        checksvd(M,U,D,V);
#endif
        return;
        }
#ifdef DEBUG
    if(!(nrows(U)==Mr && ncols(U)==Mr)) 
        throw std::runtime_error("SVD (ref version), wrong size of U");
    if(!(nrows(V)==Mc && ncols(V)==Mr)) 
        throw std::runtime_error("SVD (ref version), wrong size of V");
    if(D.size()!=Mr)
        throw std::runtime_error("SVD (ref version), wrong size of D");
#endif
    
    auto pA = M.data();
    std::vector<Real> cpA;
    cpA.resize(Mr*Mc);
    
    // LAPACK ?gesdd will read input matrix in column-major order. If we actually
    // want to perform SVD of M**T where M is stored in column-major, we have to pass
    // M**T stored in column-major. Copy of inpput matrix has to be done in any case, 
    // since input matrix is destroyed in ?gesdd
    if(isTransposed(M)) {
        for (unsigned int i=0; i<cpA.size(); i++, pA++) cpA[(i%Mc)*Mr + i/Mc] = *pA;
    } else {
        std::copy(pA,pA+Mr*Mc,cpA.data());
    }
    int info;
    dgesdd_wrapper('S', //char* specifying how much of U, V to compute
                        //choosing *jobz=='S' computes min(m,n) cols of U, V
        Mr,             //number of rows of input matrix *A
        Mc,             //number of cols of input matrix *A
        cpA.data(),     
        D.data(),       //on return, singular values of A
        U.data(),       //on return, unitary matrix U
        V.data(),       //on return, unitary matrix V transpose
        &info
    );
    // from ?gesdd:
    // if JOBZ = 'S', V contains the first min(M=Mr,N=Mc) rows of
    // V**T (the right singular vectors, stored rowwise); 
    // Lapack stores V in column-major format, while the return of this function
    // expects row-major format of V, hence the V is reordered accordingly
    auto ncV = const_cast<Real*>(V.data()); 
    auto pV  = reinterpret_cast<Real*>(ncV);
    int l = std::min(Mr,Mc);
    std::vector<Real> vt(l*Mc);
    std::copy(V.data(), V.data()+l*Mc, vt.data());
    for (unsigned int i=0; i<vt.size(); i++, pV++) *pV = vt[(i%Mc)*l + i/Mc];
    
#ifdef CHKSVD
	checksvd(M,U,D,V);
#endif
    return;

  	}

  	
  	void 
  	solve(
      MatRefc<Cplx> const& M,
      MatRef<Cplx>  const& U, 
      VectorRef     const& D, 
      MatRef<Cplx>  const& V,
      Args const& args) {

  		auto Mr = nrows(M); 
        auto Mc = ncols(M);
    // std::cout<<"R: "<< Mr <<" Mc: "<< Mc << " M**T: "<< isTransposed(M) <<std::endl;
    if(Mr > Mc)
        {
        solve(transpose(M),V,D,U,args);
        conjugate(V);
        conjugate(U);
#ifdef CHKSVD
        checksvd(M,U,D,V);
#endif
        return;
        }
#ifdef DEBUG
    if(!(nrows(U)==Mr && ncols(U)==Mr)) 
        throw std::runtime_error("SVD (ref version), wrong size of U");
    if(!(nrows(V)==Mc && ncols(V)==Mr)) 
        throw std::runtime_error("SVD (ref version), wrong size of V");
    if(D.size()!=Mr)
        throw std::runtime_error("SVD (ref version), wrong size of D");
#endif
    auto pA = M.data();
    std::vector<Cplx> cpA;
    cpA.resize(Mr*Mc);
    
    if(isTransposed(M)) {
        for (unsigned int i=0; i<cpA.size(); i++, pA++) cpA[(i%Mc)*Mr + i/Mc] = *pA;
    } else {
        std::copy(pA,pA+Mr*Mc,cpA.data());
    }    
    int info;
    zgesdd_wrapper('S', //char* specifying how much of U, V to compute
                        //choosing *jobz=='S' computes min(m,n) cols of U, V
        Mr,             //number of rows of input matrix *A
        Mc,             //number of cols of input matrix *A
        cpA.data(),
        D.data(),       //on return, singular values of A
        U.data(),       //on return, unitary matrix U
        V.data(),       //on return, unitary matrix V transpose
        &info
    );
    // In addition to column-major to row-major reordering, also complex conjugate
    // is taken
    auto ncV = const_cast<Cplx*>(V.data()); 
    auto pV  = reinterpret_cast<Cplx*>(ncV);
    int l = std::min(Mr,Mc);
    std::vector<Cplx> vt(l*Mc); 
    std::copy(V.data(), V.data()+l*Mc, vt.data());
    for (auto & e : vt) e = std::conj(e);
    for (unsigned int i=0; i<vt.size(); i++, pV++) *pV = vt[(i%Mc)*l + i/Mc];
#ifdef CHKSVD
    checksvd(M,U,D,V);
#endif
    return;
  	}

    static std::unique_ptr<GESDDSolver> create() {
      return std::unique_ptr<GESDDSolver>(new GESDDSolver());
    }

};

} // namespace itensor

#endif