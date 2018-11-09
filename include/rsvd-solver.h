#ifndef _ITENSOR_CUSTOM_SOLVERS_RSVD_H
#define _ITENSOR_CUSTOM_SOLVERS_RSVD_H

#include "itensor-svd-solvers.h"
#include "rank_revealing_algorithms_intel_mkl.h"

namespace itensor {

struct RsvdSolver : SvdSolver {

	// template<typename T>
 //  	void 
 //  	solve(
 //      MatRefc<T> const& M,
 //      MatRef<T>  const& U, 
 //      VectorRef  const& D, 
 //      MatRef<T>  const& V,
 //      Args const& args);

	
  	void 
  	solve(
      MatRefc<Real> const& M,
      MatRef<Real>  const& U, 
      VectorRef  const& D, 
      MatRef<Real>  const& V,
      Args const& args);

  	
  	void 
  	solve(
      MatRefc<Cplx> const& M,
      MatRef<Cplx>  const& U, 
      VectorRef  const& D, 
      MatRef<Cplx>  const& V,
      Args const& args);


};

} // namespace itensor

#endif