#ifndef _ITENSOR_CUSTOM_SOLVERS_RSVD_H
#define _ITENSOR_CUSTOM_SOLVERS_RSVD_H

#include "peps_config.h"

#ifdef PEPS_WITH_RSVD

#include "itensor-svd-solvers.h"
#include "rank_revealing_algorithms_intel_mkl.h"

namespace itensor {

struct RsvdSolver : SvdSolver {
	
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

    static std::unique_ptr<RsvdSolver> create();
};

} // namespace itensor

#endif // PEPS_WITH_RSVD

#endif
