#include "p-ipeps/config.h"
#include "p-ipeps/config.h"

#ifdef PEPS_WITH_RSVD

#include "p-ipeps/rsvd-solver.h"

namespace itensor {

//template<>
void
RsvdSolver::solve(
      MatRefc<Real> const& M,
      MatRef<Real>  const& U, 
      VectorRef     const& D, 
      MatRef<Real>  const& V,
      Args const& args
	)
{
	//std::cout<<"[RsvdSolver::solve<Real>] called"<<std::endl;

	auto Mr = nrows(M), 
         Mc = ncols(M);

    if(Mr > Mc)
        {
        solve(transpose(M),V,D,U,args);
        conjugate(V);
        conjugate(U);

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

    auto maxm         = args.getInt("Maxm",Mr);
	auto rsvd_power   = args.getInt("rsvd_power",2);
	auto rsvd_reortho = args.getInt("rsvd_reortho",1);
	auto rsvd_oversampling= args.getInt("rsvd_oversampling",10);

	// int64_t frank;
	// int64_t k    = maxm; //70; // rank we want
	// int64_t p    = 0; //20; // oversampling
	// int64_t q    = 2; //3; // power scheme
	// int64_t s    = 1; // re-rotho for power scheme
	// int64_t vnum = 1; // scheme to use
	int frank;
	int k    = maxm; //70; // rank we want
	int p    = rsvd_oversampling; //20; // oversampling
	int q    = rsvd_power; //3; // power scheme
	int s    = rsvd_reortho; // re-rotho for power scheme
	int vnum = 1; // scheme to use

	// we never ask for computing more singular values than the rank of the matrix
	if ( std::min(Mr,Mc) <= k+p ) {
		k = std::min(Mr,Mc);
		p = 0;
	}

	auto ncM = const_cast<double*>(M.data()); 
	auto pM  = reinterpret_cast<double*>(ncM);

	mat *wM = matrix_wrap(pM,Mr,Mc);
	mat *wU = matrix_wrap(U.data(),Mr,std::min(Mr,Mc));
	mat *wV = matrix_wrap(V.data(),Mc,std::min(Mr,Mc));
	//        mat *wU, *wV;
	mat *tS;

	//std::cout << "rsvd called"<< std::endl;
	//low_rank_svd_rand_decomp_fixed_rank(wM, k, p, vnum, q, s, &frank, &wU, &tS, &wV);
	low_rank_svd_rand_decomp_fixed_rank_byRef(wM, k, p, vnum, q, s, &frank, &wU, &tS, &wV);

	auto ncD = const_cast<double*>(D.data()); 
	auto pD  = reinterpret_cast<double*>(ncD);

	for(int ii=0; ii<k; ii++) {
	    *pD = matrix_get_element(tS, ii, ii);
	    pD++;
	}

#ifdef CHKSVD
    checksvd(M,U,D,V);
#endif
}

// No complex implementation fall back to ITensor
//template<>
void
RsvdSolver::solve(
      MatRefc<Cplx> const& M,
      MatRef<Cplx>  const& U, 
      VectorRef     const& D, 
      MatRef<Cplx>  const& V,
      Args const& args
	)
{
	// SVDRef(M,U,D,V,args);
	SvdSolver::solve(M,U,D,V,args);
}

std::unique_ptr<RsvdSolver> RsvdSolver::create() {
    return std::unique_ptr<RsvdSolver>(new RsvdSolver());
}

} // namespace itensor
#endif
