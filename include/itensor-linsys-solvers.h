#ifndef _ITENSOR_LINSYS_SOLVERS_H
#define _ITENSOR_LINSYS_SOLVERS_H

#include "itensor/all.h"

namespace itensor {

// Direct Solver
// 
// Linear system of equations
//
// The routine solves for X the system of linear equations A*X = B,
// where A is an n-by-n matrix, the columns of matrix B are individual right-hand sides,
// and the columns of X are the corresponding solutions.
//
// Resulting X is returned in B
//
struct LinSysSolver {
  	
    virtual void 
    solve(
      MatRefc<Real>	const& A,
      VecRef<Real>  const& B, 
      VecRef<Real>  const& X,
      Args const& args) 
    {  
        // To be overloaded by derived classes
        std::cout<<"[LinSysSolver::solve<Real>] called"<<std::endl;
    }

    virtual void 
    solve(
      MatRefc<Cplx>  const& A,
      VecRef<Cplx>	const& B, 
      VecRef<Cplx>	const& X, 
      Args const& args) 
    {  
        // To be overloaded by derived classes
        std::cout<<"[LinSysSolver::solve<Cplx>] called"<<std::endl;
    }
};


template<class I>
void 
linsystem(ITensorT<I> A,
          ITensorT<I> B,
          ITensorT<I> & X,
          LinSysSolver & solver,
          Args args = Args::global());

template<class I>
void 
linsystem(ITensorT<I> A,
          ITensorT<I> B,
          ITensorT<I> & X,
          LinSysSolver & solver,
          Args args)
{
    if(!args.defined("IndexName")) args.add("IndexName","ind_x");
    auto dbg = args.getBool("dbg",false);

    // TODO MORE SOPHISTICATED INDEX ANALYSIS
    //
    // Pick an arbitrary index and do some analysis
    // on its prime level spacing
    //
    auto k = A.inds().front();
    auto kps = stdx::reserve_vector<int>(rank(A));
    for(auto& i : A.inds()) if(i.noprimeEquals(k)) kps.push_back(i.primeLevel());
    if (dbg) {
      std::cout<<"[linsystem] primeLevels: ";
      for(unsigned int i=0; i<kps.size(); i++) std::cout<< kps[i] <<" ";
      std::cout<< std::endl;
    }
    if (kps.size() <= 1ul || kps.size()%2 != 0ul) {
      Error("Input tensor A (n x n Matrix of coeffs) to linsystem should \
        have pairs of indices with equally spaced prime levels. Odd number of \
        same indices (modulo primeLevel)");
    }
    
    int pdiff = -1;
    if(not args.defined("plDiff")) {
      auto nk = kps.size();
      std::sort(kps.begin(),kps.end());
      //idiff == "inner" difference between cluster of low-prime-level copies
      //         of k, if more than one
      auto idiff = kps.at(nk/2-1)-kps.front();
      //mdiff == max prime-level difference of copies of k
      auto mdiff = kps.back()-kps.front();
      //pdiff == spacing between lower and higher prime level index pairs
      pdiff = mdiff-idiff;
      if (dbg) {
        std::cout<<"[linsystem] plDiff NOT provided"<< std::endl;
        std::cout<<"[linsystem] idiff: "<< idiff <<" mdiff: "<< mdiff <<" pdiff: "
          << pdiff << std::endl;
      }
    } else { 
      pdiff = args.getInt("plDiff",-1);
    }
    if (pdiff == -1) { Error("[linsystem] Invalid plDiff value"); }

    auto inds = stdx::reserve_vector<I>(rank(A)/2);
    for(auto& i : A.inds())
    for(auto& j : A.inds())
        {
        if(i.noprimeEquals(j) && i.primeLevel()+pdiff == j.primeLevel())
            {
            inds.push_back(i);
            }
        }
    if(inds.empty() || rank(A)/2 != (long)inds.size()) 
        {
        Error("Input tensor to linsystem should have pairs of indices with equally spaced prime levels");
        }
    if (dbg) {
      std::cout<<"[linsystem] inds: ";
      for (unsigned int i=0; i<inds.size(); i++) std::cout << inds[i] <<" ";
      std::cout<< std::endl;
    }

    // compare against indices of B
    // B.inds() must match either inds or prime(inds,pdiff) exactly
    if (rank(B) != rank(A)/2) { Error("[linsystem] rank(B) != rank(A)/2"); }
    int plB_check = 0;
    auto bInds = stdx::reserve_vector<I>(rank(A)/2);
    for (auto& i : inds) {
      for (auto& j : B.inds()) {
        if (i==j)
          bInds.push_back(j);
        else if (prime(i,pdiff)==j) {
          bInds.push_back(j);
          plB_check += pdiff;
        }
      }
    }
    if (dbg) {
      std::cout<<"[linsystem] bInds: ";
      for (unsigned int i=0; i<bInds.size(); i++) std::cout << bInds[i] <<" ";
      std::cout<< std::endl;
    }
    if ((bInds.size() != rank(A)/2) || !(plB_check==0 || plB_check==rank(B)*pdiff)) { 
      std::cout << "bInds.size() != rank(A)/2 : "<< (bInds.size() != rank(A)/2) << std::endl;
      std::cout << "plB_check==0 : "<< (plB_check==0) << std::endl;
      std::cout << "plB_check==rank(B)*pdiff : "<< (plB_check==rank(B)*pdiff) << std::endl;
      Error("[linsystem] B.inds() are incompatible with indices of matrix A"); 
    }

    auto combA = combiner(std::move(inds), args);
    auto combB = combiner(std::move(bInds),args); 
    A *= combA;
    B *= combB;

    auto combAP = dag(prime(combA,pdiff));
    try {
        A *= combAP;
        }
    catch(ITError const& e)
        {
        println("Diagonalize expects opposite arrow directions for primed and unprimed indices.");
        throw e;
        }

    if (dbg) {
      std::cout<<"[linsystem] ";
      PrintData(A);
      std::cout<<"[linsystem] ";
      PrintData(B);
    }

  //   auto N = ncols(A);
  // if(N < 1) throw std::runtime_error("linSystem: 0 dimensional matrix");
  // if(N != nrows(A))
  //   {
  //   printfln("A is %dx%d",nrows(A),ncols(A));
  //   throw std::runtime_error("linSystem: Input Matrix must be square");
  //   }

    // rank(A) = 2, rank(B) = 1
    linsystemRank2(A,B,X,solver,args);

    // restore original indices
    // A *= combAP;
    // A *= combA;
    // B *= combB;
    X = combB * X;
} //linsystem


template<typename I>
void
linsystemRank2(
		   ITensorT<I>  & A, 
           ITensorT<I>  & B,
           ITensorT<I>  & X,
           LinSysSolver & solver,
           Args const& args);


template<class MatA, class VecB, class VecX,
         class = stdx::require<
         hasMatRange<MatA>,
         hasVecRange<VecB>,
         hasVecRange<VecX>
         >>
void
linsystemRef(
	    MatA && A,
        VecB && B,
        VecX && X,
        LinSysSolver & solver,
        Args const& args);


template<class MatA, 
         class VecB,
         class VecX,
         class>
void
linsystemRef(
		MatA && A,
        VecB && B,
        VecX && X,
        LinSysSolver & solver,
        Args const& args)
{
	resize(X,nrows(A));
	linsystemMatVec(makeRef(A),makeRef(B),makeRef(X),solver,args);
}

template<typename T>
void
linsystemMatVec(
	   MatRefc<T> const& A, 
       VecRef<T> const& D, 
       VecRef<T> const& V,
       LinSysSolver & solver,
       Args const& args);

} // namespace itensor

#endif