#ifndef _ITENSOR_SVD_SOLVERS_H
#define _ITENSOR_SVD_SOLVERS_H

#include "pi-peps/config.h"
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS

namespace itensor {

  struct SvdSolver {
    virtual void solve(MatRefc<Real> const& M,
                       MatRef<Real> const& U,
                       VectorRef const& D,
                       MatRef<Real> const& V,
                       Args const& args) {
      // To be overloaded by derived classes
      // std::cout<<"[SvdSolver::solve<Real>] called"<<std::endl;
      auto thresh = args.getReal("SVDThreshold", 1E-3);
      SVDRef(M, U, D, V, thresh);
    }

    virtual void solve(MatRefc<Cplx> const& M,
                       MatRef<Cplx> const& U,
                       VectorRef const& D,
                       MatRef<Cplx> const& V,
                       Args const& args) {
      // To be overloaded by derived classes
      // std::cout<<"[SvdSolver::solve<Cplx>] called"<<std::endl;
      auto thresh = args.getReal("SVDThreshold", 1E-3);
      SVDRef(M, U, D, V, thresh);
    }

    virtual ~SvdSolver() = default;

    static std::unique_ptr<SvdSolver> create() {
      return std::unique_ptr<SvdSolver>(new SvdSolver());
    }
  };

  template <class Tensor>
  Spectrum svd(Tensor AA,
               Tensor& U,
               Tensor& D,
               Tensor& V,
               SvdSolver& solver,
               Args args = Global::args());

  template <class Tensor>
  Spectrum svd(Tensor AA,
               Tensor& U,
               Tensor& D,
               Tensor& V,
               SvdSolver& solver,
               Args args) {
    using IndexT = typename Tensor::index_type;

#ifdef DEBUG
    if (!U && !V)
      Error(
        "U and V default-initialized in svd, must indicate at least one index "
        "on U or V");
#endif

    auto noise = args.getReal("Noise", 0);
    auto useOrigM = args.getBool("UseOrigM", false);

    if (noise > 0)
      Error("Noise term not implemented for svd");

    // if(isZero(AA,Args("Fast")))
    //    throw ResultIsZero("svd: AA is zero");

    // Combiners which transform AA
    // into a rank 2 tensor
    std::vector<IndexT> Uinds, Vinds;
    Uinds.reserve(AA.r());
    Vinds.reserve(AA.r());
    // Divide up indices based on U
    // If U is null, use V instead
    auto& L = (U ? U : V);
    auto &Linds = (U ? Uinds : Vinds), &Rinds = (U ? Vinds : Uinds);
    for (const auto& I : AA.inds()) {
      if (hasindex(L, I))
        Linds.push_back(I);
      else
        Rinds.push_back(I);
    }
    Tensor Ucomb, Vcomb;
    if (!Uinds.empty()) {
      Ucomb = combiner(std::move(Uinds), {"IndexName", "uc"});
      AA *= Ucomb;
    }
    if (!Vinds.empty()) {
      Vcomb = combiner(std::move(Vinds), {"IndexName", "vc"});
      AA *= Vcomb;
    }

    if (useOrigM) {
      // Try to determine current m,
      // then set minm_ and maxm_ to this.
      args.add("Cutoff", -1);
      long minm = 1, maxm = MAX_M;
      if (D.r() == 0) {
        auto mid = commonIndex(U, V, Link);
        if (mid)
          minm = maxm = mid.m();
        else
          minm = maxm = 1;
      } else {
        minm = maxm = D.inds().front().m();
      }
      args.add("Minm", minm);
      args.add("Maxm", maxm);
    }

    auto ui = commonIndex(AA, Ucomb);
    auto vi = commonIndex(AA, Vcomb);

    auto spec = svdRank2(AA, ui, vi, U, D, V, solver, args);

    U = dag(Ucomb) * U;
    V = V * dag(Vcomb);

    return spec;
  }  // svd

  template <typename IndexT>
  Spectrum svdRank2(ITensorT<IndexT> const& A,
                    IndexT const& ui,
                    IndexT const& vi,
                    ITensorT<IndexT>& U,
                    ITensorT<IndexT>& D,
                    ITensorT<IndexT>& V,
                    SvdSolver& solver,
                    Args args = Args::global());

  template <class MatM,
            class MatU,
            class VecD,
            class MatV,
            class = stdx::require<hasMatRange<MatM>,
                                  hasMatRange<MatU>,
                                  hasVecRange<VecD>,
                                  hasMatRange<MatV>>>
  void SVD(MatM&& M,
           MatU&& U,
           VecD&& D,
           MatV&& V,
           SvdSolver& solver,
           Args const& args);

  template <class MatM, class MatU, class VecD, class MatV, class>
  void SVD(MatM&& M,
           MatU&& U,
           VecD&& D,
           MatV&& V,
           SvdSolver& solver,
           Args const& args) {
    auto Mr = nrows(M), Mc = ncols(M);
    auto nsv = std::min(Mr, Mc);
    resize(U, Mr, nsv);
    resize(V, Mc, nsv);
    resize(D, nsv);
    SVDRef(makeRef(M), makeRef(U), makeRef(D), makeRef(V), solver, args);
  }

  template <typename T>
  void SVDRef(MatRefc<T> const& M,
              MatRef<T> const& U,
              VectorRef const& D,
              MatRef<T> const& V,
              SvdSolver& solver,
              Args const& args);

}  // namespace itensor

#endif
