#include <algorithm>
#include <tuple>
#include "p-ipeps/itensor-svd-solvers.h" 

namespace itensor {

const auto MAX_INT = std::numeric_limits<int>::max();

using std::swap;
using std::istream;
using std::ostream;
using std::vector;
using std::find;
using std::pair;
using std::make_pair;
using std::string;
using std::sqrt;
using std::move;
using std::tie;

template <typename T>
Spectrum
svdImpl(ITensor const& A,
        Index const& ui, 
        Index const& vi,
        ITensor & U, 
        ITensor & D, 
        ITensor & V,
        SvdSolver & solver,
        Args const& args
        )
    {
    SCOPED_TIMER(7);
    auto do_truncate = args.getBool("Truncate");
    auto thresh = args.getReal("SVDThreshold",1E-3);
    auto cutoff = args.getReal("Cutoff",MIN_CUT);
    auto maxm = args.getInt("Maxm",MAX_M);
    auto minm = args.getInt("Minm",1);
    auto doRelCutoff = args.getBool("DoRelCutoff",true);
    auto absoluteCutoff = args.getBool("AbsoluteCutoff",false);
    auto lname = args.getString("LeftIndexName","ul");
    auto rname = args.getString("RightIndexName","vl");
    auto itype = getIndexType(args,"IndexType",Link);
    auto litype = getIndexType(args,"LeftIndexType",itype);
    auto ritype = getIndexType(args,"RightIndexType",itype);
    auto show_eigs = args.getBool("ShowEigs",false);

    auto M = toMatRefc<T>(A,ui,vi);

    Mat<T> UU,VV;
    Vector DD;

    TIMER_START(6)
    SVD(M,UU,DD,VV,solver,args);
    TIMER_STOP(6)

    //conjugate VV so later we can just do
    //U*D*V to reconstruct ITensor A:
    conjugate(VV);

    //
    // Truncate
    //
    Vector probs;
    if(do_truncate || show_eigs)
        {
        probs = DD;
        for(auto j : range(probs)) probs(j) = sqr(probs(j));
        }

    Real truncerr = 0;
    Real docut = -1;
    long m = DD.size();
    if(do_truncate)
        {
        tie(truncerr,docut) = truncate(probs,maxm,minm,cutoff,
                                       absoluteCutoff,doRelCutoff,args);
        m = probs.size();
        resize(DD,m);
        reduceCols(UU,m);
        reduceCols(VV,m);
        }


    if(show_eigs) 
        {
        auto showargs = args;
        showargs.add("Cutoff",cutoff);
        showargs.add("Maxm",maxm);
        showargs.add("Minm",minm);
        showargs.add("Truncate",do_truncate);
        showargs.add("DoRelCutoff",doRelCutoff);
        showargs.add("AbsoluteCutoff",absoluteCutoff);
        showEigs(probs,truncerr,A.scale(),showargs);
        }
    
    Index uL(lname,m,litype),
          vL(rname,m,ritype);

    //Fix sign to make sure D has positive elements
    Real signfix = (A.scale().sign() == -1) ? -1 : +1;
    D = ITensor({uL,vL},
                Diag<Real>{DD.begin(),DD.end()},
                A.scale()*signfix);
    U = ITensor({ui,uL},Dense<T>(move(UU.storage())),LogNum(signfix));
    V = ITensor({vi,vL},Dense<T>(move(VV.storage())));

    //Square all singular values
    //since convention is to report
    //density matrix eigs
    for(auto& el : DD) el = sqr(el);

#ifdef USESCALE
    if(A.scale().isFiniteReal()) 
        {
        DD *= sqr(A.scale().real0());
        }
    else                         
        {
        println("Warning: scale not finite real after svd");
        }
#endif

    return Spectrum(move(DD),{"Truncerr",truncerr});
    }


template<typename IndexT>
Spectrum 
svdRank2(ITensorT<IndexT> const& A, 
         IndexT const& ui, 
         IndexT const& vi,
         ITensorT<IndexT> & U, 
         ITensorT<IndexT> & D, 
         ITensorT<IndexT> & V,
         SvdSolver & solver,
         Args args)
    {
    auto do_truncate = args.defined("Cutoff") 
                    || args.defined("Maxm");
    if(not args.defined("Truncate")) 
        {
        args.add("Truncate",do_truncate);
        }

    if(A.r() != 2) 
        {
        Print(A);
        Error("A must be matrix-like (rank 2)");
        }
    if(isComplex(A))
        {
        return svdImpl<Cplx>(A,ui,vi,U,D,V,solver,args);
        }
    return svdImpl<Real>(A,ui,vi,U,D,V,solver,args);
    }
template
Spectrum 
svdRank2(ITensor const&,Index const&,Index const&,
         ITensor &,ITensor &,ITensor &,SvdSolver &,Args );
// template Spectrum 
// svdRank2(IQTensor const&,IQIndex const&,IQIndex const&,
//          IQTensor &,IQTensor &,IQTensor &,Args , SvdSolver);
	

template<typename T>
void
SVDRef(MatRefc<T> const& M,
       MatRef<T>  const& U,
       VectorRef  const& D,
       MatRef<T>  const& V,
       SvdSolver & solver,
       Args const& args)
    {
        solver.solve(M,U,D,V,args);
    }
template void SVDRef(MatRefc<Real> const&,MatRef<Real> const&, VectorRef const&, 
    MatRef<Real> const&, SvdSolver & solver, Args const& args);
template void SVDRef(MatRefc<Cplx> const&,MatRef<Cplx> const&, VectorRef const&, 
	MatRef<Cplx> const&, SvdSolver & solver, Args const& args);

} // namespace itensor
