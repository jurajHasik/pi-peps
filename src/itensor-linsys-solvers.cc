#include "itensor-linsys-solvers.h" 

namespace itensor {

template<typename T>
void
linsystemImpl(ITensor & A, 
          	  ITensor & B,
          	  ITensor & X,
          	  LinSysSolver & solver,
          	  Args const& args)
{
    auto dbg = args.getBool("dbg",false);

    auto i1 = A.inds().front();
    auto i2 = A.inds().back();
    auto active = (i1.primeLevel() < i2.primeLevel()) ? i1 : i2;
    auto pdiff = std::abs(i1.primeLevel()-i2.primeLevel());

    auto ib    = B.inds().front();
    auto dummy = Index("dummy",1);

    if (dbg) {
    	std::cout<<"[linsystemImpl] primeLevel difference on A: "<< pdiff << std::endl;
    	std::cout<<"[linsystemImpl] indices of A: "<< std::endl;
    	std::cout<<"[linsystemImpl] active: "<< active <<" other: "
    		<< prime(active) << std::endl;
    	std::cout<<"[linsystemImpl] indices of B: "<< ib << std::endl;
    }

	Vec<T> XX;
    auto RA = toMatRefc<T>(A,active,prime(active));

    auto extractT = [](Dense<T> const& d) { return d.store; };
	auto storageB = applyFunc(extractT,B.store());
    auto RB = makeVecRef<T>(storageB.data(), storageB.size());
    
    linsystemRef(RA,RB,XX,solver,args);


    X = ITensor({ib,dummy},Dense<T>{move(XX.storage())});
    // X = ITensor({ib,dummy},Dense<T>{move(XX.storage())},A.scale());
    // if(not A.scale().isTooBigForReal()) {
         X *= (B.scale().real0()/A.scale().real0());
    // } else {
    //     println("lin_systemImpl: scale too big for Real");
    // }

    // absorb dummy index (TODO if necessary)
    auto combX = combiner(ib,dummy);
    X = X*combX;
    X = X*delta(commonIndex(combX,X),ib);
    if (dbg) {
    	std::cout<<"[linsystemImpl] combX: "<< combX;
    	PrintData(X);
    }
}

template<typename I>
void
linsystemRank2(
           ITensorT<I>  & A, 
           ITensorT<I>  & B,
           ITensorT<I>  & X,
           LinSysSolver & solver,
           Args const& args)
{
    if(isComplex(A))
        {
        return linsystemImpl<Cplx>(A,B,X,solver,args);
        }
    return linsystemImpl<Real>(A,B,X,solver,args);
}
template
void
linsystemRank2(
            ITensor &,
            ITensor &,
            ITensor &,
            LinSysSolver &,
            Args const&);


template<typename T>
void
linsystemMatVec(
	   MatRefc<T>  const& A,
       VecRef<T>  const& B,
       VecRef<T>  const& X,
       LinSysSolver & solver,
       Args const& args)
{
        solver.solve(A,B,X,args);
}
template void linsystemMatVec(MatRefc<Real> const&,VecRef<Real> const&, VecRef<Real> const&, 
    LinSysSolver & solver, Args const& args);
template void linsystemMatVec(MatRefc<Cplx> const&,VecRef<Cplx> const&, VecRef<Cplx> const&, 
	LinSysSolver & solver, Args const& args);

} //namespace itensor