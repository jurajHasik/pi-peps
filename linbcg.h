#ifndef __NR_LINBCG_H_
#define __NR_LINBCG_H_

struct Linbcg {
	virtual void asolve(VecDoub_I &b, VecDoub_O &x, const Int itrnsp) = 0;
	virtual void atimes(VecDoub_I &x, VecDoub_O &r, const Int itrnsp) = 0;
	void solve(VecDoub_I &b, VecDoub_IO &x, const Int itol, const Doub tol,
		const Int itmax, Int &iter, Doub &err);
	Doub snrm(VecDoub_I &sx, const Int itol);
};


//-----------------------------------------------------------------------------
struct FULSCG : Linbcg {
	itensor::ITensor & M;
	itensor::ITensor & B;
	itensor::ITensor & A;
	itensor::ITensor cmbA;
	itensor::ITensor cmbKet;
	double svd_cutoff = 1.0e-15;

	FULSCG(itensor::ITensor & MM, itensor::ITensor & BB, 
		itensor::ITensor & AA, itensor::ITensor ccmbA, itensor::ITensor ccmbKet,
		double ssvd_cutoff);

	void asolve(VecDoub_I &b, VecDoub_O &x, const Int itrnsp);
    
    void atimes(VecDoub_I &x, VecDoub_O &r, const Int itrnsp);
};


//-----------------------------------------------------------------------------
struct FULSCG_IT {
	itensor::ITensor & M;
	itensor::ITensor & B;
	itensor::ITensor & A;
	itensor::ITensor cmbA;
	itensor::ITensor cmbKet;
	double svd_cutoff = 1.0e-15;

	FULSCG_IT(itensor::ITensor & MM, itensor::ITensor & BB, 
		itensor::ITensor & AA, itensor::ITensor ccmbA, itensor::ITensor ccmbKet,
		double ssvd_cutoff);

	void asolve(itensor::ITensor const& b, itensor::ITensor & x, const Int itrnsp);
    
    void atimes(itensor::ITensor const& x, itensor::ITensor & r, const Int itrnsp);

    void solveIT(itensor::ITensor const& b, itensor::ITensor & x, const Int itol, const Doub tol,
		const Int itmax, Int &iter, Doub &err);
	Doub snrmIT(itensor::ITensor const& sx, const Int itol);
};

#endif