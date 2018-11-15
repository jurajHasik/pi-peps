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
	itensor::Args const& args;

	double res  = 0.0;
	double nres = 0.0; // residue normalized by right hand side
	bool dbg           = false;
	std::string solver = "pseudoinverse";

	FULSCG_IT(itensor::ITensor & MM, itensor::ITensor & BB, 
		itensor::ITensor & AA, itensor::ITensor ccmbA, itensor::ITensor ccmbKet,
		itensor::Args const& aargs = itensor::Args::global());

	void asolve(itensor::ITensor const& b, itensor::ITensor & x, const Int itrnsp);
    
	void asolve_pinv(itensor::ITensor const& b, itensor::ITensor & x);

	void asolve_linsystem(itensor::ITensor & x);

    void atimes(itensor::ITensor const& x, itensor::ITensor & r, const Int itrnsp);

    void solve(itensor::ITensor const& b, itensor::ITensor & x, Int &iter, Doub &err, 
    	itensor::Args const& args = itensor::Args::global());

    void solve(itensor::ITensor const& b, itensor::ITensor & x, Int &iter, Doub &err,
    	itensor::LinSysSolver const& ls,
    	itensor::Args const& args = itensor::Args::global());

    void solveBiCG(itensor::ITensor const& b, itensor::ITensor & x, const Int itol, const Doub tol,
		const Int itmax, Int &iter, Doub &err);

	Doub snrm(itensor::ITensor const& sx, const Int itol);
};

struct CG4S_IT {
	std::vector< itensor::ITensor > const& pc; // protocorners 
	std::array< itensor::Index, 4 > const& aux;  // aux indices
	std::vector< std::string > const& tn;      // site IDs
	std::vector< int > const& pl;              // primelevels of aux indices          
	itensor::ITensor const& op4s;              // four-site operator
	Cluster & cls;
	itensor::Args const& args;

	itensor::ITensor protoK;
	std::vector< itensor::ITensor > g, xi, h;

	double epsdistf;
	double normUPsi; // <psi|U^dag U|psi>
	double inst_normPsi;
	double inst_overlap;

	CG4S_IT(
		std::vector< itensor::ITensor > const& ppc, // protocorners 
		std::array< itensor::Index, 4 > const& aaux,  // aux indices
		std::vector< std::string > const& ttn,      // site IDs
		std::vector< int > const& ppl,              // primelevels of aux indices          
		itensor::ITensor const& oop4s,              // four-site operator
		Cluster & ccls,
		itensor::Args const& aargs);

	void minimize();
	
	double func();

	double linmin(double fxi, std::vector< itensor::ITensor > const& g);

	void grad(std::vector< itensor::ITensor > &grad);
    
 //    void atimes(itensor::ITensor const& x, itensor::ITensor & r, const Int itrnsp);

 //    void solveIT(itensor::ITensor const& b, itensor::ITensor & x, const Int itol, const Doub tol,
	// 	const Int itmax, Int &iter, Doub &err);

	// Doub snrmIT(itensor::ITensor const& sx, const Int itol);
};

#endif