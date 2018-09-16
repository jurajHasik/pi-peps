//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __FULL_UPDT_TEST_H_
#define __FULL_UPDT_TEST_H_

#include "full-update.h"
#include "nr3.h"
#include "linbcg.h"
#include "mins.h"
#include "mins_ndim.h"

itensor::Args fullUpdate_COMB_INV(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::ITensor psInv(itensor::ITensor const& M,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_COMB_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

struct Funcd {
	itensor::ITensor const& N;
	itensor::ITensor const& cmbKet;
	itensor::ITensor const& cmbBra;
	std::vector<double> const& vecB;
	double fconst;

	Funcd(itensor::ITensor const& NN, itensor::ITensor const& ccmbKet,
		itensor::ITensor const& ccmbBra, std::vector<double> const& vvecB, 
		double ffconst);

	Doub operator() (VecDoub_I &x);

	void df(VecDoub_I &x, VecDoub_O &deriv);
};

itensor::Args fullUpdate_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

struct FuncCG {
	itensor::ITensor const& N, protoK;
	itensor::ITensor const& cmbX1, cmbX2, cmbX3;
	std::array<itensor::Index, 4> const& aux;
	std::vector<int> const& pl;
	double psiUNorm;
	double finit;
	double psiNorm;
	int evalCount;
	int dfCount;

	FuncCG(itensor::ITensor const& NN, itensor::ITensor const& pprotoK, 
		itensor::ITensor const& ccmbX1, itensor::ITensor const& ccmbX2,
		itensor::ITensor const& ccmbX3, 
		std::array<itensor::Index, 4> const& aaux,
		std::vector<int> const& ppl,
		double ppsiUNorm,
		double ffinit);

	Doub operator() (VecDoub_I &x);

	void df(VecDoub_I &x, VecDoub_O &deriv);
};

struct FuncCGV2 {
	itensor::ITensor const& N, protoK;
	itensor::ITensor const& cmbX1, cmbX2, cmbX3;
	std::array<itensor::Index, 4> const& aux;
	std::vector<int> const& pl;
	double psiUNorm;
	double finit;
	double psiNorm;
	int evalCount;
	int dfCount;

	FuncCGV2(itensor::ITensor const& NN, itensor::ITensor const& pprotoK, 
		itensor::ITensor const& ccmbX1, itensor::ITensor const& ccmbX2,
		itensor::ITensor const& ccmbX3, 
		std::array<itensor::Index, 4> const& aaux,
		std::vector<int> const& ppl,
		double ppsiUNorm,
		double ffinit);

	Doub operator() (VecDoub_I &x);

	void df(VecDoub_I &x, VecDoub_O &deriv);
};

itensor::Args fullUpdate_ALS_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

struct FuncALS_CG {
	itensor::ITensor const& M;
	itensor::ITensor & K;
	itensor::ITensor cmbKet;
	Doub psiUNorm;
	Doub finit;
	Doub psiNorm;

	FuncALS_CG(itensor::ITensor const& MM, itensor::ITensor & KK,
		itensor::ITensor ccmbKet, double ppsiUNorm, double ffinit, double ppsiNorm);

	void setup(itensor::ITensor ccmbKet, double ppsiUNorm, double ffinit, double ppsiNorm);

	Doub operator() (VecDoub_I &x);

	void df(VecDoub_I &x, VecDoub_O &deriv);
};

itensor::Args fullUpdate_ALS_LSCG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_LSCG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

// using CG implemented by ITensor
itensor::Args fullUpdate_ALS3S_LSCG_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_ALS_PINV_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_LSCG_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());


itensor::Args fullUpdate_CG_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_ALS4S_LSCG_IT(OpNS const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_CG_full4S(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

#endif