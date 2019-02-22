//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __FULL_UPDT_H_
#define __FULL_UPDT_H_

#include <cmath>
#include <limits>
#include <chrono>
#include "p-ipeps/models.h"
#include "p-ipeps/ctm-cluster-env_v2.h"
#include "p-ipeps/ctm-cluster-global.h"
#include "p-ipeps/ctm-cluster.h"
#include "p-ipeps/su2.h"
#include "p-ipeps/linalg/itensor-linsys-solvers.h"
#include "itensor/all.h"

itensor::ITensor pseudoInverse(itensor::ITensor const& M,
	itensor::Args const& args = itensor::Args::global());


itensor::Args fullUpdate_ALS2S_IT(MPO_2site const& mpo, Cluster & cls,
	CtmEnv const& ctmEnv,
	std::vector<std::string> const& tn, std::vector<int> pl,
	itensor::LinSysSolver const& ls,
	itensor::Args const& args = itensor::Args::global());


itensor::Args fullUpdate_2S(MPO_2site const& mpo, Cluster & cls,
	CtmEnv const& ctmEnv,
	std::vector<std::string> const& tn, std::vector<int> pl,
	itensor::LinSysSolver const& ls,
	itensor::Args const& args = itensor::Args::global());


itensor::Args fullUpdate_ALS3S_IT(MPO_3site const& uJ1J2, Cluster & cls, 
	CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::LinSysSolver const& ls,
	itensor::Args const& args = itensor::Args::global());

struct FUlinSys {
	itensor::ITensor & M;
	itensor::ITensor & B;
	itensor::ITensor & A;
	itensor::ITensor cmbA;
	itensor::ITensor cmbKet;
	itensor::Args const& args;

	double res  = 0.0;
	double nres = 0.0; // residue normalized by right hand side
	bool dbg    = false;

	FUlinSys(itensor::ITensor & MM, itensor::ITensor & BB, 
		itensor::ITensor & AA, itensor::ITensor ccmbA, itensor::ITensor ccmbKet,
		itensor::Args const& aargs = itensor::Args::global());

    void solve(itensor::ITensor const& b, itensor::ITensor & x, 
    	int &iter, double &err,
    	itensor::LinSysSolver const& ls,
    	itensor::Args const& args = itensor::Args::global());
};

itensor::Args fullUpdate_CG_full4S(OpNS const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

struct FU4SiteGradMin {
	CtmEnv const& ctmEnv;
	std::vector< itensor::ITensor > const& pc; // protocorners 
	std::vector< std::string > const& tn;      // site IDs
	std::vector< int > const& pl;              // primelevels of aux indices          
	itensor::ITensor const& op4s;              // four-site operator
	std::vector< itensor::ITensor > & rX;
	std::vector< itensor::Index > const& iQX;
	itensor::Args const& args;

	itensor::ITensor protoK;
	std::vector< itensor::ITensor > g, xi, h;

	double epsdistf;
	double normUPsi; // <psi|U^dag U|psi>
	double inst_normPsi;
	double inst_overlap;

	FU4SiteGradMin(
		CtmEnv const& cctmEnv,
		std::vector< itensor::ITensor > const& ppc, // protocorners 
		std::vector< std::string > const& ttn,      // site IDs
		std::vector< int > const& ppl,              // primelevels of aux indices          
		itensor::ITensor const& oop4s,              // four-site operator
		std::vector< itensor::ITensor > & rrX,
		std::vector< itensor::Index > const& iiQX,
		itensor::Args const& aargs);

	// Linear Model
	itensor::Real operator()(itensor::Vec<itensor::Real> const& x, 
			itensor::Vec<itensor::Real> & grad);

	// void minimize();
	
	std::vector<double> func() const;

	// double linmin(double fxi, std::vector< itensor::ITensor > const& g);

	void gradient(std::vector< itensor::ITensor > & grad);
};

#endif