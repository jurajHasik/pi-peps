//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __FULL_UPDT_H_
#define __FULL_UPDT_H_

#include <cmath>
#include <limits>
#include <chrono>
#include "models.h"
#include "ctm-cluster-env_v2.h"
#include "ctm-cluster-global.h"
#include "ctm-cluster.h"
#include "su2.h"
#include "itensor-linsys-solvers.h"
#include "itensor/all.h"

itensor::ITensor pseudoInverse(itensor::ITensor const& M,
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_ALS2S_IT(MPO_2site const& mpo, Cluster & cls,
	CtmEnv const& ctmEnv,
	std::vector<std::string> const& tn, std::vector<int> pl,
	itensor::LinSysSolver const& ls,
	itensor::Args const& args = itensor::Args::global());

#endif