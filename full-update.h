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
#include "itensor/all.h"

const int IOFFSET = 100;

// defines the order in which on-site tensors are contracted for
// various choice of reduction tensors to optimize
const std::array< std::array<int, 4>, 4> ORD = 
	{{{1,2,3,0},{0,3,2,1},{2,3,0,1},{1,0,3,2}}};

const std::array< std::array<int, 3>, 4> ORD_R = 
	{{{1,2,0},{0,2,1},{2,0,1},{1,0,2}}};

const std::array< std::array<int, 2>, 2> PL_R =
	{{ 
		{4,3},
		{1,2},
	}};

const std::array< std::array<int, 2>, 2> A_R =
	{{ 
		{2,1},
		{0,1},
	}};

const std::array<int, 4> ORD_DIR = {{1,-1,1,-1}};

// defines assignment of reduction tensors given an order from
// ORD for Matrix M <phi~|phi~>
// {ket,bra,ket,bra}
const std::array< std::array< std::array<int, 4>, 4>, 4> RTPM =
	{{ 
		{{{1,1,2,2},{3,3,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}}},
		{{{0,0,-1,-1},{-1,-1,-1,-1},{3,3,-1,-1},{2,2,-1,-1}}},
		{{{3,3,-1,-1},{-1,-1,-1,-1},{0,0,-1,-1},{1,1,-1,-1}}},
		{{{1,1,2,2},{0,0,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}}}
	}};

const std::array< std::array< std::array<int, 2>, 3>, 4> RTPM_R =
	{{ 
		{{{1,2},{3,-1},{-1,-1}}},
		{{{0,-1},{3,-1},{2,-1}}},
		{{{3,-1},{0,-1},{1,-1}}},
		{{{1,2},{0,-1},{-1,-1}}}
	}};

// defines assignment of reduction tensors given an order from
// ORD for Matrix K <phi|U|phi~>= <phi'|phi~>
const std::array< std::array< std::array<int, 4>, 4>, 4> RTPK =
	{{ 
		{{{-1,1,-1,2},{-1,3,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}}},
		{{{-1,0,-1,-1},{-1,-1,-1,-1},{-1,3,-1,-1},{-1,2,-1,-1}}},
		{{{-1,3,-1,-1},{-1,-1,-1,-1},{-1,0,-1,-1},{-1,1,-1,-1}}},
		{{{-1,1,-1,2},{-1,0,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}}}
	}};

/*
 * Initialize isometry (reduction tensors - RT) which define a 
 * PEPS ansatz of auxDim d for a PEPS state with auxDim d and 
 * 2-site/3-site/... gate applied 
 *
 */
void initRT_basic(itensor::ITensor& rt, std::string INIT_METHOD, 
	itensor::Args const& args = itensor::Args::global());

/*
 * contract on-site bra-ket tensor s with given operator op, 
 * possibly apply reduction tensors rt and merge aux-indices with prime level
 * plToEnv for contraction with the environment
 *
 * reduction tensors are given as array of pointers, NULL entry defines
 * no reduction tensor applied. Reduction tensor at rt[0] and rt[2] are applied
 * to |ket> part of resulting tensor, while rt[1] and rt[3] are applied on <bra| 
 *
 * merging of on-site aux-indices is as defined follows
 * plToEnv[0]=I_XH1, plToEnv[1]=Index(), ...
 * where default index Index() is evaluated to false <=> unmerged
 * position within plToEnv gives aux-index (primeLevel of aux-index) 
 *
 */
itensor::ITensor getT(itensor::ITensor const& s, 
	std::array<itensor::Index, 4> const& plToEnv, bool dbg = false);

itensor::ITensor getT(itensor::ITensor const& s, 
	std::array<itensor::Index, 4> const& plToEnv, itensor::ITensor const& op,
	std::array<const itensor::ITensor *, 4> rt, bool dbg = false);

itensor::ITensor getT(itensor::ITensor const& s, 
	std::array<itensor::Index, 4> const& plToEnv, itensor::ITensor const& op,
	std::array<const itensor::ITensor *, 4> rt, bool pIcont, bool dbg);

itensor::ITensor getketT(itensor::ITensor const& s, itensor::ITensor const& op,
	std::array<const itensor::ITensor *, 2> rt, bool dbg = false);

/*
 * tn - defines sequence of on site tensors over which uJ1J2 is applied
 * pl - defines primeLevel on the indices of tensor given in tn
 *
 * Hence, full string of tensors + indices by which they should be contracted
 * is given as
 *
 * pl[0]--tn[0]--pl[1]--pl[2]--tn[1]--pl[3]--pl[4]--tn[2]--pl[5]--...
 *   ...--pl[6]--tn[3]--pl[7]--(pl[0]--tn[0])
 *
 */
itensor::Args fullUpdate(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	std::vector< itensor::ITensor > & iso_store, 
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_2site(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	std::vector< itensor::ITensor > & iso_store, 
	itensor::Args const& args = itensor::Args::global());

itensor::Args fullUpdate_2site_v2(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	std::vector< itensor::ITensor > & iso_store, 
	itensor::Args const& args = itensor::Args::global());

#endif