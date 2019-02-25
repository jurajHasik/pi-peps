//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __SMPL_UPDT_H_
#define __SMPL_UPDT_H_

#include "p-ipeps/config.h"
#include <iomanip>
#include <cmath>
#include <limits>
#include "p-ipeps/models.h"
#include "p-ipeps/mpo.h"
#include "p-ipeps/ctm-cluster.h"
#include "p-ipeps/ctm-cluster-global.h"
#include "p-ipeps/su2.h"
#include "itensor/all.h"

// ----- 2-Site operator functions ------------------------------------

/*
 * Apply 2-site MPO to T1 and T2 tensors connected through weight L
 *
 */
void applyH_T1_L_T2(MPO_2site const& mpo2s, 
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & L,
	itensor::ITensor & LI, bool dbg = false);

void applyH_T1_L_T2_v2(MPO_2site const& mpo2s, 
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & L,
	bool dbg = false);

void applyH_T1_L_T2_v2_notReduceTensors(MPO_2site const& mpo2s, 
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & L,
	bool dbg = false);


// ----- 2-Site operator functions ------------------------------------

// ----- 3-Site operator functions ------------------------------------

/*
 * Apply MPO_3site over three sites ABC
 *
 */
void applyH_123_v1(MPO_3site const& mpo3s, 
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & T3, 
	itensor::ITensor & l12, itensor::ITensor & l23, bool dbg = false);

void applyH_123_v2(MPO_3site const& mpo3s,
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & T3, 
	itensor::ITensor & l12, itensor::ITensor & l23, itensor::Args const& args);

void applyH_123_v3(MPO_3site const& mpo3s, 
	itensor::ITensor & T1, itensor::ITensor & T2, itensor::ITensor & T3, 
	itensor::ITensor & l12, itensor::ITensor & l23,
	itensor::ITensor & l12I, itensor::ITensor & l23I, bool dbg = false);

// ----- 3-Site operator functions ------------------------------------

itensor::Args simpleUpdate(MPO_2site const& u12, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args simpleUpdate(MPO_3site const& u123, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::Args simpleUpdate(OpNS const& u12, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	itensor::Args const& args = itensor::Args::global());

itensor::ITensor getInvDiagT(itensor::ITensor const& t);

#endif
