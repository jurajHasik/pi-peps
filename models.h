#ifndef __MODELS_
#define __MODELS_

#include <string>
#include "json.hpp"
#include "su2.h"
#include "ctm-cluster.h"
#include "ctm-cluster-global.h"
#include "itensor/all.h"

// ----- Main MPO Structures ------------------------------------------
// Index names of 3-site MPO indices
const std::string TAG_MPO3S_PHYS1 = "I_MPO3S_S1";
const std::string TAG_MPO3S_PHYS2 = "I_MPO3S_S2";
const std::string TAG_MPO3S_PHYS3 = "I_MPO3S_S3";
const std::string TAG_MPO3S_12LINK = "I_MPO3S_L12";
const std::string TAG_MPO3S_23LINK = "I_MPO3S_L23";

// types for auxiliary indices of MPO tensors
const auto MPOLINK = itensor::IndexType(TAG_IT_MPOLINK);

/*
 * this struct holds instance of particular 3-site MPO composed
 * by three tensors
 *
 *    s1'              s2'              s3'
 *    |                |                | 
 *   |H1|--I_MPO3s12--|H2|--I_MPO3s23--|H3|
 *    |                |                | 
 *    s1               s2               s3
 *
 * exposing the physical indices s1,s2,s3
 *
 */
struct MPO_3site {
    itensor::ITensor H1, H2, H3;

    // expose internal indices
    itensor::Index a12, a23;

    // expose physical indices
    itensor::Index Is1, Is2, Is3;
};

struct MPO_2site {
    itensor::ITensor H1, H2;

    // expose physical indices
    itensor::Index Is1, Is2;
};
// ----- END Main MPO Structures --------------------------------------

// ----- Trotter gates (2Site, 3site, ...) MPOs -----------------------
MPO_3site symmMPO3Sdecomp(itensor::ITensor const& u123, 
    itensor::Index const& s1, itensor::Index const& s2, 
    itensor::Index const& s3);

MPO_3site ltorMPO3Sdecomp(itensor::ITensor const& u123, 
    itensor::Index const& s1, itensor::Index const& s2, 
    itensor::Index const& s3);

MPO_3site ltorMPO2StoMPO3Sdecomp(itensor::ITensor const& u123, 
    itensor::Index const& s1, itensor::Index const& s2);

MPO_3site getMPO3s_Id(int physDim);

MPO_3site getMPO3s_Id_v2(int physDim, bool dbg = false);

MPO_3site getMPO3s_Uj1j2(double tau, double J1, double J2);

MPO_3site getMPO3s_Uj1j2_v2(double tau, double J1, double J2, double lambda);

MPO_3site getMPO3s_Uladder(double tau, double J, double Jp);

MPO_3site getMPO3s_Uladder_v2(double tau, double J, double Jp);

MPO_3site getMPO3s_Ising_v2(double tau, double J, double h);

MPO_3site getMPO3s_Ising_2site(double tau, double J, double h);
// ----- END Trotter gates (2Site, 3site, ...) MPOs -------------------

// ----- Model Definitions --------------------------------------------
void getModel_J1J2(nlohmann::json & json_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel_NNHLadder(nlohmann::json & json_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel_Ising(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);
// ----- END Model Definitions ----------------------------------------


#endif