#ifndef __CLS_EV_BUILDER_
#define __CLS_EV_BUILDER_

#include <string>
#include <iostream>
#include "ctm-cluster-env.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster.h"
#include "ctm-cluster-global.h"
#include "su2.h"
#include "itensor/all.h"

struct MpoNS {
    // number of sites over which this MPO acts
    int nSite;

    // individual MPOs
    std::vector<itensor::ITensor> mpo;

    // siteId of sites on which the MPOs are constructed
    std::vector<std::string> siteIds;
};

class EVBuilder {

    std::string name;

    // Cluster
    Cluster cls;

    // With Environment Data one can compute expectation values
    CtmData cd;

    public:

    // Default Constructor
    EVBuilder();

    // Basic Constructor
    EVBuilder(std::string in_name, Cluster const& in_cls, CtmData const& in_cd);

    // Supported types of 1-site operators
    enum MPO_1S {
        MPO_Id,     // Identity
        MPO_S_Z,    // Projection on S_z
        MPO_S_Z2,   // S_z^2
        MPO_S_P,    // S_plus
        MPO_S_M     // S_minus
    };

    // Get on-site contracted tensor <T(bra)|MPO|T(ket)>,
    // with prime level "l"
    MpoNS getTOT_DBG(MPO_1S mpo, std::string siteId, int primeLvl) const;
        
    // Compute expectation value of 1-site operator O
    /*
     *  Arg op - result of getTOT = <bra|op|ket> with indices in accordance
     *           to definition of TN
     *  Arg site - defines the position within cluster, where original on-site
     *             tensor is replaced with tensor op
     *  
     */
    double eV_1sO(MpoNS const& op, std::pair<int,int> site) const;

    // Supported types of 2-site operators
    enum OP_2S {
        OP2S_Id,        // Identity
        OP2S_AKLT_S2_H, // Hamiltonian - P(rojector)^{S=4} on S=4 subspace
        OP2S_SS         // Hamiltonian - NN-Heisenberg
    };

    // Get 2-site operator as a pair of 1-site operators with auxiliary
    // index coming from SVD of the original 2-site op
    std::pair< itensor::ITensor,itensor::ITensor > get2STOT_DBG(OP_2S op2s,
        itensor::ITensor const& TA, itensor::ITensor const& TB) const;

    std::pair< itensor::ITensor,itensor::ITensor > get2STOT(OP_2S op2s,
        itensor::ITensor const& TA, itensor::ITensor const& TB) const;

    // Compute expectation value of 2-site operator O given
    // by its decomposition into MPOs OA * OB = O placed on siteA and siteB 
    // within the cluster surrounded by environment
     double eV_2sO_DBG(
        std::pair< itensor::ITensor,itensor::ITensor > const& Op,
        std::pair<int,int> siteA, std::pair<int,int> siteB) const;

    double eV_2sO(
        std::pair< itensor::ITensor,itensor::ITensor > const& Op,
        std::pair<int,int> siteA, std::pair<int,int> siteB) const;

    // compute norm (contract TN) of NxM==(N,M) copies of cluster
    // surrounded by environment
    double getNormSupercell_DBG(std::pair<int,int> sc) const;

    double getNormSupercell(std::pair<int,int> sc) const;

    void setCtmData(CtmData const& new_cd);

    // Correlation function
    // Compute expectation value of two 1-site operators O1, O2
    // spaced by "dist" sites in horizontal direction
    /*  _    _    _         _    _    _
     * |C|--|T|--|T|--...--|T|--|T|--|C|
     *  |    |    |         |    |    |
     * |T|--|O1|-|X|--...--|X|-|O2|--|T| 
     *  |    |    |         |    |    |
     * |C|--|T|--|T|--...--|T|--|T|--|C| 
     *           
     *           <--"dist"-->
     * Hence "dist" = 0, means adjacent sites 
     *
     */
   // std::complex<double> expVal_1sO1sO_H(int dist, 
    //        itensor::ITensor const& op1, itensor::ITensor const& op2);

    // Correlation function
    // Compute expectation value of two 1-site operators O1, O2
    // spaced by "dist" sites in vertical direction
    /*  _    _    _  
     * |C|--|T|--|C|
     *  |    |    |   
     * |T|--|O1|-|T|
     *  |    |    |  
     * |T|--|X|--|T|   ^
     *  |    |    |    |
     *      ...      "dist"
     *  |    |    |    |
     * |T|--|X|--|T|   V
     *  |    |    |
     * |T|--|O2|-|T|
     *  |    |    |
     * |C|--|T|--|C| 
     *     
     * Hence "dist" = 0, means adjacent sites 
     *
     */
  //  std::complex<double> expVal_1sO1sO_V(int dist, 
   //         itensor::ITensor const& op1, itensor::ITensor const& op2);

    // Correlation function
    // Compute expectation value of two 2-site operators O1, O2
    // spaced by "dist" sites in horizontal direction
    /*  _    _    _         _    _    _
     * |C|--|T|--|T|--...--|T|--|T|--|C|
     *  |    |_   |         |   _|    |
     * |T|--|  |-|X|--...--|X|-|  |--|T|
     *  |   |O1|  |         |  |O2|    |
     * |T|--| _|-|X|--...--|X|-|_ |--|T|
     *  |    |    |         |    |    |
     * |C|--|T|--|T|--...--|T|--|T|--|C|
     *           
     *           <--"dist"-->
     * Hence "dist" = 0, means adjacent sites 
     *
     */
   // std::complex<double> expVal_2sOV2sOV_H(int dist, 
   //         Mpo2S const& op1, Mpo2S const& op2);

    // Correlation function
    // Compute expectation value of two 2-site operators O1, O2
    // spaced by "dist" sites in horizontal direction
    /*  _    _    _    _         _    _    _    _
     * |C|--|T|--|T|--|T|--...--|T|--|T|--|T|--|C|
     *  |    |____|    |         |    |____|    |
     * |T|--|__OP1_|--|X|--...--|X|--|_OP2__|--|T|
     *  |    |    |    |         |    |    |    |
     * |C|--|T|--|T|--|T|--...--|T|--|T|--|T|--|C|
     *           
     *           <--"dist"-->
     * Hence "dist" = 0, means adjacent sites 
     *
     */
  //  std::complex<double> expVal_2sOH2sOH_H(int dist, 
  //          Mpo2S const& op1, Mpo2S const& op2);


    /*
     * wrapper around SU2_getSpinOp(SU2O su2o, itensor::Index const& s)
     * from su2.h
     *
     */
    itensor::ITensor getSpinOp(MPO_1S mpo, itensor::Index const& s) const;

    std::ostream& print(std::ostream& s) const;
};

std::ostream&
operator<<(std::ostream& s, EVBuilder const& ev);

#endif
