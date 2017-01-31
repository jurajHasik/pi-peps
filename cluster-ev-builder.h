#ifndef __CLS_EV_BUILDER_
#define __CLS_EV_BUILDER_

#include <string>
#include <iostream>
#include "su2.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster.h"
#include "ctm-cluster-global.h"
#include "itensor/all.h"

class EVBuilder {

    std::string name;

    // Cluster
    Cluster cls;

    // With Environment one can compute expectation values
    CtmEnv env;

    public:

    // Default Constructor
    EVBuilder();

    // Basic Constructor
    EVBuilder(std::string in_name, Cluster const& in_cls, CtmEnv const& in_env);

    // Supported types of 1-site operators
    // NOTE: They are part of global(!) namespace
    enum MPO {
        MPO_Id,     // Identity
        MPO_S_Z,    // Projection on S_z
        MPO_S_Z2,   // S_z^2
        MPO_S_P,    // S_plus
        MPO_S_M     // S_minus
    };

    // Get on-site contracted tensor <T(bra)|MPO|T(ket)>,
    // with prime level "l",
    /*
     * TODO include (arbitrary) rotation matrix on physical index of 
     *      on-site tensor T as argument
     *
     */
    itensor::ITensor getTOT_DBG(MPO mpo, itensor::ITensor const& T,
        int primeLvl) const;
        
    // Compute expectation value of 1-site operator O
    /*
     *  Arg op - result of getTOT = <bra|op|ket> with indices in accordance
     *           to definition of TN
     *  Arg site - defines the position within cluster, where original on-site
     *             tensor is replaced with tensor op
     *  
     */
    double eV_1sO(itensor::ITensor const& op, std::pair<int,int> site) const;

    /*
     * Struct holding 2-site operator as decomposition into
     * 2 1-site (MPO) operators with additional index
     *
     *  _|__|      |     | 
     * |_2OP_| == |OA|--|OB|
     *   |  |      |     |
     *
     */
    struct Mpo2S {
        itensor::ITensor opA, opB;
    };

    enum OP2S {
        OP2S_Id,        // Identity
        OP2S_AKLT_S2_H, // Hamiltonian - P(rojector)^{S=4} on S=4 subspace
        OP2S_SS         // Hamiltonian - NN-Heisenberg
    };

    //Mpo2S get2STOT(OP2S op2s, itensor::Index const& i_Xh, 
   //         itensor::Index const& i_Xv);

    // Compute expectation value of 2-site operator O given
    // by its decomposition into MPOs OA * OB = O
    /*
     *  |___|     |     |
     * |__O__| = |OA|--|OB|
     *  |   |     |     |  
     *
     * Arg opA - contracted X_opA <bra|OA|ket> with
     *           extra auxiliary index        
     * Arg opB - contracted X_opB <bra|OB|ket> with
     *           extra auxiliary index
     *
     */
   // std::complex<double> expVal_2sO(itensor::ITensor const& opA,
    //        itensor::ITensor const& opB);

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

    // itensor::ITensor getSpinOp(ExpValBuilder::MPO mpo, int l, 
    //         itensor::Index const& i_braS, itensor::Index const& i_ketS,
    //         bool isB);

    std::ostream& print(std::ostream& s) const;
};

std::ostream&
operator<<(std::ostream& s, EVBuilder const& ev);

std::ostream&
operator<<(std::ostream& s, EVBuilder::Mpo2S const& mpo2S);

#endif
