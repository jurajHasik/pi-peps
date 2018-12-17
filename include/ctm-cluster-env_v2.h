//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __CTM_CLS_ENV_H_
#define __CTM_CLS_ENV_H_

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "ctm-cluster.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster-global.h"
#include "itensor-svd-solvers.h"
#include "itensor/all.h"


class CtmEnv
{
    // ########################################################################
    // Additional structures
    public:

    const Cluster * p_cluster;

    itensor::SvdSolver & solver;

    enum class DIRECTION {
        LEFT = 0, UP = 1, RIGHT = 2, DOWN = 3
    };

    typedef enum INIT_ENV {
        INIT_ENV_const1,
        INIT_ENV_rnd,
        INIT_ENV_ctmrg,
        INIT_ENV_obc,
        INIT_ENV_pwr,
        INIT_ENV_file
    } init_env_type;

    typedef enum ISOMETRY {
        ISOMETRY_T1,
        ISOMETRY_T2,
        ISOMETRY_T3,
        ISOMETRY_T4
    } isometry_type;

    typedef enum NORMALIZATION {
        NORM_BLE,
        NORM_PTN
    } normalization_type;

    // Holding the current spectrum of singular values of C_*
    struct CtmSpec {
        std::vector< std::vector<double> > spec_clu;
        std::vector< std::vector<double> > spec_cru;
        std::vector< std::vector<double> > spec_crd;
        std::vector< std::vector<double> > spec_cld;
    };

    // ########################################################################
    // data holding the environment
    bool DBG = false;
    int  DBG_LVL = 0;

    std::string m_name;
    bool default_pinv_cutoff  = true;
    double isoPseudoInvCutoff = 1.0e-14;
    double isoMinElemWarning  = 1.0e-4;
    double isoMaxElemWarning  = 1.0e4;
    std::string SVD_METHOD = "itensor";
    int rsvd_power = 2;
    int rsvd_reortho = 1;
    int rsvd_oversampling = 10;
    
    /*
     * Auxiliary dimension of bond-indices of on-site tensor TT given by 
     * contraction of physical index "s" of bra & ket tensor T* resp. T  
     *
     * given:       contract "s":
     *    
     *         D D                D D     D D            D^2
     *         |/                  \|      \|             |
     *     D--|T|~~s           D--|T*|~~s~~|T|--D = D^2--|TT|--D^2
     *         |                    |       |             |
     *         D                    D       D            D^2
     *
     * And call D^2 = d. Hence dimension auxBondDim == d
     *
     */
    int d;

    /*
     * Auxiliary dimension of the environment - dimension
     * of indices connection corner & half-row/column tensor 
     * auxEnvDim == x
     *
     */
    int x;

    int sizeN, sizeM; // (N)rows x (M)columns of cluster of sites
    /*
     * Holds the following tensor networks - environment
     * of for each non-equivalent sites within cluster
     *
     *      <--   M coord   -->       
     *    ______    _____    _______
     *   |C_LUxy|--|T_Uxy|--|C_RU xy|      A
     *    _|____    _|___     |            |
     *   |T_L0xy|--| X_xy|--|T_Rxy|     N coord
     *    _|____    _|___     |            |
     *   |C_LDxy|--|T_Dxy|--|C_RDxy|       V 
     *
     * where C_* are corner matrices
     *       T_* are row/column tensors
     *       X** is an on-site tensor formed by contraction of given 
     *           tensor T* (bra) & T (ket) through physical index
     *
     * NOTE: Cluster AB has just two non-equivalent sites, even though 
     *               BA 
     *       sizeX x sizeY = 4
     *              
     */

    // holds inequivalent on-site tensors (after contraction) within cluster 
    std::vector< itensor::ITensor > sites;
    std::vector< std::string > siteIds;

    // map from cluster to sites
    std::map< std::pair<int,int>, int> cToS;

    // arrays holding half-row/column tensors
    std::vector< itensor::ITensor > T_U, T_R, T_D, T_L;

    // corner tensors
    std::vector< itensor::ITensor > C_LD, C_LU, C_RU, C_RD;
   
    // aux indices of environment tensors (of auxEnvDim == x)
    itensor::Index I_U, I_R, I_D, I_L;
    
    // from C_LU corner in counter-clockwise fashion
    // C_LU--0--T_U--1--C_RU
    //  7                2
    // T_L              T_R
    //  6                3  
    // C_LD--5--T_D--4--C_RD
    //
    // mapping to 0->I_U0, 1->I_U1, 2->I_R0, 3->I_R1, 4->I_D1, 5->I_D0,
    //            6->I_L1, 7->I_L0 
    std::map<std::string, std::vector<itensor::Index> > eaux; // environment aux indices

    std::vector<itensor::Index> envIndPair(
        std::string const& id0, int i0, 
        std::string const& id1, int i1) const {
        return { eaux.at(id0)[i0], eaux.at(id1)[i1] };
    }

    itensor::ITensor DContractEnv(
        std::string const& id0, int i0, 
        std::string const& id1, int i1) const {
        auto tmp = envIndPair(id0,i0,id1,i1);
        return itensor::delta(tmp[0],tmp[1]);
    }

    // corner_index maps to
    // 0|1 => C_LU|C_RU
    // 3|2 => C_LD|C_RD
    std::vector<itensor::Index> envIndCornerPair(
        std::string const& id, int ci) const {
        int i1 = ci * 2;
        int i2 = (ci * 2 + 7) % 8;
        return { eaux.at(id)[i1], eaux.at(id)[i2] };
    }

    // aux bond indices of sites ( of auxBondDim == d)
    itensor::Index I_XH, I_XV;
    
    
    // vector indexes combiners 0 1 2 3 in respect to four directions on
    // lattice L, U, R, D for each site
    std::map< std::string, std::vector<itensor::ITensor> > CMB;
    // TODO
    // since combiner cant be contracted with delta we have to
    // keep the map from directions to fused site indices I_XH and I_XV
    // L->0->I_XH, U->1->I_XV, R->2->prime(I_XH), D->3->prime(I_XV)
    std::vector<itensor::Index> fusedSiteI;
    std::map< std::string, std::vector<itensor::Index> > faux; // fused index

    // itensor::ITensor DContractSiteBraKet(Vertex const& v, int dir) const {
    //     return delta(CMB.at(id)[dir],faux.at(id)[dir]);
    // }

    itensor::ITensor DContractCmbEnvInd(std::string const& id, int dir) const {
        return itensor::delta(combinedIndex(CMB.at(id)[dir]),faux.at(id)[dir]);
    }

    // simple wrapper around spectra of corner matrices 
    CtmSpec spec;

    // ########################################################################
    // member methods of CtmEnv
    
    // Default constructor
    //CtmEnv();

    // Basic constructor 
    CtmEnv(std::string t_name, int t_x, Cluster const& c,
        itensor::SvdSolver & solver, 
        itensor::Args const& args = itensor::Args::global());

    // Reconstructs CtmEnv from given CtmData - a CtmEnv for each
    // of non-equivalent sites within cluster (compatibility with
    // LEGACY ctm-cluster-io.h )
    CtmEnv(std::string t_name, std::vector<CtmData> const& ctmD, 
        Cluster const& c, itensor::SvdSolver & solver,
        itensor::Args const& args = itensor::Args::global()); 

    /*
     * Return tensor resulting from contraction of on-site tensor T (ket)
     * with its T* (bra) counterpart through physical index and fuse aux-bonds
     * of bar and ket parts
     * If expose = true, the physical index is not contracted, but aux-bonds are 
     * fused
     */
    itensor::ITensor contractOST(itensor::ITensor const& T,
        bool expose = false) const;

    // ########################################################################
    // environment initalization methods

    // Init environment with simple tensors - every element is set to 1+0i
    void initMockEnv();
    
    // Init environment with random R/C tensors elements depending on isComplex
    void initRndEnv(bool isComplex);

    // Init environment from files with tensors stored in format
    // specified by IO_ENV_FMT
//    void initFromFile(int chi, IO_ENV_FMT ioFmt, std::string TAG);

    void initCtmrgEnv(bool dbg = false);

    void initOBCEnv(bool dbg = false);

    void initPWREnv(bool dbg = false);

    void init(INIT_ENV initEnvType, bool isComplex = false, 
        bool dbg = false);

    void symmetrizeEnv(bool dbg = false);

    void testCtmrgEnv();

    // ########################################################################
    // CTM iterative methods

    void move_unidirectional(unsigned int direction, ISOMETRY iso_type,
        Cluster const& c, std::vector<double> & accT);

    void move_singleDirection(unsigned int direction, ISOMETRY iso_type,
        Cluster const& c, std::vector<double> & accT);

    // ########################################################################
    // isometries
    
    void compute_IsometriesT3(unsigned int direction, Cluster const& c,
        itensor::Index & ip, itensor::Index & ipt, itensor::Index & ia,
        std::vector<itensor::ITensor> & P, 
        std::vector<itensor::ITensor> & Pt,
        std::vector<double> & accT) const;

    void compute_IsometriesT4(unsigned int direction, Cluster const& c,
        itensor::Index & ip, itensor::Index & ipt, itensor::Index & ia,
        std::vector<itensor::ITensor> & P, 
        std::vector<itensor::ITensor> & Pt,
        std::vector<double> & accT) const;

    // build reduced density matrix of 2x2 cluster with cut(=uncontracted
    // indices) along one of the CTM directions U,R,D or L starting from
    // position (col,row), where starting site is always nearest site in
    // clock-wise direction wrt. to cut
    // CONVENTION indices clockwise wrt. to cut have primeLevel 0
    itensor::ITensor build_2x2_RDM(char ctmMove, int col, int row) const;

    void build_halves_V2(
        unsigned int direction, Cluster const& c, Vertex const& v,
        itensor::ITensor & H, itensor::ITensor & Ht) const;

    // builds the corner of environment of site (col,row) + site where 
    // corner is 1,2,3 or 4 according to following key 1|2
    //                                                 4|3
    itensor::ITensor build_corner_V2(unsigned int direction, 
        Cluster const& c, Vertex const& v) const;

    // ########################################################################
    // environment normalization methods

    double getNorm() const;

    void normalizeBLE();

    void normalizeBLE_ctmStep(char ctmMove, int col, int row, bool dbg = false);

    void normalizePTN();

    void normalizePTN_ctmStep(char ctmMove, int col, int row);

    // ########################################################################
    // methods handling SVD spectra of corner matrices

    // Computes and stores new SVD spectra of canonical C_* in spec
    /* Where canonical C_* are one of the environment surrounding
     * original cluster
     *
     */
    void computeSVDspec();

    // Print stored SVD spectrum of corner matrices C_*
    void printSVDspec() const;

    // Compute the spectral dist between stored spectrum
    // in spec_c* and current spectrum
    //void specDist(CtmSpec const& s1, CtmSpec const& s2) const;

    // Update original cluster with new one of same type
    void updateCluster(Cluster const& c);

    CtmSpec getCtmSpec() const;

    // DEPRECATED
    // Compatibility with LEGACY ctm-cluster-io.h, cluster-ev-builder.h
    /* Store canonical environment = one of the original cluster
     */
    CtmData getCtmData() const;
    // DEPRECATED
    CtmData getCtmData_DBG() const;

    /* 
     * Export Full environment of cluster - C's and T's for every 
     * non-equivalent lattice site
     *
     */
    CtmData_Full getCtmData_Full_DBG(bool dbg = false) const;

    std::ostream& print(std::ostream& s) const;
};

// ############################################################################
// IO toString methods

/* 
 * Defining these methods enables printing of 
 *     CtmEnv, CtmEnv::CtmSpec 
 * objects using 
 *     cout << o ;
 * where o is an instance of one of mentioned objects
 *
 * It also allows printing using the print,println,printf, and
 * printfln functions. Use the "%s" flag to print custom objects
 * with printf and printfln.
 *
 */
std::ostream&
operator<<(std::ostream& s, CtmEnv const& c);

std::ostream&
operator<<(std::ostream& s, CtmEnv::CtmSpec const& spec);

// string to enum conversion
CtmEnv::INIT_ENV toINIT_ENV(std::string const& inEnv);

CtmEnv::ISOMETRY toISOMETRY(std::string const& isoType);

CtmEnv::NORMALIZATION toNORMALIZATION(std::string const& normType);

#endif
