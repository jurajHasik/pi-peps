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
#include "ctm-cluster-io.h"
#include "ctm-cluster-global.h"
#include "itensor-svd-solvers.h"
#include "rsvd-solver.h"
#include "itensor/all.h"

class CtmEnv
{
    // ########################################################################
    // Additional structures
    public:

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
    double isoPseudoInvCutoff = 1.0e-14;
    double isoMinElemWarning  = 1.0e-4;
    double isoMaxElemWarning  = 1.0e4;
    std::string SVD_METHOD = "itensor";
    int rsvd_power = 2;
    int rsvd_reortho = 1;
    
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

    // aux bond indices of sites ( of auxBondDim == d)
    itensor::Index I_XH, I_XV;

    // simple wrapper around spectra of corner matrices 
    CtmSpec spec;

    // ########################################################################
    // member methods of CtmEnv
    
    // Default constructor
    CtmEnv();

    // Basic constructor 
    CtmEnv(std::string t_name, int t_x, Cluster const& c, 
        itensor::Args const& args = itensor::Args::global());

    // Reconstructs CtmEnv from given CtmData - a CtmEnv for each
    // of non-equivalent sites within cluster (compatibility with
    // LEGACY ctm-cluster-io.h )
    CtmEnv(std::string t_name, std::vector<CtmData> const& ctmD, 
        Cluster const& c); 

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

    void symmetrizeEnv(bool dbg = false);

    void testCtmrgEnv();

    // ########################################################################
    // CTM iterative methods

    // Insert, Absorb & Renormalize U(p) row 
    void insURow_DBG(ISOMETRY iso_type, NORMALIZATION norm_type, 
        std::vector<double> & accT, bool dbg = false);
    // void insURow(ISOMETRY iso_type, NORMALIZATION norm_type);
    
    // -||- R(ight) column
    void insRCol_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT, bool dbg = false);
    // void insRCol(ISOMETRY iso_type, NORMALIZATION norm_type);

    // -||- D(own) row
    void insDRow_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT, bool dbg = false);
    // void insDRow(ISOMETRY iso_type, NORMALIZATION norm_type);
    
    // Insert, Absorb & Renormalize L(eft) column
    void insLCol_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT, bool dbg = false);
    // void insLCol(ISOMETRY iso_type, NORMALIZATION norm_type);

    // ########################################################################
    // isometries
    
    std::vector<itensor::ITensor> isoT1(char ctmMove, int col, int row);
    std::vector<itensor::ITensor> isoT2(char ctmMove, int col, int row,
        std::vector<double> & accT, bool dbg = false);
    std::vector<itensor::ITensor> isoT3(char ctmMove, int col, int row,
        std::vector<double> & accT, bool dbg = false);
    std::vector<itensor::ITensor> isoT4(char ctmMove, int col, int row,
        std::vector<double> & accT, bool dbg = false);

    // build reduced density matrix of 2x2 cluster with cut(=uncontracted
    // indices) along one of the CTM directions U,R,D or L starting from
    // position (col,row), where starting site is always nearest site in
    // clock-wise direction wrt. to cut
    // CONVENTION indices clockwise wrt. to cut have primeLevel 0
    itensor::ITensor build_2x2_RDM(char ctmMove, int col, int row) const;

    std::pair<itensor::ITensor, itensor::ITensor> build_halves(
        char ctmMove, int col, int row, bool dbg = false) const;

    // builds the corner of environment of site (col,row) + site where 
    // corner is 1,2,3 or 4 according to following key 1|2
    //                                                 4|3
    itensor::ITensor build_corner(char corner, int col, int row,
        bool dbg = false) const;

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
