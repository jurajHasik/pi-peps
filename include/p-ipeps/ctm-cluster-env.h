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
#include "p-ipeps/ctm-cluster-io.h"
#include "p-ipeps/ctm-cluster-global.h"
#include "itensor/all.h"

class CtmEnv
{
    // ########################################################################
    // Additional structures
    public:

    typedef enum INIT_ENV {
        INIT_ENV_const1,
        INIT_ENV_rnd,
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
        std::vector< double > spec_clu;
        std::vector< double > spec_cru;
        std::vector< double > spec_crd;
        std::vector< double > spec_cld;
    };

    // ########################################################################
    // Private data holding the environment
//    private:

    std::string m_name;
    
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

    /*
     * Holds following the tensor network
     *    ____    _____         _______
     *   |C_LU|--|T_U0 |--...--|T_Um-1 |--|C_RU|
     *    _|__    _|___         _|_____     |
     *   |T_L0|--| X00 |--...--| X0m-1 |--|T_R0|  
     *     |                                | 
     *    ...                              ...
     *     |                                |
     * |T_Ln-1|--|Xn-10|--...--|Xn-1m-1|--|T_Rn-1|       
     *    _|__    _|___          |          | 
     *   |C_LD|--|T_D0 |--...--|T_Dm-1 |--|C_RD|
     *
     * where C_* are corner matrices
     *       T_* are row/column tensors
     *       X** is an on-site tensor formed by contraction of given 
     *           tensor T* (bra) & T (ket) through physical index
     */
    int sizeN, sizeM; // row x columns of cluster of sites

    // holds inequivalent on-site tensors (after contraction) within cluster 
    std::vector< itensor::ITensor > sites;
    
    // map from cluster to sites
    std::map< std::pair<int,int>, int> cToS;

    // arrays holding half-row/column tensors
    std::vector< itensor::ITensor > T_U;
    std::vector< itensor::ITensor > T_R;
    std::vector< itensor::ITensor > T_D;
    std::vector< itensor::ITensor > T_L;

    // corner tensors
    itensor::ITensor C_LD, C_LU, C_RU, C_RD;
   
    // aux indices of environment tensors (of auxEnvDim == x)
    itensor::Index I_U, I_R, I_D, I_L;

    // aux bond indices of sites ( of auxBondDim == d)
    itensor::Index I_XH, I_XV;

    // simple wrapper around spectra of corner matrices 
    CtmSpec spec;

    // ########################################################################
    // member methods of CtmEnv
//    public:
    
    // Default constructor
    CtmEnv();

    // Basic constructor 
    CtmEnv(std::string t_name, int t_x, Cluster const& c);

    // Reconstructs CtmEnv from given CtmData
    CtmEnv(std::string t_name, CtmData const& ctmD, Cluster const& c); 

    /*
     * Return tensor resulting from contraction of on-site tensor T (ket)
     * with its T* (bra) counterpart through physical index
     *
     */
    itensor::ITensor contractOST(itensor::ITensor const& T) const;

    // ########################################################################
    // environment initalization methods

    // Init environment with simple tensors - every element is set to 1+0i
    void initMockEnv();
    
    // Init environment with random R/C tensors elements depending on isComplex
    void initRndEnv(bool isComplex);

    // Init environment from files with tensors stored in format
    // specified by IO_ENV_FMT
//    void initFromFile(int chi, IO_ENV_FMT ioFmt, std::string TAG);

    // ########################################################################
    // CTM iterative methods

    // Insert, Absorb & Renormalize U(p) row 
    void insURow_DBG(ISOMETRY iso_type, NORMALIZATION norm_type, 
        std::vector<double> & accT);
    void insURow(ISOMETRY iso_type, NORMALIZATION norm_type);
    
    // -||- D(own) row
    void insDRow_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT);
    void insDRow(ISOMETRY iso_type, NORMALIZATION norm_type);
    
    // Insert, Absorb & Renormalize L(eft) column
    void insLCol_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT);
    void insLCol(ISOMETRY iso_type, NORMALIZATION norm_type);
    
    // -||- R(ight) column
    void insRCol_DBG(ISOMETRY iso_type, NORMALIZATION norm_type,
        std::vector<double> & accT);
    void insRCol(ISOMETRY iso_type, NORMALIZATION norm_type);

    // ########################################################################
    // isometries

    // NOT WORKING in current form
    std::pair< itensor::ITensor,itensor::ITensor > isoT1(
        itensor::IndexSet const& iS_tU, 
        std::pair< itensor::Index, itensor::Index > const& iS_delta,
        itensor::ITensor const& t1, itensor::ITensor const& t2) const;

    /*
     * Isometry type 2
     *
     * Compute half-row/column reduced density matrix and perform an 
     * SVD over indices, increased in dimension by row/column absorption
     * Use part of obtained eigenvectors to reduce dimension of new C's 
     * and T's
     * 
     */
    void isoT2(char ctmMove);

    /*
     * Isometry type 3
     *
     * Perform an eigenvalue decomp of t1 t1^dag + t2 t2^dag (thus 
     * hermitian) and use obtained eigenvectors (corresponding to the 
     * chi highest eigenvalues) to reduce the environment tensors
     *
     */
    void isoT3(char ctmMove);

    /*
     * Isometry type 4 - 2X2 CLUSTER
     *
     * Perform an eigenvalue decomp of t1 t1^dag + t2 t2^dag (thus 
     * hermitian) and use obtained eigenvectors (corresponding to the 
     * chi highest eigenvalues) tZ, tZstar to reduce the two corner 
     * tensor C_*
     *
     * Combine unreduced C_*s and T_*s to obtain second isometry
     * tW,tWstar and use both tW,tWstar AND tZ,tZstar to reduce
     * half-row/column tensors
     * 
     */
    void isoT4(char ctmMove);

    // ########################################################################
    // environment normalization methods

    // print (Euclidean) norms of C_* & T_*
//    void printENorms(std::string tag);

    double getNorm() const;

    void normalizeBLE();

    void normalizeBLE_ctmStep(char ctmMove);

    void normalizePTN();

    void normalizePTN_ctmStep(char ctmMove);

//    void normalizeCs();

//    void normalize();

    // ########################################################################
    // methods handling SVD spectra of corner matrices

    // Computes and stores new SVD spectra of C_* in spec
    void computeSVDspec();

    // Print stored SVD spectrum of corner matrices C_*
    void printSVDspec() const;

    // Compute the spectral dist between stored spectrum
    // in spec_c* and current spectrum
    void specDist(CtmSpec const& s1, CtmSpec const& s2) const;

//    itensor::Index const& getIXH() const;
//    itensor::Index const& getIXV() const;

    CtmSpec getCtmSpec() const;

    CtmData getCtmData() const;

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
