#ifndef __CTM_CLS_IO_
#define __CTM_CLS_IO_

#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include "json.hpp"
#include "ctm-cluster-global.h"
#include "itensor/all.h"

const std::string WS4(4,' ');

typedef enum IO_ENV_FMT {
    IO_ENV_FMT_txt,
    IO_ENV_FMT_bin
} io_env_fmt_type;

// string to enum conversion
IO_ENV_FMT toIO_ENV_FMT(std::string const& ioFmt);

// ############################################################################
// IO for cluster definition using JSON data format

/*
 * Struct holding the raw environment
 * 
 */
struct Cluster {
    // size of n x m cluster
    int sizeN, sizeM;
    
    // auxiliary bond dimension
    int auxBondDim;
    // dimension of local Hilbert space = dimension of physical index
    int physDim;

    // siteIds
    std::vector< std::string > siteIds;

    // inequivalent sites
    std::map< std::string, itensor::ITensor > sites;

    // map from inequivalent sites to cluster sites
    std::map< std::pair< int,int >, std::string > cToS; 
};

Cluster readCluster(std::string const& filename);

/* 
 * Read elements of an on-site tensor T on a SQUARE lattice
 *
 *           I(s) 
 *      I(v) /
 *        | /
 * I(h)--|T|--I(h)'
 *        |
 *      I(v)'
 * 
 * where the indices h, h', v, v' denote bond indices within
 * a square lattice and index "s" is a physical index enumerating
 * local Hilbert space
 *
 * The input for tensor elements is expected as string per non-zero 
 * tensor element in format
 *
 * s h v h' v' Re(T) Im(T) <-> ids[0..4] Re(val) Im(val) 
 *
 */
itensor::ITensor readOnSiteT(nlohmann::json const& j);

// ############################################################################
// IO for environment of nxm cluster

/*
 * Struct holding the raw environment
 *
 * TODO possibly redundant entries for size of the cluster
 * 
 */
struct CtmData {
    int auxDimEnv;
    int auxDimSite;

    int sizeN, sizeM; // size of n(row) x m(columns) cluster

    // arrays holding half-row/column tensors
    std::vector< itensor::ITensor > T_U;
    std::vector< itensor::ITensor > T_R;
    std::vector< itensor::ITensor > T_D;
    std::vector< itensor::ITensor > T_L;

    // corner tensors
    itensor::ITensor C_LU, C_RU, C_RD, C_LD;
   
    itensor::Index I_U, I_R, I_D, I_L;
    itensor::Index I_XH, I_XV;
};

/*
 * store/read environment to/from file(s)
 *
 */
void writeEnv(IO_ENV_FMT ioFmt, std::string TAG, CtmData const& ctmD);

/*
 * store/read environment to/from file(s)
 *
 */
CtmData readEnv(IO_ENV_FMT ioFmt, std::string const& TAG, Cluster const& cls);

// ############################################################################
// IO for ITensor tensors

// precision of double variable
typedef std::numeric_limits< double > DBL;

// Write a tensor into file in text (human-readable) format
void writeTensorF(std::string fname, itensor::ITensor t);

// Write tensor in binary form into file
void writeTensorB(std::string const& fname, itensor::ITensor const& t);

// Read a tensor from file in text (human-readable) format
itensor::ITensor readTensorF(std::string const& fname);

// Read tensor from file holding its binary form
itensor::ITensor readTensorB(std::string const& fname);

// ############################################################################
// IO toString methods

/* 
 * Defining these methods enables printing of 
 *     Cluster, CtmData 
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
operator<<(std::ostream& s, Cluster const& c);

std::ostream& 
operator<<(std::ostream& s, CtmData const& d);

#endif
