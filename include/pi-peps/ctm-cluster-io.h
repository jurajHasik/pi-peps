#ifndef __CTM_CLS_IO_
#define __CTM_CLS_IO_

#include "pi-peps/config.h"
#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include "json.hpp"
#include "pi-peps/ctm-cluster.h"
#include "pi-peps/ctm-cluster-basic.h"
#include "pi-peps/ctm-cluster-global.h"
#include "pi-peps/cluster-factory.h"
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS

const std::string WS4(4,' ');

typedef enum IO_ENV_FMT {
    IO_ENV_FMT_txt,
    IO_ENV_FMT_bin
} io_env_fmt_type;

// string to enum conversion
IO_ENV_FMT toIO_ENV_FMT(std::string const& ioFmt);

struct IO_site {
    itensor::Index phys;
    std::vector<itensor::Index> aux;
    itensor::ITensor t;
};

Cluster readCluster(std::string const& filename);

Cluster readCluster(nlohmann::json const& jsonCls);

std::unique_ptr<Cluster> p_readCluster(std::string const& filename);

std::unique_ptr<Cluster> p_readCluster(nlohmann::json const& jsonCls);

void readClusterWeights(Cluster & cls, nlohmann::json const& jsonCls);

void writeCluster(std::string const& filename, Cluster const& cls);

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
 *               dir 0  2   1  3  
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

//        dir index 
std::pair<int,itensor::Index> readAuxIndex(nlohmann::json const& j);

// itensor::ITensor readOnSiteT(nlohmann::json const& j, int offset = 1);

itensor::ITensor readTfromJSON(nlohmann::json const& j, int offset = 1);

std::pair< std::vector<itensor::Index>, itensor::ITensor> readIndsAndTfromJSON(
    nlohmann::json const& j, int offset = 1);

void readOnSiteFromJSON(Cluster & c, nlohmann::json const& j, bool dbg = false);

void setOnSiteTensorsFromFile(Cluster & cls, std::string const& filename, bool dbg = false);

void setOnSiteTensorsFromJSON(Cluster & cls, nlohmann::json const& j, bool dbg = false);

/* 
 * write elements of an on-site tensor T on a SQUARE lattice into a vector
 * in the format as described in readOnSiteT. (Optional) Include elements
 * of abs value > threshold
 */
void writeOnSiteTElems(std::vector< std::string > & tEs,
    Cluster const& cls, std::string id, int offset = 1, double threshold = 1.0e-10);

// ############################################################################
// IO for environment of nxm cluster

/*
 * Struct holding the raw environment
 *
 * TODO possibly redundant entries for size of the cluster
 * 
 */
// struct CtmData {
//     int auxDimEnv;
//     int auxDimSite;

//     int sizeN, sizeM; // size of n(row) x m(columns) cluster

//     std::vector< itensor::ITensor > sites;
//     // map from cluster to sites
//     std::map< std::pair<int,int>, int> cToS;

//     // arrays holding half-row/column tensors
//     std::vector< itensor::ITensor > T_U;
//     std::vector< itensor::ITensor > T_R;
//     std::vector< itensor::ITensor > T_D;
//     std::vector< itensor::ITensor > T_L;

//     // corner tensors
//     itensor::ITensor C_LU, C_RU, C_RD, C_LD;
   
//     itensor::Index I_U, I_R, I_D, I_L;
//     itensor::Index I_XH, I_XV;
// };

struct CtmData_Full {
    int auxDimEnv;
    int auxDimSite;

    int sizeN, sizeM; // size of n(row) x m(columns) cluster

    std::vector< itensor::ITensor > sites;
    // map from cluster to sites
    std::map< std::pair<int,int>, int> cToS;

    // arrays holding half-row/column tensors
    std::vector< itensor::ITensor > T_U;
    std::vector< itensor::ITensor > T_R;
    std::vector< itensor::ITensor > T_D;
    std::vector< itensor::ITensor > T_L;

    // corner tensors
    std::vector< itensor::ITensor > C_LU;
    std::vector< itensor::ITensor > C_RU;
    std::vector< itensor::ITensor > C_RD;
    std::vector< itensor::ITensor > C_LD;
   
    itensor::Index I_U, I_R, I_D, I_L;
    itensor::Index I_XH, I_XV;
};

/*
 * store/read environment to/from file(s)
 *
 */
// void writeEnv(IO_ENV_FMT ioFmt, std::string TAG, CtmData const& ctmD);

// /*
//  * store/read environment to/from file(s)
//  *
//  */
// CtmData readEnv(IO_ENV_FMT ioFmt, std::string const& TAG, Cluster const& cls);


//  * Read full environment of a cluster composed from simple environments
//  * of non-equivalent sites
 
// std::vector<CtmData> readEnv_V2(IO_ENV_FMT ioFmt, std::string const& TAG, 
//     Cluster const& cls);

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
 *     CtmData 
 * objects using 
 *     cout << o ;
 * where o is an instance of one of mentioned objects
 *
 * It also allows printing using the print,println,printf, and
 * printfln functions. Use the "%s" flag to print custom objects
 * with printf and printfln.
 *
 */
// std::ostream& 
// operator<<(std::ostream& s, CtmData const& d);

// std::ostream& 
// operator<<(std::ostream& s, CtmData_Full const& d);

#endif
