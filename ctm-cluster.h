//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __CTM_CLS_H_
#define __CTM_CLS_H_

#include <iostream>
#include <vector>
#include <map>
#include "itensor/all.h"

// ############################################################################
// IO for cluster definition using JSON data format

// SITE auxiliary indices 
const std::string TAG_I_AUX  = "I_AUX";
const std::string TAG_I_PHYS = "I_PHYS";

const char* const TAG_IT_AUXLINK  = "AUXlink";
const char* const TAG_IT_PHYSSITE = "Psite";

// types for auxiliary indices of on-Site tensors
const auto AUXLINK = itensor::IndexType(TAG_IT_AUXLINK);
const auto PHYS    = itensor::IndexType(TAG_IT_PHYSSITE);

/*
 * Struct holding the raw environment
 * 
 */
struct Cluster {
    // meta information about the origin of cluster
    std::string metaInfo;

    // size of n=Y(=rows) x m=X(=cols) cluster
    int sizeN, sizeM;
    
    // auxiliary bond dimension
    int auxBondDim;
    // dimension of local Hilbert space = dimension of physical index
    int physDim;

    // siteIds
    std::vector< std::string > siteIds;

    // inequivalent sites
    std::map< std::string, itensor::ITensor > sites;

    // map from cluster sites to inequivalent sites
    std::map< std::pair< int,int >, std::string > cToS;
};

itensor::ITensor contractCluster(Cluster const& c, bool dbg = false);

std::ostream& 
operator<<(std::ostream& s, Cluster const& c);

#endif