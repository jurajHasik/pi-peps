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

struct LinkWeight {
    std::vector< std::string > sId; // Ids of sites connected by weight
    std::vector<int> dirs;          // auxlinks of sites connected by weight
    std::string wId;                // weight Id
};

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
    std::map< std::string, int> SI;

    // aux indices and phys indicies of sites siteIds[0] <-> aux[0],phys[0] 
    std::vector< itensor::Index > aux, phys;

    // inequivalent sites
    std::map< std::string, itensor::ITensor > sites;

    // map from cluster sites to inequivalent sites
    std::map< std::pair< int,int >, std::string > cToS;

    // each link between two sites might hold a matrix of weights
    // each site identified by siteId holds information about all
    // four links attached
    std::map< std::string, std::vector< LinkWeight > > siteToWeights;

    std::map< std::string, itensor::ITensor > weights;
};

void initClusterSites(Cluster & c, bool dbg = false);

void initClusterWeights(Cluster & c, bool dbg = false);

void setWeights(Cluster & c, std::string option, bool dbg = false);

void setSites(Cluster & c, std::string option, bool dbg = false);

itensor::ITensor contractCluster(Cluster const& c, bool dbg = false);

itensor::ITensor clusterDenMat(Cluster const& c, bool dbg = false);

void absorbWeightsToSites(Cluster & cls, bool dbg = false);

std::ostream& 
operator<<(std::ostream& s, Cluster const& c);

std::ostream& 
operator<<(std::ostream& s, LinkWeight const& lw);

#endif