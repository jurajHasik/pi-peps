//The following ifndef/define/endif pattern is called a 
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __CTM_CLS_H_
#define __CTM_CLS_H_

#include <iostream>
#include <vector>
#include <map>
#include "json.hpp"
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
 * Generic displacement vector on a square lattice.
 * Support +,-,-=,+=,==,!= 
 */ 
struct Shift {
    std::array<int, 2> d;

    Shift(int dx, int dy) : d({dx,dy}) {}

    bool operator== (const Shift &s) const {
        return (this->d[0] == s.d[0]) && (this->d[1] == s.d[1]);
    }
 
    bool operator!= (const Shift &s) const {
        return (this->d[0] != s.d[0]) || (this->d[1] != s.d[1]);
    }

    Shift operator+(Shift const& s) const {
        return Shift(this->d[0] + s.d[0], this->d[1] + s.d[1]);
    }

    Shift operator-(Shift const& s) const {
        return Shift(this->d[0] - s.d[0], this->d[1] - s.d[1]);
    }

    Shift& operator+=(Shift const& s) {
        this->d[0] += s.d[0]; this->d[1] += s.d[1];
        return *this;
    }

    Shift& operator-=(Shift const& s) {
        this->d[0] -= s.d[0]; this->d[1] -= s.d[1];
        return *this;
    }
};

/*
 * A vertex of a square lattice. One can obtain new vertices
 * by applying displacement: Vertex <- Vertex [+,-,+=,-=] Shift 
 */ 
struct Vertex {
    std::array<int, 2> r;

    Vertex(int x, int y) : r({x,y}) {}

    bool operator== (const Vertex &v) const {
        return (this->r[0] == v.r[0]) && (this->r[1] == v.r[1]);
    }
 
    bool operator!= (const Vertex &v) const {
        return (this->r[0] != v.r[0]) || (this->r[1] != v.r[1]);
    }

    bool operator> (const Vertex &v) const { return this->r > v.r; }

    bool operator< (const Vertex &v) const { return this->r < v.r; }

    Vertex operator+(Shift const& s) const {
        return Vertex(this->r[0] + s.d[0], this->r[1] + s.d[1]);
    }

    Vertex operator-(Shift const& s) const {
        return Vertex(this->r[0] - s.d[0], this->r[1] - s.d[1]);
    }

    Vertex& operator+=(Shift const& s) {
        this->r[0] += s.d[0]; this->r[1] += s.d[1];
        return *this;
    }

    Vertex& operator-=(Shift const& s) {
        this->r[0] -= s.d[0]; this->r[1] -= s.d[1];
        return *this;
    }
};

struct LinkWeight {
    std::vector< std::string > sId; // Ids of sites connected by weight
    std::vector<int> dirs;          // auxlinks of sites connected by weight
    std::string wId;                // weight Id
};

/*
 * Struct holding the supercell data. Non-equivalent tensors, 
 * optional weights on links and physical and bond dimensions
 *
 */
struct Cluster {
    // meta information about the origin of cluster
    std::string metaInfo;
    nlohmann::json simParam;

    // size of n=Y(=rows) x m=X(=cols) cluster
    int sizeN, sizeM;
    int lX, lY;

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
    std::map< Vertex, std::string > vToId;

    // each link between two sites might hold a matrix of weights
    // each site identified by siteId holds information about all
    // four links attached
    std::map< std::string, std::vector< LinkWeight > > siteToWeights;

    std::map< std::string, itensor::ITensor > weights;
    // storing previous weights to compute spectral distance of current
    // weights wrt to old_weights
    std::map< std::string, itensor::ITensor > old_weights;

    // Implements Boundary condition of cluster by derived class
    std::string virtual vertexToId(Vertex const& v) const { 
        return vToId.at(v); 
    }

    itensor::ITensor const& getSiteRefc(Vertex const& v) const {
        return sites.at(vertexToId(v));
    }

    itensor::ITensor & getSiteRef(Vertex const& v) {
        return sites.at(vertexToId(v));
    }

    // using copy constructor of ITensor
    itensor::ITensor getSite(Vertex const& v) const {
        itensor::ITensor t = sites.at(vertexToId(v));
        return t;
    }
};

void initClusterSites(Cluster & c, bool dbg = false);

void initClusterWeights(Cluster & c, bool dbg = false);

void setWeights(Cluster & c, std::string option, bool dbg = false);

void saveWeights(Cluster & c, bool dbg = false);

double weightDist(Cluster const& c);

void setSites(Cluster & c, std::string option, bool dbg = false);

itensor::ITensor contractCluster(Cluster const& c, bool dbg = false);

itensor::ITensor clusterDenMat(Cluster const& c, bool dbg = false);

void absorbWeightsToSites(Cluster & cls, bool dbg = false);

void mvSite(Cluster const& c, std::pair<int,int> &s, int dir);

void mvSite(Cluster const& c, std::pair<int,int> &s, std::pair<int,int> const& disp);

std::pair<int,int> getNextSite(Cluster const& c, std::pair<int,int> const& s, int dir);


std::ostream& 
operator<<(std::ostream& os, Shift const& s);

std::ostream& 
operator<<(std::ostream& os, Vertex const& v);

std::ostream& 
operator<<(std::ostream& s, Cluster const& c);

std::ostream& 
operator<<(std::ostream& s, LinkWeight const& lw);

#endif