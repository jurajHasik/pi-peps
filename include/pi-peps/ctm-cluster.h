// The following ifndef/define/endif pattern is called a
// scope guard, and prevents the C++ compiler (actually, preprocessor)
// from including a header file more than once.
#ifndef __CTM_CLS_H_
#define __CTM_CLS_H_

#include "pi-peps/config.h"
#include "json.hpp"
#include "pi-peps/lattice.h"
#include <iostream>
#include <map>
#include <vector>
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS

// ############################################################################
// IO for cluster definition using JSON data format

// SITE auxiliary indices
const std::string TAG_I_AUX = "I_AUX";
const std::string TAG_I_PHYS = "I_PHYS";

const char* const TAG_IT_AUXLINK = "AUXlink";
const char* const TAG_IT_PHYSSITE = "Psite";

// types for auxiliary indices of on-Site tensors
const auto AUXLINK = itensor::IndexType(TAG_IT_AUXLINK);
const auto PHYS = itensor::IndexType(TAG_IT_PHYSSITE);

struct LinkWeight {
  std::vector<std::string> sId;  // Ids of sites connected by weight
  std::vector<int> dirs;         // auxlinks of sites connected by weight
  std::string wId;               // weight Id
};

std::ostream& operator<<(std::ostream& s, LinkWeight const& lw);

/*
 * Struct holding the supercell data. Non-equivalent tensors,
 * optional weights on links and physical and bond dimensions
 *
 */
struct Cluster {
  static const int BRAKET_OFFSET;

  std::string cluster_type;

  // meta information about the origin of cluster
  std::string metaInfo;
  nlohmann::json simParam;

  // size of n=Y(=rows) x m=X(=cols) cluster
  int sizeN, sizeM;
  int lX, lY;

  // siteIds
  std::vector<std::string> siteIds;
  std::map<std::string, int> SI;

  // mapt to physical indices of on-site tensors
  std::map<std::string, itensor::Index> mphys;

  // map to individual aux indices of on-site tensors
  // A -> { index in dir 0, ..., index in dir 3 }
  //         dir 1
  //          |
  // dir 0 -- A -- dir 2
  //          |
  //         dir 3
  std::map<std::string, std::vector<itensor::Index>> caux;

  itensor::Index const& AIc(Vertex const& v, int dir) const {
    return AIc(vertexToId(v), dir);
  }

  itensor::Index const& AIc(std::string const& id, int dir) const {
    return caux.at(id)[dir];
  }

  std::vector<itensor::Index> AIBraKetPair(Vertex const& v, int dir) const {
    return AIBraKetPair(vertexToId(v), dir);
  }

  std::vector<itensor::Index> AIBraKetPair(std::string const& id,
                                           int dir) const {
    return {AIc(id, dir), prime(AIc(id, dir), BRAKET_OFFSET)};
  }

  itensor::ITensor DContract(Vertex const& v0,
                             int dir0,
                             Vertex const& v1,
                             int dir1) const {
    return itensor::delta(AIc(v0, dir0), AIc(v1, dir1));
  }

  itensor::ITensor DContract(std::string const& id0,
                             int dir0,
                             std::string const& id1,
                             int dir1) const {
    return itensor::delta(AIc(id0, dir0), AIc(id1, dir1));
  }

  // inequivalent sites
  std::map<std::string, itensor::ITensor> sites;

  // map from cluster sites to inequivalent sites
  std::map<std::pair<int, int>, std::string> cToS;
  std::map<Vertex, std::string> vToId;
  std::map<std::string, Vertex> idToV;

  // each link between two sites might hold a matrix of weights
  // each site identified by siteId holds information about all
  // four links attached
  std::map<std::string, std::vector<LinkWeight>> siteToWeights;

  std::map<std::string, itensor::ITensor> weights;
  // storing previous weights to compute spectral distance of current
  // weights wrt to old_weights
  std::map<std::string, itensor::ITensor> old_weights;

  bool weights_absorbed = false;

  Cluster() {}

  Cluster(int lX_, int lY_)
    : cluster_type("DEFAULT"), sizeN(lY_), sizeM(lX_), lX(lX_), lY(lY_) {}

  static std::unique_ptr<Cluster> create(nlohmann::json const& json_cluster);

  // Implements Boundary condition of cluster by derived class
  // default assumes simple PBC
  std::string virtual vertexToId(Vertex const& v) const {
    auto elemV = Vertex((v.r[0] + std::abs(v.r[0]) * lX) % lX,
                        (v.r[1] + std::abs(v.r[1]) * lY) % lY);
    return vToId.at(elemV);
  }

  itensor::ITensor const& getSiteRefc(Vertex const& v) const {
    return sites.at(vertexToId(v));
  }

  itensor::ITensor& getSiteRef(Vertex const& v) {
    return sites.at(vertexToId(v));
  }

  // using copy constructor of ITensor
  itensor::ITensor getSite(Vertex const& v) const {
    itensor::ITensor t = sites.at(vertexToId(v));
    return t;
  }

  void absorbWeightsToSites(bool dbg = false);

  void absorbWeightsToLinks(bool dbg = false);

  /** make sure the right dtor is called */
  virtual ~Cluster() = default;
};

void initClusterWeights(Cluster& c, bool dbg = false);

void setWeights(Cluster& c, std::string option, bool dbg = false);

void saveWeights(Cluster& c, bool dbg = false);

double weightDist(Cluster const& c);

std::ostream& operator<<(std::ostream& s, Cluster const& c);

#endif
