#include "pi-peps/config.h"
#include "pi-peps/ctm-cluster.h"

using namespace itensor;

const int Cluster::BRAKET_OFFSET = 4;

std::unique_ptr<Cluster> Cluster::create(nlohmann::json const& json_cluster) {
  int lX = json_cluster.value("lX", -1);
  if (lX < 0) {
    std::cout << "[Cluster::create] key lX not found."
              << "Using legacy key sizeM" << std::endl;
    lX = json_cluster["sizeM"].get<int>();
  }
  int lY = json_cluster.value("lY", -1);
  if (lY < 0) {
    std::cout << "[Cluster::create] key lY not found."
              << "Using legacy key sizeN" << std::endl;
    lY = json_cluster["sizeN"].get<int>();
  }

  return std::unique_ptr<Cluster>(new Cluster(lX, lY));
}

void initClusterWeights(Cluster& c, bool dbg) {
  if (c.siteToWeights.size() == 0) {
    std::cout << "[initClusterWeights]"
              << " no weights stored for this cluster" << std::endl;
    throw std::runtime_error("Invalid input");
  }

  // reset
  c.weights.clear();

  // create map holding LinkWeights
  std::map<std::string, LinkWeight> tmpLWs;

  for (const auto& lwSet :
       c.siteToWeights)                  // map< string, vector<LinkWeight> >
    for (const auto& lw : lwSet.second)  // loop over linkWeights
      if ((tmpLWs.find(lw.wId) == tmpLWs.end()))
        tmpLWs[lw.wId] = lw;

  for (const auto& lw : tmpLWs)
    c.weights[lw.second.wId] =
      ITensor(c.AIc(lw.second.sId[0], lw.second.dirs[0]),
              c.AIc(lw.second.sId[1], lw.second.dirs[1]));
}

void setWeights(Cluster& c, std::string option, bool dbg) {
  if (c.siteToWeights.size() == 0) {
    std::cout << "[setWeights]"
              << " no weights stored for this cluster" << std::endl;
    throw std::runtime_error("Invalid input");
  }

  if (option == "DELTA") {
    for (auto& wEntry : c.weights)  // map < string, tensor >
    {
      std::vector<double> tmpD(wEntry.second.inds()[0].m(), 1.0);
      wEntry.second = diagTensor(tmpD, wEntry.second.inds());
    }
  } else {
    std::cout << "[setWeights] ctm-cluster setWeights unsupported option: "
              << option << std::endl;
    throw std::runtime_error("Invalid option");
  }
}

void saveWeights(Cluster& c, bool dbg) {
  c.old_weights = c.weights;
}

double weightDist(Cluster const& c) {
  double res = 0.0;
  for (auto const& e : c.weights) {
    res += norm(c.weights.at(e.first) - c.old_weights.at(e.first));
  }
  return res;
}

void Cluster::absorbWeightsToSites(bool dbg) {
  if (not weights_absorbed) {
    auto sqrtT = [](double r) { return std::sqrt(r); };

    for (auto& siteEntry : sites) {
      auto sId = siteEntry.first;
      // contract each on-site tensor with its weights
      // and set back the original index
      for (auto const& stw : siteToWeights.at(sId)) {
        ITensor tmp_weight = weights.at(stw.wId);
        tmp_weight.apply(sqrtT);
        siteEntry.second *= tmp_weight;
        siteEntry.second *= delta(tmp_weight.inds());
      }
    }
    weights_absorbed = true;
  } else {
    std::cout << "[absorbWeightsToSites] Weights already absorbed" << std::endl;
  }
}

void Cluster::absorbWeightsToLinks(bool dbg) {
  if (weights_absorbed) {
    double machine_eps = std::numeric_limits<double>::epsilon();

    for (auto& siteEntry : sites) {
      auto sId = siteEntry.first;
      // contract each on-site tensor with its weights
      // and set back the original index
      for (auto const& stw : siteToWeights.at(sId)) {
        ITensor tmp_weight = weights.at(stw.wId);

        auto ind1 = tmp_weight.inds()[0];
        auto ind2 = tmp_weight.inds()[1];
        std::vector<double> tmpD;
        double const tol =
          std::sqrt(tmp_weight.real(ind1(1), ind2(1)) *
                    std::max(ind1.m(), ind2.m()) * machine_eps);

        for (int i = 1; i <= ind1.m(); i++) {
          double elem = tmp_weight.real(ind1(i), ind2(i));
          if (std::abs(elem) > tol) {
            tmpD.push_back(1.0 / std::sqrt(elem));
          } else
            tmpD.push_back(0.0);
        }
        tmp_weight = diagTensor(tmpD, ind1, ind2);

        siteEntry.second *= tmp_weight;
        siteEntry.second *= delta(tmp_weight.inds());
      }
    }
    weights_absorbed = false;
  } else {
    std::cout << "[absorbWeightsToLinks] Weights are not absorbed to sites"
              << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, Shift const& s) {
  return os << "Shift(" << s.d[0] << " " << s.d[1] << ")";
}

std::ostream& operator<<(std::ostream& os, Vertex const& v) {
  return os << "Vertex(" << v.r[0] << " " << v.r[1] << ")";
}

std::ostream& operator<<(std::ostream& s, Cluster const& c) {
  s << "Cluster( metaInfo: " << c.metaInfo << " type: " << c.cluster_type
    << ", sizeM: " << c.sizeM << ", sizeN: " << c.sizeN << " | lX: " << c.lX
    << " , lY: " << c.lY << std::endl;

  s << "siteIds: [ ";
  for (const auto& siteId : c.siteIds) {
    s << siteId << " ";
  }
  s << "]" << std::endl;

  s << "SI: [" << std::endl;
  for (const auto& idToPos : c.SI) {
    s << idToPos.first << " --> " << idToPos.second << " --> "
      << c.sites.at(idToPos.first) << std::endl;
  }
  s << "]" << std::endl;

  s << "PHYS indices: [" << std::endl;
  for (auto const& e : c.mphys)
    s << e.first << " : " << e.second << std::endl;
  s << "]" << std::endl;

  s << "AUX indices: [" << std::endl;
  for (auto const& e : c.caux) {
    s << e.first << " : ";
    for (auto const& i : e.second)
      s << i << " ";
    s << std::endl;
  }
  s << "]" << std::endl;

  s << "clusterToSite: [" << std::endl;
  for (const auto& cToSEntry : c.cToS) {
    s << "(" << cToSEntry.first.first << ", " << cToSEntry.first.second
      << ") -> " << cToSEntry.second << " -> " << c.SI.at(cToSEntry.second)
      << std::endl;
  }
  s << "]" << std::endl;

  s << "VertexToSite: [" << std::endl;
  for (const auto& vId : c.vToId) {
    s << vId.first << " --> " << vId.second << " --> " << c.getSite(vId.first)
      << std::endl;
  }
  s << "]" << std::endl;

  s << "siteToWeights: [" << std::endl;
  for (const auto& lwEntrySet : c.siteToWeights) {
    s << lwEntrySet.first << " --> [" << std::endl;
    for (const auto& lwEntry : lwEntrySet.second)
      s << lwEntry << std::endl;
    s << "]" << std::endl;
  }
  s << std::endl;

  s << "weights: [" << std::endl;
  for (const auto& wEntry : c.weights) {
    s << wEntry.first << " --> ";
    printfln("%f", wEntry.second);
  }
  s << "]" << std::endl;

  return s;
}

std::ostream& operator<<(std::ostream& s, LinkWeight const& lw) {
  s << "LinkWeight( " << lw.wId << " [" << lw.sId[0] << "," << lw.sId[1]
    << "], [" << lw.dirs[0] << "," << lw.dirs[1] << "])";

  return s;
}
