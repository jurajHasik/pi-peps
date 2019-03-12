#include "pi-peps/config.h"
#include "pi-peps/ctm-cluster-io.h"

using namespace std;

IO_ENV_FMT toIO_ENV_FMT(string const& ioFmt) {
  if (ioFmt == "IO_ENV_FMT_txt")
    return IO_ENV_FMT_txt;
  if (ioFmt == "IO_ENV_FMT_bin")
    return IO_ENV_FMT_bin;
  cout << "Unsupported IO_ENV_FMT" << std::endl;
  exit(EXIT_FAILURE);
  // return -1;
}

// ############################################################################
// IO for cluster definition using JSON data format

std::unique_ptr<Cluster> p_readCluster(string const& filename) {
  ifstream infile;
  infile.open(filename, ios::in);

  nlohmann::json jsonCls;
  infile >> jsonCls;
  jsonCls["initBy"] = "FILE";

  return p_readCluster(jsonCls);
}

// TODO build register-enable factory
std::unique_ptr<Cluster> p_readCluster(nlohmann::json const& jsonCls) {
  ClusterFactory cf = ClusterFactory();
  auto p_cls = cf.create(jsonCls);

  for (const auto& mapEntry : jsonCls["map"].get<vector<nlohmann::json>>()) {
    p_cls->cToS[make_pair(mapEntry["x"].get<int>(), mapEntry["y"].get<int>())] =
      mapEntry["siteId"].get<string>();
    p_cls->vToId[Vertex(mapEntry["x"].get<int>(), mapEntry["y"].get<int>())] =
      mapEntry["siteId"].get<string>();
    p_cls->idToV[mapEntry["siteId"].get<string>()] =
      Vertex(mapEntry["x"].get<int>(), mapEntry["y"].get<int>());
  }

  for (const auto& siteIdEntry : jsonCls["siteIds"].get<vector<string>>()) {
    p_cls->siteIds.push_back(siteIdEntry);
    p_cls->SI[siteIdEntry] = p_cls->siteIds.size() - 1;
  }

  for (const auto& siteEntry : jsonCls["sites"]) {
    auto id = siteEntry["siteId"].get<string>();
    auto tmp = readIndsAndTfromJSON(siteEntry);

    p_cls->mphys[id] = tmp.first[0];
    p_cls->caux[id] = std::vector<itensor::Index>(tmp.first.size() - 1);
    for (int i = 1; i < tmp.first.size(); i++)
      p_cls->caux[id][i - 1] = tmp.first[i];
    // std::copy( tmp.first.begin()+1, tmp.first.end(), c.caux[id] );
    p_cls->sites[id] = tmp.second;  // tensor
  }

  // construction of weights on links within c
  if (jsonCls.value("linkWeightsUsed", false)) {
    readClusterWeights(*p_cls, jsonCls);  // reads the link-weights data
    initClusterWeights(*p_cls);           // creates the link-weight tensors
  }

  return p_cls;
}

void readClusterWeights(Cluster& cls, nlohmann::json const& jsonCls) {
  for (const auto& siteIdEntry : jsonCls["siteIds"].get<vector<string>>())
    cls.siteToWeights[siteIdEntry] = vector<LinkWeight>();

  for (const auto& lwEntry :
       jsonCls["linkWeights"].get<vector<nlohmann::json>>()) {
    LinkWeight lw = {lwEntry["sites"].get<vector<string>>(),
                     lwEntry["directions"].get<vector<int>>(),
                     lwEntry["weightId"].get<string>()};
    LinkWeight rlw = lw;
    reverse(rlw.sId.begin(), rlw.sId.end());
    reverse(rlw.dirs.begin(), rlw.dirs.end());

    cls.siteToWeights[lw.sId[0]].push_back(lw);
    cls.siteToWeights[rlw.sId[0]].push_back(rlw);
  }
}

/*
 * TODO object Cluster contains map with actual ITensor objects
 *      which are not suitable for json representations
 *
 */
void writeCluster(string const& filename, Cluster const& cls) {
  ofstream outf;
  outf.open(filename, ios::out);

  nlohmann::json jCls;
  jCls["type"] = cls.cluster_type;
  jCls["meta"] = cls.metaInfo;
  jCls["simParam"] = cls.simParam;
  jCls["lX"] = cls.lX;
  jCls["lY"] = cls.lY;

  vector<nlohmann::json> jcToS;
  for (auto const& entry : cls.cToS) {
    nlohmann::json jentry;
    jentry["x"] = entry.first.first;
    jentry["y"] = entry.first.second;
    jentry["siteId"] = entry.second;
    jcToS.push_back(jentry);
  }
  jCls["map"] = jcToS;

  jCls["siteIds"] = cls.siteIds;

  if (cls.siteToWeights.size() > 0) {
    jCls["linkWeightsUsed"] = true;

    vector<nlohmann::json> jlws;
    vector<std::string> lwIds;
    for (auto const& stw : cls.siteToWeights)
      for (auto const& lw : stw.second)
        if (std::find(lwIds.begin(), lwIds.end(), lw.wId) == lwIds.end()) {
          nlohmann::json jentry;
          jentry["sites"] = lw.sId;
          jentry["directions"] = lw.dirs;
          jentry["weightId"] = lw.wId;

          jlws.push_back(jentry);
          lwIds.push_back(lw.wId);
        }

    jCls["linkWeights"] = jlws;
  }

  vector<nlohmann::json> jsites;
  for (auto const& entry : cls.sites) {
    auto siteId = entry.first;

    nlohmann::json jentry;
    jentry["siteId"] = siteId;
    jentry["physDim"] = cls.mphys.at(siteId).m();
    vector<nlohmann::json> auxInds(cls.caux.at(siteId).size());
    for (int i = 0; i < cls.caux.at(siteId).size(); i++) {
      auto ind = cls.caux.at(siteId)[i];
      auxInds[i] = {{"dir", i}, {"ad", ind.m()}, {"name", ind.rawname()}};
    }
    jentry["auxInds"] = auxInds;
    vector<string> tensorElems;
    writeOnSiteTElems(tensorElems, cls, siteId);
    jentry["numEntries"] = tensorElems.size();
    jentry["entries"] = tensorElems;
    jsites.push_back(jentry);
  }
  jCls["sites"] = jsites;

  outf << jCls.dump(4) << endl;
}

/*
 * TODO? implement named indices in input file processing
 * TODO? expose indices of returned tensor
 * TODO? check auxBondDim vs auxDim per site consistency
 *
 */
pair<int, itensor::Index> readAuxIndex(nlohmann::json const& j) {
  auto dir = j["dir"].get<int>();
  auto auxDim = j["ad"].get<int>();
  auto name = j["name"].get<string>();
  return make_pair(dir, itensor::Index(name, auxDim, AUXLINK, dir));
}

itensor::ITensor readTfromJSON(nlohmann::json const& j, int offset) {
  auto result = readIndsAndTfromJSON(j, offset);
  return result.second;
}

// returns pair < indices, tensor >
// IndexSet of ITensor, where 0th index is the physical one. Remaining
// indices are auxiliary
pair<vector<itensor::Index>, itensor::ITensor> readIndsAndTfromJSON(
  nlohmann::json const& j,
  int offset) {
  auto id = j["siteId"].get<string>();

  vector<itensor::Index> ti(5, itensor::Index());
  ti[0] = itensor::Index(id + "-" + TAG_I_PHYS, j["physDim"].get<int>(), PHYS);

  // check if tensor has custom set of auxiliary indices provided
  auto p_json_inds_array = j.find("auxInds");
  if (p_json_inds_array == j.end()) {
    std::cout << "[readOnSiteT] " << id << " : auxInds array not found. "
              << " Assuming identical auxiliary indices on all bonds."
              << std::endl;

    ti[1] =
      itensor::Index(id + "-" + TAG_I_AUX, j["auxDim"].get<int>(), AUXLINK);
    for (int i = 2; i < ti.size(); i++)
      ti[i] = prime(ti[1], i - 1);
  } else {
    if (j["auxInds"].size() != 4) {
      std::cout << "[readOnSiteT] " << id << " : auxInds has to contain"
                << " four auxiliary indices" << std::endl;
      throw std::runtime_error("Invalid input");
    }

    for (const auto& i : j["auxInds"]) {
      auto tmp = readAuxIndex(i);
      ti[1 + tmp.first] = tmp.second;
    }

    // verify, that indices for each direction have been read
    for (int i = 1; i < ti.size(); i++) {
      if (not ti[i]) {
        std::cout << "[readOnSiteT] " << id << " : auxInds does not contain "
                  << " auxiliary index for direction " << i - 1 << std::endl;
        throw std::runtime_error("Invalid input");
      }
    }
  }

  auto t = itensor::ITensor(ti);

  string token[7];
  int pI, aI0, aI1, aI2, aI3;
  char delim = ' ';
  for (const auto& tEntry : j["entries"].get<vector<string>>()) {
    istringstream iss(tEntry);

    token[6] = "0.0";

    for (int i = 0; i < 7; i++) {
      getline(iss, token[i], delim);
    }

    // ITensor indices start from 1, hence if input file indices start from
    // 0 use offset 1
    pI = offset + stoi(token[0]);
    aI0 = offset + stoi(token[1]);
    aI1 = offset + stoi(token[2]);
    aI2 = offset + stoi(token[3]);
    aI3 = offset + stoi(token[4]);

    t.set(ti[0](pI), ti[1](aI0), ti[2](aI1), ti[3](aI2), ti[4](aI3),
          complex<double>(stod(token[5]), stod(token[6])));
  }

  return make_pair(ti, t);
}

void setOnSiteTensorsFromFile(Cluster& c, string const& filename, bool dbg) {
  ifstream infile;
  infile.open(filename, ios::in);

  nlohmann::json jsonCls;
  infile >> jsonCls;

  // TODO - Warnings
  // if (jsonCls["physDim"] != c.physDim) {}
  // if (jsonCls["auxBondDim"] > c.auxBondDim) {}

  setOnSiteTensorsFromJSON(c, jsonCls, dbg);
}

void setOnSiteTensorsFromJSON(Cluster& c, nlohmann::json const& j, bool dbg) {
  for (const auto& siteEntry : j["sites"].get<vector<nlohmann::json>>()) {
    pair<vector<itensor::Index>, itensor::ITensor> tmp = readIndsAndTfromJSON(siteEntry);

    auto id = siteEntry["siteId"].get<string>();
    c.mphys[id] = tmp.first[0];
    c.caux[id] = std::vector<itensor::Index>(tmp.first.size() - 1);
    for (int i = 1; i < tmp.first.size(); i++)
      c.caux[id][i - 1] = tmp.first[i];
    c.sites[id] = tmp.second;  // tensor
  }
}

void writeOnSiteTElems(vector<string>& tEs,
                       Cluster const& c,
                       std::string id,
                       int offset,
                       double threshold) {
  auto pI = c.mphys.at(id);
  auto aI = c.caux.at(id);

  string t_entry_str;
  // ostringstream ost;
  // ost.precision( numeric_limits< double >::max_digits10 );
  for (int p = 0; p < pI.m(); p++) {
    for (int a0 = 0; a0 < aI[0].m(); a0++) {
      for (int a1 = 0; a1 < aI[1].m(); a1++) {
        for (int a2 = 0; a2 < aI[2].m(); a2++) {
          for (int a3 = 0; a3 < aI[3].m(); a3++) {
            complex<double> elem = c.sites.at(id).cplx(
              pI(p + offset), aI[0](a0 + offset), aI[1](a1 + offset),
              aI[2](a2 + offset), aI[3](a3 + offset));
            if (abs(elem) >= threshold) {
              t_entry_str = to_string(p) + " " + to_string(a0) + " " +
                            to_string(a1) + " " + to_string(a2) + " " +
                            to_string(a3) + " ";
              ostringstream ost;
              ost.precision(numeric_limits<double>::max_digits10);
              ost << elem.real() << " " << elem.imag();
              t_entry_str += ost.str();
              // ost.clear();
              // ost.seekp(0);
              tEs.push_back(t_entry_str);
            }
          }
        }
      }
    }
  }
}

/*
 * Write out tensor in given (human-readable) format to output file
 * [using itensor::PrintData(t) format]
 *
 * TODO implement custom print fuction for tensor instead of
 *      redirecting stdout to file using C function
 *
 */
void writeTensorF(string fname, itensor::ITensor t) {
  ofstream outf;
  outf.open(fname, ios::out);

  // Print full information about tensor in following format
  /*
   * ITensor r=t.r() t.inds()
   * (*,*,...) Re + Im
   * ...
   *
   * where t.r() is the rank of the tensor, t.inds() is the IndexSet
   * holding indices of tensor and each consecutive line holds
   * single non-zero tensor element with values of indices in the
   * same order as in t.inds() IndexSet
   *
   */

  // double threshold = 1.0e-14;
  // Apply lambda expression to each tensor element
  // t.apply([&threshold](itensor::Cplx val) {
  //    if(abs(val) <= threshold) val = 0.0;
  //    return val;
  //});

  // Save old settings
  ios::fmtflags old_settings = cout.flags();
  int old_precision = cout.precision();

  cout.precision(numeric_limits<double>::max_digits10);

  streambuf* coutbuf = cout.rdbuf();  // save old buf
  cout.rdbuf(outf.rdbuf());           // redirect std::cout to out.txt!

  switch (t.r()) {
    case 2: {
      // itensor::printfln("t=\n%s", t);
      cout << "placeholder_for_name =" << endl << t;
      auto inds = t.inds();
      for (int i = 1; i <= inds[0].m(); i++) {
        for (int j = 1; j <= inds[1].m(); j++) {
          cout << noshowpos << "(" << i << "," << j << ") "
               << t.cplx(inds[0](i), inds[1](j)).real() << showpos
               << t.cplx(inds[0](i), inds[1](j)).imag() << "i" << endl;
        }
      }
      break;
    }
    case 3: {
      // itensor::printfln("t=\n%s", t);
      cout << "placeholder_for_name =" << endl << t;
      auto inds = t.inds();
      for (int i = 1; i <= inds[0].m(); i++) {
        for (int j = 1; j <= inds[1].m(); j++) {
          for (int k = 1; k <= inds[2].m(); k++) {
            cout << noshowpos << "(" << i << "," << j << "," << k << ") "
                 << t.cplx(inds[0](i), inds[1](j), inds[2](k)).real() << showpos
                 << t.cplx(inds[0](i), inds[1](j), inds[2](k)).imag() << "i"
                 << endl;
          }
        }
      }
      break;
    }
    default: {
      itensor::PrintData(t);
      break;
    }
  }

  cout.rdbuf(coutbuf);  // reset to standard output again

  cout.flags(old_settings);
  cout.precision(old_precision);

  outf.close();
}

/*
 * Write out tensor in binary itensor format to output file
 * (presents ITensor::Index ids)
 *
 */
void writeTensorB(string const& fname, itensor::ITensor const& t) {
  ofstream outf;
  outf.open(fname, ios::out | ios::binary);

  itensor::write(outf, t);

  outf.close();
}

/*
 * Read tensor in format given by PrintData of itensor into t
 *
 * TODO Implement read for general rank of tensor
 *
 */
itensor::ITensor readTensorF(string const& fname) {
  ifstream inf;
  inf.open(fname, ios::in);
  if (!inf.good()) {
    cout << "Failed opening file: " << fname << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Reading: " << fname << endl;

  // Read tensor (object) name
  string nameLine;
  getline(inf, nameLine);
  string name = nameLine.substr(0, nameLine.find("=") - 1);
  cout << name << " ";

  // read in the line containing the rank of the tensor and IndexSet
  string tDescLine;
  getline(inf, tDescLine);

  // Parse the rank
  int rank;
  string rankStr;
  int rankBegin = tDescLine.find("r=") + 2;
  int rankEnd = tDescLine.find_first_of(':');
  rankStr = tDescLine.substr(rankBegin, rankEnd - rankBegin);
  rank = stoi(rankStr);
  cout << "RANK " << rank << endl;

  // Parse indices
  stringstream ssInds(tDescLine.substr(tDescLine.find_first_of('(')));
  char delimI = ' ';   // Delim between index entries
  char delimIF = ',';  // Delim for fields of single index entry
  string indToken, indCore, indPrime;
  vector<itensor::Index> inds = vector<itensor::Index>(rank);
  string indFields[4];
  int primeLvl;
  for (int i = 0; i < rank; i++) {
    getline(ssInds, indToken, delimI);
    // Separate core index properties and prime level
    indCore = indToken.substr(1, indToken.find_last_of(')') - 1);
    indPrime = indToken.substr(indToken.find_last_of(')') + 1);
    // Tokenize individual fields of the index
    stringstream ss(indCore);
    for (int j = 0; j < 4; j++) {
      getline(ss, indFields[j], delimIF);
    }
    // Get the prime level
    // TODO compares unsigned with signed
    if (indPrime.size() == count(indPrime.begin(), indPrime.end(), '\'')) {
      // prime level <= 3 is indicated by ' or '' or '''
      primeLvl = indPrime.size();
    } else {
      // prime level > 3 is indicated as 'n where n is the prime level
      primeLvl = stoi(indPrime.substr(1));
    }

    // TODO? convert to switch - requires enum and string to enum map
    if (indFields[2] == TAG_IT_ULINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), ULINK);
    } else if (indFields[2] == TAG_IT_RLINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), RLINK);
    } else if (indFields[2] == TAG_IT_DLINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), DLINK);
    } else if (indFields[2] == TAG_IT_LLINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), LLINK);
    } else if (indFields[2] == TAG_IT_HSLINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), HSLINK);
    } else if (indFields[2] == TAG_IT_VSLINK) {
      inds[i] = itensor::Index(indFields[0], stoi(indFields[1]), VSLINK);
    }

    inds[i].prime(primeLvl);
  }
  cout << itensor::IndexSet(inds) << endl;
  // Construct Tensor
  auto t = itensor::ITensor(inds);

  // Skip line with tensor "scale" and "norm" information
  string metaLine;
  getline(inf, metaLine);

  // Get individual tensor elements
  char delimIvs = ',';
  string itElem, itIv;
  int ivs[rank];
  string valStr, valRe, valIm;
  int cplxPos;
  complex<double> itElemVal;
  while (getline(inf, itElem) && (itElem.length() > 0)) {
    // Parse indices value
    stringstream itElemIvs(itElem.substr(1, itElem.find_first_of(')') - 1));
    for (int i = 0; i < rank; i++) {
      getline(itElemIvs, itIv, delimIvs);
      ivs[i] = stoi(itIv);
    }
    // Parse tensor element value
    valStr = itElem.substr(itElem.find_first_of(')') + 1);
    cplxPos = valStr.find_last_of("+-");
    valRe = valStr.substr(0, cplxPos);
    valIm = valStr.substr(cplxPos, valStr.length() - cplxPos - 1);
    itElemVal = complex<double>(stod(valRe), stod(valIm));
    switch (rank) {
      case 2: {
        t.set(inds[0](ivs[0]), inds[1](ivs[1]), itElemVal);
        break;
      }
      case 3: {
        t.set(inds[0](ivs[0]), inds[1](ivs[1]), inds[2](ivs[2]), itElemVal);
        break;
      }
      default: {
        cout << "ERROR: Unsupported tensor rank" << endl;
        exit(EXIT_FAILURE);
        break;
      }
    }
  }

  inf.close();

  return t;
}

/*
 * Read in tensor in binary itensor format from input file
 *
 */
itensor::ITensor readTensorB(string const& fname) {
  ifstream inf;
  inf.open(fname, ios::out | ios::binary);
  if (!inf.good()) {
    cout << "Failed opening file: " << fname << "\n";
    exit(EXIT_FAILURE);
  }
  cout << "Reading: " << fname << "\n";

  itensor::ITensor t;
  itensor::read(inf, t);

  inf.close();

  return t;
}