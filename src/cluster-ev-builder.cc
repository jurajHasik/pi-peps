#include "pi-peps/config.h"
#include "pi-peps/cluster-ev-builder.h"

#define pow_2(a) ((a) * (a))

using namespace itensor;

EVBuilder::EVBuilder(std::string in_name, Cluster const& cls, CtmEnv const& env)
  : name(in_name), p_cluster(&cls), p_ctmEnv(&env) {}

/*
 * TODO include (arbitrary) rotation matrix on physical index of
 *      on-site tensor T as argument
 *
 */
MpoNS EVBuilder::getTOT(MPO_1S mpo,
                        std::string siteId,
                        int primeLvl,
                        bool DBG) const {
  // Construct MPO
  auto op = getSpinOp(mpo, p_cluster->mphys.at(siteId), DBG);

  return getTOT(op, siteId, primeLvl, DBG);
}

MpoNS EVBuilder::getTOT(MPO_1S mpo,
                        Vertex const& v,
                        int primeLvl,
                        bool DBG) const {
  auto siteId = p_cluster->vertexToId(v);

  // Construct MPO
  auto op = getSpinOp(mpo, p_cluster->mphys.at(siteId), DBG);

  return getTOT(op, siteId, primeLvl, DBG);
}

MpoNS EVBuilder::getTOT(ITensor const& op,
                        std::string siteId,
                        int primeLvl,
                        bool DBG) const {
  if (DBG)
    std::cout << "===== getTOT called for site: " << siteId
              << " =====" << std::endl;
  if (op.r() < 2) {
    std::cout << "getTOT op.rank() < 2" << std::endl;
    exit(EXIT_FAILURE);
  }
  // TODO proper check 2 PHYS with prime level 0 and 1, everything else is AUX
  // if ((op.inds()[0].type() != PHYS) || (op.inds()[1].type() != PHYS)) {
  //     std::cout <<"getTOT op does not have two PHYS indices"<< std::endl;
  //     exit(EXIT_FAILURE);
  // }
  /*
   * Construct on-site MPO given by the contraction of bra & ket
   * on-site tensor T^dag & T through physical index s
   *
   *      I7 I6              I3 I2
   *       | /       _       | /
   *      |T*|~~s'~~|O|~~s~~|T |          =>
   *      / |               / |
   *    I4  I5             I0 I1
   *
   *                     I7(x)I3
   *                       ||
   *  =>        I4(x)I0==|T*OT|==I6(x)I2
   *                       ||
   *                    I5(x)I1
   *
   * where indices I[4..7] correspond to bra on-site tensor and
   * indices I[0..3] to ket. (x) denotes a tensor product of indices.
   * To obtain final form of TOT we need to cast tensor product of
   * indices, say I4(x)I0, into a single index I_XH
   * of size dim(I_XH) = dim(I*)^2 = D^2 = d
   * in accordance with CTM tensor network as defined in ctm-cluster.cc
   * Therefore we define a conversion tensor Y as follows
   *            _
   *       I4--| \
   *           |Y --I_XH =: Y(h)
   *       I0--|_/
   *
   * with the only non-zero elements being
   *
   *   Y(I4=i, I0=j, I_XH=D*(i-1)+j) = 1.0
   *
   * then we obtain X with proper indices as
   *
   * TOT = Y(h)*Y(h')*Y(v)*Y(v')*|T*OT|
   *
   */

  ITensor const& T = p_cluster->sites.at(siteId);

  // Get physical index of T and op
  auto s = p_cluster->mphys.at(siteId);
  auto opI = noprime(findtype(op, PHYS));

  if (DBG) {
    Print(s);
    Print(opI);
  }

  ITensor D = (s == opI) ? ITensor(1.0) : delta(s, opI);
  auto TOT = (((T * D) * op) * prime(D, 1)) *
             (dag(T).prime(AUXLINK, p_cluster->BRAKET_OFFSET).prime(PHYS, 1));

  MpoNS result;
  result.nSite = 1;
  result.mpo.push_back(TOT);
  result.siteIds.push_back(siteId);

  if (DBG)
    std::cout << "===== getTOT done =====" << siteId << std::endl;

  return result;
}

/*
 * TODO consider imaginary part of the result as well
 * TODO optimize memory usage (potentially)
 *
 */
double EVBuilder::eV_1sO_1sENV(MPO_1S op1s, Vertex const& v, bool DBG) const {
  auto mpo = getTOT(op1s, v, 0, DBG);
  return eV_1sO_1sENV(mpo, v, DBG);
}

double EVBuilder::eV_1sO_1sENV(MpoNS const& op,
                               Vertex const& v,
                               bool DBG) const {
  auto siteId = p_cluster->vertexToId(v);

  if (op.nSite != 1) {
    std::cout << "MPO with #sites != 1 (#sites = " << op.nSite << std::endl;
    exit(EXIT_FAILURE);
  }
  if (op.siteIds[0] != siteId) {
    std::cout << "WARNING: MPO constructed on site " << op.siteIds[0]
              << " inserted at site " << siteId << std::endl;
  }

  auto ev = p_ctmEnv->C_LU.at(siteId);
  ev *= p_ctmEnv->T_L.at(siteId);
  ev *= p_ctmEnv->C_LD.at(siteId);

  ev *= p_ctmEnv->T_U.at(siteId);
  // substitute original on-site tensor for op at position site
  if (DBG)
    std::cout << "OP inserted at " << v << " -> " << siteId << std::endl;
  ev *= op.mpo[0];
  ev *= p_ctmEnv->T_D.at(siteId);

  ev *= p_ctmEnv->C_RU.at(siteId);
  ev *= p_ctmEnv->T_R.at(siteId);
  ev *= p_ctmEnv->C_RD.at(siteId);

  return sumels(ev) / getNorm_Rectangle(DBG, v, v);
}

// entanglement of the corner
std::vector<double> EVBuilder::eeCorner_1s_inner(Vertex const& v,
                                                 bool DBG) const {
  auto getSiteBraKet = [this](std::string const& id) {
    return p_cluster->sites.at(id) *
           dag(p_cluster->sites.at(id))
             .prime(AUXLINK, p_cluster->BRAKET_OFFSET);
  };

  auto computeEE = [](ITensor const& t) {
    // Check hermicity
    auto i1 = t.inds()[0];
    auto i2 = t.inds()[1];
    auto i1T = Index("i1T", i1.m());
    auto i2T = Index("i2T", i2.m());

    // auto t_sym = 0.5 * ((t * delta(i1, i1T)) * delta(i2, i2T) +
    //                     (dag(t) * delta(i1, i2T)) * delta(i2, i1T));
    // auto t_asym = 0.5 * ((t * delta(i1, i1T)) * delta(i2, i2T) -
    //                      (dag(t) * delta(i1, i2T)) * delta(i2, i1T));
    // //
    // std::cout << "norm(t_sym): " << norm(t_sym)
    //           << " norm(t_asym): " << norm(t_asym) << std::endl;

    //
    ITensor U(i1), S, Vt;
    svd(t, U, S, Vt, {"Truncate", false});

    // compute normalization const
    auto i1d = S.inds()[0];
    auto i2d = S.inds()[1];
    std::vector<double> elemD(i1d.m(), 0.0);
    double trD = 0.0;
    for (int i = 1; i <= i1d.m(); i++) {
      elemD[i] =
        (S.real(i1d(i), i2d(i)) > 0.0) ? S.real(i1d(i), i2d(i)) : 1.0e-16;
      trD += elemD[i];
    }

    // EE = -Tr ( rho log (rho) ) = - 1/(sum_i D_i) sum_j (D_j log( D_j / (sum_k
    // D_k)) )
    //                            = - 1/(sum_i D_i) sum_j (D_j (log D_j - log
    //                            (sum_k D_k)) )
    double ee = 0.0;
    for (int j = 1; j <= i1d.m(); j++)
      ee += -(1.0 / trD) * elemD[j] * (std::log(elemD[j]) - std::log(trD));

    return ee;
  };

  auto siteId = p_cluster->vertexToId(v);

  std::vector<double> eecorner;
  ITensor ev;

  // upper-left corner
  // ev = p_ctmEnv->C_RU.at(siteId);
  // ev *= p_ctmEnv->T_R.at(siteId);
  // ev *= p_ctmEnv->C_RD.at(siteId);

  // ev *= p_ctmEnv->T_D.at(siteId);
  // ev *= getSiteBraKet(siteId);
  // ev *= p_ctmEnv->T_U.at(siteId);

  // ev *= p_ctmEnv->C_LD.at(siteId);
  // ev *= p_ctmEnv->T_L.at(siteId);
  ev = p_ctmEnv->C_LU.at(siteId);
  eecorner.push_back(computeEE(ev));

  // upper-right corner
  // ev = p_ctmEnv->C_LU.at(siteId);
  // ev *= p_ctmEnv->T_L.at(siteId);
  // ev *= p_ctmEnv->C_LD.at(siteId);

  // ev *= p_ctmEnv->T_D.at(siteId);
  // ev *= getSiteBraKet(siteId);
  // ev *= p_ctmEnv->T_U.at(siteId);

  // ev *= p_ctmEnv->C_RD.at(siteId);
  // ev *= p_ctmEnv->T_R.at(siteId);
  ev = p_ctmEnv->C_RU.at(siteId);
  eecorner.push_back(computeEE(ev));

  // lower-right corner
  // ev = p_ctmEnv->C_LU.at(siteId);
  // ev *= p_ctmEnv->T_L.at(siteId);
  // ev *= p_ctmEnv->C_LD.at(siteId);

  // ev *= p_ctmEnv->T_D.at(siteId);
  // ev *= getSiteBraKet(siteId);
  // ev *= p_ctmEnv->T_U.at(siteId);

  // ev *= p_ctmEnv->C_RU.at(siteId);
  // ev *= p_ctmEnv->T_R.at(siteId);
  ev = p_ctmEnv->C_RD.at(siteId);
  eecorner.push_back(computeEE(ev));

  // lower-left corner
  // ev = p_ctmEnv->C_RU.at(siteId);
  // ev *= p_ctmEnv->T_R.at(siteId);
  // ev *= p_ctmEnv->C_RD.at(siteId);

  // ev *= p_ctmEnv->T_D.at(siteId);
  // ev *= getSiteBraKet(siteId);
  // ev *= p_ctmEnv->T_U.at(siteId);

  // ev *= p_ctmEnv->C_LU.at(siteId);
  // ev *= p_ctmEnv->T_L.at(siteId);
  ev = p_ctmEnv->C_LD.at(siteId);
  eecorner.push_back(computeEE(ev));

  return eecorner;
}

std::vector<double> EVBuilder::eeCorner_1s_outer(Vertex const& v,
                                                 bool DBG) const {
  auto getSiteBraKet = [this](std::string const& id) {
    return p_cluster->sites.at(id) *
           dag(p_cluster->sites.at(id))
             .prime(AUXLINK, p_cluster->BRAKET_OFFSET);
  };

  auto computeEE = [](ITensor const& t) {
    // Check hermicity
    auto i1 = t.inds()[0];
    auto i2 = t.inds()[1];
    auto i1T = Index("i1T", i1.m());
    auto i2T = Index("i2T", i2.m());

    // auto t_sym = 0.5 * ((t * delta(i1, i1T)) * delta(i2, i2T) +
    //                     (dag(t) * delta(i1, i2T)) * delta(i2, i1T));
    // auto t_asym = 0.5 * ((t * delta(i1, i1T)) * delta(i2, i2T) -
    //                      (dag(t) * delta(i1, i2T)) * delta(i2, i1T));
    // //
    // std::cout << "norm(t_sym): " << norm(t_sym)
    //           << " norm(t_asym): " << norm(t_asym) << std::endl;

    //
    ITensor U(i1), S, Vt;
    svd(t, U, S, Vt, {"Truncate", false});

    // compute normalization const
    auto i1d = S.inds()[0];
    auto i2d = S.inds()[1];
    std::vector<double> elemD(i1d.m(), 0.0);
    double trD = 0.0;
    for (int i = 1; i <= i1d.m(); i++) {
      elemD[i] =
        (S.real(i1d(i), i2d(i)) > 0.0) ? S.real(i1d(i), i2d(i)) : 1.0e-16;
      trD += elemD[i];
    }

    // EE = -Tr ( rho log (rho) ) = - 1/(sum_i D_i) sum_j (D_j log( D_j / (sum_k
    // D_k)) )
    //                            = - 1/(sum_i D_i) sum_j (D_j (log D_j - log
    //                            (sum_k D_k)) )
    double ee = 0.0;
    for (int j = 1; j <= i1d.m(); j++)
      ee += -(1.0 / trD) * elemD[j] * (std::log(elemD[j]) - std::log(trD));

    return ee;
  };

  auto siteId = p_cluster->vertexToId(v);

  std::vector<double> eecorner;
  ITensor ev;

  // upper-left corner
  ev = p_ctmEnv->C_RU.at(siteId);
  ev *= p_ctmEnv->T_R.at(siteId);
  ev *= p_ctmEnv->C_RD.at(siteId);

  ev *= p_ctmEnv->T_D.at(siteId);
  ev *= getSiteBraKet(siteId);
  ev *= p_ctmEnv->T_U.at(siteId);

  ev *= p_ctmEnv->C_LD.at(siteId);
  ev *= p_ctmEnv->T_L.at(siteId);
  eecorner.push_back(computeEE(ev));

  // upper-right corner
  ev = p_ctmEnv->C_LU.at(siteId);
  ev *= p_ctmEnv->T_L.at(siteId);
  ev *= p_ctmEnv->C_LD.at(siteId);

  ev *= p_ctmEnv->T_D.at(siteId);
  ev *= getSiteBraKet(siteId);
  ev *= p_ctmEnv->T_U.at(siteId);

  ev *= p_ctmEnv->C_RD.at(siteId);
  ev *= p_ctmEnv->T_R.at(siteId);
  eecorner.push_back(computeEE(ev));

  // lower-right corner
  ev = p_ctmEnv->C_LU.at(siteId);
  ev *= p_ctmEnv->T_L.at(siteId);
  ev *= p_ctmEnv->C_LD.at(siteId);

  ev *= p_ctmEnv->T_D.at(siteId);
  ev *= getSiteBraKet(siteId);
  ev *= p_ctmEnv->T_U.at(siteId);

  ev *= p_ctmEnv->C_RU.at(siteId);
  ev *= p_ctmEnv->T_R.at(siteId);
  eecorner.push_back(computeEE(ev));

  // lower-left corner
  ev = p_ctmEnv->C_RU.at(siteId);
  ev *= p_ctmEnv->T_R.at(siteId);
  ev *= p_ctmEnv->C_RD.at(siteId);

  ev *= p_ctmEnv->T_D.at(siteId);
  ev *= getSiteBraKet(siteId);
  ev *= p_ctmEnv->T_U.at(siteId);

  ev *= p_ctmEnv->C_LU.at(siteId);
  ev *= p_ctmEnv->T_L.at(siteId);
  eecorner.push_back(computeEE(ev));

  return eecorner;
}

/*
 * TODO implement evaluation on arbitrary large cluster
 *
 */
double EVBuilder::eV_2sO_Rectangle(
  std::pair<itensor::ITensor, itensor::ITensor> const& Op,
  Vertex const& v1,
  Vertex const& v2,
  bool DBG) const {
  return get2SOPTN(DBG, Op, v1, v2) / getNorm_Rectangle(DBG, v1, v2);
}

// TODO REDUNDANCY FOR HANDLING s1 = s2 case
double EVBuilder::getNorm_Rectangle(bool DBG,
                                    Vertex const& v1,
                                    Vertex const& v2) const {
  auto o1(getTOT(MPO_Id, v1, 0, DBG));
  auto o2(getTOT(MPO_Id, v2, 0, DBG));

  return get2SOPTN(DBG, std::make_pair(o1.mpo[0], o2.mpo[0]), v1, v2);
}

double EVBuilder::get2SOPTN(bool DBG,
                            std::pair<ITensor, ITensor> const& Op,
                            Vertex const& v1,
                            Vertex const& v2) const {
  auto tN = EVBuilder::insert2S(DBG, Op, v1, v2);

  if (tN.r() > 0) {
    std::string error_msg = "Unexpected rank r=" + tN.r();
    throw std::runtime_error(error_msg);
  }
  if (DBG)
    std::cout << "===== EVBuilder::get2SOPTN done =====" << std::string(36, '=')
              << std::endl;

  return sumels(tN);
}

ITensor EVBuilder::redDenMat_2S(Vertex const& v1,
                                Vertex const& v2,
                                bool DBG) const {
  auto BRAKET_OFFSET = p_cluster->BRAKET_OFFSET;

  // find corresponding sites in elementary cluster
  auto id1 = p_cluster->vertexToId(v1);
  auto id2 = p_cluster->vertexToId(v2);
  // find the index of site given its elem position within cluster
  auto pI1 = p_cluster->mphys.at(id1);
  auto pI2 = p_cluster->mphys.at(id2);

  auto tmp_Op_v1 = p_cluster->sites.at(id1) *
                   delta(prime(pI1, 1), prime(pI1, BRAKET_OFFSET)) *
                   dag(p_cluster->sites.at(id1)).prime(BRAKET_OFFSET);

  auto tmp_Op_v2 = p_cluster->sites.at(id2) *
                   delta(prime(pI2, 1), prime(pI2, BRAKET_OFFSET)) *
                   dag(p_cluster->sites.at(id2)).prime(BRAKET_OFFSET);

  return insert2S(DBG, std::make_pair(tmp_Op_v1, tmp_Op_v2), v1, v2);
}

ITensor EVBuilder::insert2S(bool DBG,
                            std::pair<ITensor, ITensor> const& Op,
                            Vertex const& v1,
                            Vertex const& v2) const {
  using DIRECTION = CtmEnv::DIRECTION;

  int const tmp_prime_offset = 10;

  auto vToId = [this](Vertex const& v) { return p_cluster->vertexToId(v); };

  auto getSiteBraKet = [this, &vToId](Vertex const& v) {
    return p_cluster->sites.at(vToId(v)) *
           dag(p_cluster->sites.at(vToId(v)))
             .prime(AUXLINK, p_cluster->BRAKET_OFFSET);
  };

  auto applyDeltaEdge = [this](ITensor& t, Vertex const& v, DIRECTION edge,
                               DIRECTION dir) {
    // (edge = LEFT or RIGHT => dir = UP or DOWN) or
    // (edge = UP or DOWN => dir = LEFT or RIGHT)
    if ((edge == dir) || ((edge + 2) % 4 == dir)) {
      std::cout << "[get2SOPTN::applyDeltaEdge] Invalid input: edge= " << edge
                << " dir: " << dir << std::endl;
      throw std::runtime_error("[get2SOPTN::applyDeltaEdge] Invalid input");
    }

    Shift s;
    switch (dir) {
      case DIRECTION::LEFT: {
        s = Shift(1, 0);
        break;
      }
      case DIRECTION::UP: {
        s = Shift(0, 1);
        break;
      }
      case DIRECTION::RIGHT: {
        s = Shift(-1, 0);
        break;
      }
      case DIRECTION::DOWN: {
        s = Shift(0, -1);
        break;
      }
    }

    t *= delta(p_ctmEnv->tauxByVertex(edge, v + s, dir),
               p_ctmEnv->tauxByVertex(edge, v, (dir + 2) % 4));
  };

  auto applyDeltaSite = [this](ITensor& t, Vertex const& v, DIRECTION dir) {
    Shift s;
    switch (dir) {
      case DIRECTION::LEFT: {
        s = Shift(-1, 0);
        break;
      }
      case DIRECTION::UP: {
        s = Shift(0, -1);
        break;
      }
      case DIRECTION::RIGHT: {
        s = Shift(1, 0);
        break;
      }
      case DIRECTION::DOWN: {
        s = Shift(0, 1);
        break;
      }
    }

    t *= p_cluster->DContract(v + s, (dir + 2) % 4, v, dir);
    t *= p_cluster->DContract(v + s, (dir + 2) % 4, v, dir)
           .prime(p_cluster->BRAKET_OFFSET);
  };

  if (DBG)
    std::cout << "===== EVBuilder::get2SOPTN called ====="
              << std::string(34, '=') << std::endl;
  /*
   *  Contract network defined as a rectangle by sites s1 and s2
   *
   *  s1b(efore)s2 = true     s1b2 = false
   *  x1<x2 && y1<y2          x1<x2 && y1>y2
   *  && (x2-x1)>(y2-y1)      && (x2-x1)>(y1-y2)
   *
   *  C T  T ... T  T  C      C T  T ... T  T C
   *  T v1 . ... .  .  T      T .  . ... . v2 T
   *  T .  . ... .  v2 T      T v1 . ... .  . T
   *  C T  T ... T  T  C  or  C T  T ... T  T C
   *
   *  && (x2-x1)<(y2-y1)      && (x2-x1)<(y1-y2)
   *
   *  C T  T C                C T  T C
   *  T v1 . T                T . v2 T
   *  ...                     ...
   *  T . v2 T                T v1 . T
   *  C T  T C            or  C T  T C
   */
  bool singleSite = false;  // Assume v1 != v2
  bool s1bs2 = true;        // Assume y1 <= y2
  bool wBEh = true;         // Assume width of rectangle >= height

  // Perform some coord manipulation (assumed coords are >= 0) and
  // s1.first <= s2.second && s1.second <= s2.second
  if ((v1.r[0] < 0) || (v1.r[1] < 0) || (v2.r[0] < 0) || (v2.r[1] < 0)) {
    std::cout << "[get2SOPTN] Improper coordinates of sites v1, v2"
              << std::endl;
    throw std::runtime_error("[get2SOPTN] Invalid input");
  }
  if (v1.r[0] > v2.r[0]) {
    // TODO transform input
  }

  if (v1 == v2) {
    if (DBG)
      std::cout << "s1 = s2 => Computing norm for single site" << std::endl;
    singleSite = true;
  }

  Vertex v_init = v1;
  int sXdiff = std::abs(v2.r[0] - v1.r[0]);
  int sYdiff = std::abs(v2.r[1] - v1.r[1]);

  // Analyse which branch sequence of contraction to perform
  if (sXdiff < sYdiff) {
    wBEh = false;
    if (DBG)
      std::cout << "TN Width < Height => contracting row by row" << std::endl;
  }
  if (v1.r[1] > v2.r[1]) {
    s1bs2 = false;
    v_init = v1 + sYdiff * Shift(0, -1);
  }
  if (!(sXdiff >= sYdiff))
    wBEh = false;

  Vertex v = v_init;
  ITensor tN;
  if (wBEh) {
    // 1) ##### Construct LEFT edge
    // ########################################################
    if (DBG)
      std::cout << "C_LU[" << v << " => " << vToId(v) << "]" << std::endl;
    tN = p_ctmEnv->C_LU.at(vToId(v));
    for (int row = 0; row <= sYdiff; row++) {
      if (row > 0)
        v = v + Shift(0, 1);

      if (DBG)
        std::cout << "T_L[" << v << " =>" << vToId(v) << "]" << std::endl;

      if (row > 0)
        applyDeltaEdge(tN, v, DIRECTION::LEFT, DIRECTION::DOWN);
      tN *= prime(p_ctmEnv->T_L.at(vToId(v)), AUXLINK, tmp_prime_offset * row);
    }
    if (DBG)
      std::cout << "C_LD[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_LD.at(vToId(v));
    if (DBG)
      Print(tN);
    // ##### Construct LEFT edge DONE
    // ###################################################### 2) ##### APPEND
    // COLUMNS #############################################################
    for (int col = 0; col <= sXdiff; col++) {
      v = v_init + col * Shift(1, 0);

      if (DBG)
        std::cout << "T_U[" << v << " => " << vToId(v) << "]" << std::endl;

      if (col > 0)
        applyDeltaEdge(tN, v, DIRECTION::UP, DIRECTION::RIGHT);
      tN *= p_ctmEnv->T_U.at(vToId(v));

      if (DBG)
        Print(tN);

      for (int row = 0; row <= sYdiff; row++) {
        if (row > 0)
          v = v + Shift(0, 1);

        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        int tmp_pl0 = std::min(0, row - 1) * tmp_prime_offset;
        int tmp_pl1 = row * tmp_prime_offset;
        Index tmp_down, tmp_right;
        if (row > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::UP)
          tmp_down =
            prime(p_cluster->AIc(v + Shift(0, -1), DIRECTION::DOWN), tmp_pl0);
        } else {
          tmp_down = p_cluster->AIc(v, DIRECTION::UP);
        }
        if (col > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::LEFT)
          tmp_right =
            prime(p_cluster->AIc(v + Shift(-1, 0), DIRECTION::RIGHT), tmp_pl1);
        } else {
          tmp_right = prime(p_cluster->AIc(v, DIRECTION::LEFT), tmp_pl1);
        };

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 =
          combiner(prime(p_cluster->AIc(v, DIRECTION::UP), tmp_pl1),
                   prime(p_cluster->AIc(v, DIRECTION::UP),
                         tmp_pl1 + p_cluster->BRAKET_OFFSET),
                   prime(p_cluster->AIc(v, DIRECTION::LEFT), tmp_pl1),
                   prime(p_cluster->AIc(v, DIRECTION::LEFT),
                         tmp_pl1 + p_cluster->BRAKET_OFFSET));

        tN *= tmp_cmb0;
        // TODO use delta instead of reindex
        tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        if (v == v1) {
          tN *= prime(Op.first, AUXLINK, tmp_pl1) * tmp_cmb1;
        } else if (v == v2) {
          tN *= prime(Op.second, AUXLINK, tmp_pl1) * tmp_cmb1;
        } else {
          // tN *= getSiteBraKet(v);
          tN *= prime(getSiteBraKet(v), AUXLINK, tmp_pl1) * tmp_cmb1;
        }
      }

      if (DBG)
        std::cout << "T_D[" << v << " => " << vToId(v) << "]" << std::endl;

      if (col > 0)
        applyDeltaEdge(tN, v, DIRECTION::DOWN, DIRECTION::RIGHT);
      tN *=
        prime(p_ctmEnv->T_D.at(vToId(v)), AUXLINK, sYdiff * tmp_prime_offset);

      if (DBG)
        std::cout << ">>>>> Appended col X= " << col << " <<<<<" << std::endl;
      if (DBG)
        Print(tN);
    }
    // 2) ##### APPEND COLUMNS DONE
    // ######################################################## 3) #####
    // Construct RIGHT edge
    // #######################################################
    v = v_init + sXdiff * Shift(1, 0);
    if (DBG)
      std::cout << "C_RU[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_RU.at(vToId(v));
    for (int row = 0; row <= sYdiff; row++) {
      if (row > 0)
        v = v + Shift(0, 1);
      if (DBG)
        std::cout << "T_R[" << v << " =>" << vToId(v) << "]" << std::endl;

      if (row > 0)
        applyDeltaEdge(tN, v, DIRECTION::RIGHT, DIRECTION::DOWN);
      tN *= prime(p_ctmEnv->T_R.at(vToId(v)), AUXLINK, row * tmp_prime_offset);
    }
    if (DBG)
      std::cout << "C_RD[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_RD.at(vToId(v));
    // 3) ##### Construct RIGHT edge DONE
    // ##################################################
  } else {
    // 1) ##### Construct UP edge
    // ########################################################
    if (DBG)
      std::cout << "C_LU[" << v << " => " << vToId(v) << "]" << std::endl;
    tN = p_ctmEnv->C_LU.at(vToId(v));
    for (int col = 0; col <= sXdiff; col++) {
      if (col > 0)
        v = v + Shift(1, 0);

      if (DBG)
        std::cout << "T_U[" << v << " =>" << vToId(v) << "]" << std::endl;

      if (col > 0)
        applyDeltaEdge(tN, v, DIRECTION::UP, DIRECTION::RIGHT);
      tN *= prime(p_ctmEnv->T_U.at(vToId(v)), AUXLINK, tmp_prime_offset * col);
    }
    if (DBG)
      std::cout << "C_RU[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_RU.at(vToId(v));
    if (DBG)
      Print(tN);
    // ##### Construct LEFT edge DONE
    // ###################################################### 2) ##### APPEND
    // ROWS #############################################################
    for (int row = 0; row <= sYdiff; row++) {
      v = v_init + row * Shift(0, 1);

      if (DBG)
        std::cout << "T_L[" << v << " => " << vToId(v) << "]" << std::endl;

      if (row > 0)
        applyDeltaEdge(tN, v, DIRECTION::LEFT, DIRECTION::DOWN);
      tN *= p_ctmEnv->T_L.at(vToId(v));

      if (DBG)
        Print(tN);

      for (int col = 0; col <= sXdiff; col++) {
        if (col > 0)
          v = v + Shift(1, 0);

        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        int tmp_pl0 = std::min(0, col - 1) * tmp_prime_offset;
        int tmp_pl1 = col * tmp_prime_offset;
        Index tmp_down, tmp_right;

        if (row > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::UP)
          tmp_down =
            prime(p_cluster->AIc(v + Shift(0, -1), DIRECTION::DOWN), tmp_pl1);
        } else {
          tmp_down = prime(p_cluster->AIc(v, DIRECTION::UP), tmp_pl1);
        }
        if (col > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::LEFT)
          tmp_right =
            prime(p_cluster->AIc(v + Shift(-1, 0), DIRECTION::RIGHT), tmp_pl1);
        } else {
          tmp_right = p_cluster->AIc(v, DIRECTION::LEFT);
        };

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 =
          combiner(prime(p_cluster->AIc(v, DIRECTION::UP), tmp_pl1),
                   prime(p_cluster->AIc(v, DIRECTION::UP),
                         tmp_pl1 + p_cluster->BRAKET_OFFSET),
                   prime(p_cluster->AIc(v, DIRECTION::LEFT), tmp_pl1),
                   prime(p_cluster->AIc(v, DIRECTION::LEFT),
                         tmp_pl1 + p_cluster->BRAKET_OFFSET));

        tN *= tmp_cmb0;
        // TODO use delta instead of reindex
        tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        if (v == v1) {
          tN *= prime(Op.first, AUXLINK, tmp_pl1) * tmp_cmb1;
        } else if (v == v2) {
          tN *= prime(Op.second, AUXLINK, tmp_pl1) * tmp_cmb1;
        } else {
          // tN *= getSiteBraKet(v);
          tN *= prime(getSiteBraKet(v), AUXLINK, tmp_pl1) * tmp_cmb1;
        }
      }

      if (DBG)
        std::cout << "T_R[" << v << " => " << vToId(v) << "]" << std::endl;

      if (row > 0)
        applyDeltaEdge(tN, v, DIRECTION::RIGHT, DIRECTION::DOWN);
      tN *=
        prime(p_ctmEnv->T_R.at(vToId(v)), AUXLINK, sXdiff * tmp_prime_offset);

      if (DBG)
        std::cout << ">>>>> Appended row X= " << row << " <<<<<" << std::endl;
      if (DBG)
        Print(tN);
    }
    // 3) ##### Construct RIGHT edge
    // #######################################################
    v = v_init + sYdiff * Shift(0, 1);
    if (DBG)
      std::cout << "C_LD[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_LD.at(vToId(v));
    for (int col = 0; col <= sXdiff; col++) {
      if (col > 0)
        v = v + Shift(1, 0);
      if (DBG)
        std::cout << "T_D[" << v << " =>" << vToId(v) << "]" << std::endl;

      if (col > 0)
        applyDeltaEdge(tN, v, DIRECTION::DOWN, DIRECTION::RIGHT);
      tN *= prime(p_ctmEnv->T_D.at(vToId(v)), AUXLINK, col * tmp_prime_offset);
    }
    if (DBG)
      std::cout << "C_RD[" << v << " => " << vToId(v) << "]" << std::endl;
    tN *= p_ctmEnv->C_RD.at(vToId(v));
    // 3) ##### Construct RIGHT edge DONE
    // ##################################################
  }

  return tN;
}

double EVBuilder::evalSS(Vertex const& v1,
                         Vertex const& v2,
                         std::vector<double> coefs,
                         bool DBG) const {
  // find corresponding sites in elementary cluster
  auto id1 = p_cluster->vertexToId(v1);
  auto id2 = p_cluster->vertexToId(v2);
  // find the index of site given its elem position within cluster
  auto pI1 = p_cluster->mphys.at(id1);
  auto pI2 = p_cluster->mphys.at(id2);

  auto opId = get2SiteSpinOP(OP2S_Id, pI1, pI2, DBG);
  auto n = contract2Smpo(opId, v1, v2, DBG);

  auto SPSM = std::make_pair(getSpinOp(MPO_S_P, pI1), getSpinOp(MPO_S_M, pI2));
  auto SMSP = std::make_pair(getSpinOp(MPO_S_M, pI1), getSpinOp(MPO_S_P, pI2));
  auto SZSZ = std::make_pair(getSpinOp(MPO_S_Z, pI1), getSpinOp(MPO_S_Z, pI2));

  auto spsm = contract2Smpo(SPSM, v1, v2, DBG);
  auto smsp = contract2Smpo(SMSP, v1, v2, DBG);
  auto szsz = contract2Smpo(SZSZ, v1, v2, DBG);

  double ss = (coefs[2] * szsz + 0.5 * (spsm + smsp)) / n;
  return ss;
}

double EVBuilder::eval2Smpo(OP_2S op2s,
                            Vertex const& v1,
                            Vertex const& v2,
                            bool DBG) const {
  return contract2Smpo(op2s, v1, v2, DBG) / contract2Smpo(OP2S_Id, v1, v2, DBG);
}

double EVBuilder::eval2Smpo(std::pair<ITensor, ITensor> const& Op,
                            Vertex const& v1,
                            Vertex const& v2,
                            bool DBG) const {
  // find corresponding sites in elementary cluster
  auto id1 = p_cluster->vertexToId(v1);
  auto id2 = p_cluster->vertexToId(v2);
  // find the index of site given its elem position within cluster
  auto pI1 = p_cluster->mphys.at(id1);
  auto pI2 = p_cluster->mphys.at(id2);

  // Assume Op.first and Op.second have both pair of physical indices with
  // primes 0 and 1
  auto tmpOp = Op;
  auto tmpPI1 = noprime(findtype(tmpOp.first, PHYS));
  auto tmpPI2 = noprime(findtype(tmpOp.second, PHYS));

  tmpOp.first *= delta(tmpPI1, pI1);
  tmpOp.first *= delta(prime(tmpPI1), prime(pI1));
  tmpOp.second *= delta(tmpPI2, pI2);
  tmpOp.second *= delta(prime(tmpPI2), prime(pI2));

  return contract2Smpo(tmpOp, v1, v2, DBG) /
         contract2Smpo(OP2S_Id, v1, v2, DBG);
}

double EVBuilder::contract2Smpo(OP_2S op2s,
                                Vertex const& v1,
                                Vertex const& v2,
                                bool DBG) const {
  // find corresponding sites in elementary cluster
  auto id1 = p_cluster->vertexToId(v1);
  auto id2 = p_cluster->vertexToId(v2);
  // find the index of site given its elem position within cluster
  auto pI1 = p_cluster->mphys.at(id1);
  auto pI2 = p_cluster->mphys.at(id2);

  auto op = get2SiteSpinOP(op2s, pI1, pI2, DBG);

  return contract2Smpo(op, v1, v2, DBG);
}

double EVBuilder::contract2Smpo(std::pair<ITensor, ITensor> const& Op,
                                Vertex const& v1,
                                Vertex const& v2,
                                bool DBG) const {
  auto BRAKET_OFFSET = p_cluster->BRAKET_OFFSET;

  // find corresponding sites in elementary cluster
  auto id1 = p_cluster->vertexToId(v1);
  auto id2 = p_cluster->vertexToId(v2);
  // find the index of site given its elem position within cluster
  auto pI1 = p_cluster->mphys.at(id1);
  auto pI2 = p_cluster->mphys.at(id2);

  auto tmp_Op_v1 = (p_cluster->sites.at(id1) * Op.first) *
                   delta(prime(pI1, 1), prime(pI1, BRAKET_OFFSET)) *
                   dag(p_cluster->sites.at(id1)).prime(BRAKET_OFFSET);

  auto tmp_Op_v2 = (p_cluster->sites.at(id2) * Op.second) *
                   delta(prime(pI2, 1), prime(pI2, BRAKET_OFFSET)) *
                   dag(p_cluster->sites.at(id2)).prime(BRAKET_OFFSET);

  return get2SOPTN(DBG, std::make_pair(tmp_Op_v1, tmp_Op_v2), v1, v2);
}

double EVBuilder::eval2x2Diag11(OP_2S op2s, Vertex const& v1, bool DBG) const {
  return contract2x2Diag11(op2s, v1, DBG) / contract2x2Diag11(OP2S_Id, v1, DBG);
}

// Diagonal s1, s1+[1,1]
double EVBuilder::eval2x2Diag11(std::pair<ITensor, ITensor> const& Op,
                                Vertex const& v1,
                                bool DBG) const {
  return contract2x2Diag11(Op, v1, DBG) / contract2x2Diag11(OP2S_Id, v1, DBG);
}

double EVBuilder::contract2x2Diag11(OP_2S op2s,
                                    Vertex const& v1,
                                    bool DBG) const {
  auto vToId = [this](Vertex const& v) { return p_cluster->vertexToId(v); };

  auto v2 = v1 + Shift(1, 1);
  auto pI1 = p_cluster->mphys.at(vToId(v1));
  auto pI2 = p_cluster->mphys.at(vToId(v2));

  // if the indices are identical, create a dummy index for creation of
  // the operator
  std::pair<ITensor, ITensor> op;
  if (pI1 == pI2) {
    auto p_dummy = Index("dummyPhys", pI2.m(), pI2.type(), pI2.primeLevel());
    op = get2SiteSpinOP(op2s, pI1, p_dummy, DBG);
    op.second *= delta(p_dummy, pI2);
    op.second *= delta(prime(p_dummy), prime(pI2));
  } else {
    op = get2SiteSpinOP(op2s, pI1, pI2, DBG);
  }

  return contract2x2Diag11(op, v1, DBG);
}

double EVBuilder::contract2x2Diag11(std::pair<ITensor, ITensor> const& op,
                                    Vertex const& v1,
                                    bool DBG) const {
  using DIRECTION = CtmEnv::DIRECTION;

  const int AUXLINK_OFFSET = 100;

  auto vToId = [this](Vertex const& v) { return p_cluster->vertexToId(v); };

  auto getSiteBraKet = [this, &vToId](Vertex const& v) {
    return p_cluster->sites.at(vToId(v)) *
           dag(p_cluster->sites.at(vToId(v)))
             .prime(AUXLINK, p_cluster->BRAKET_OFFSET);
  };

  auto get2dirSiteCombiner = [this](ITensor& cmb, Vertex const& v,
                                    DIRECTION dir0, DIRECTION dir1) {
    cmb = combiner(p_cluster->AIc(v, dir0),
                   prime(p_cluster->AIc(v, dir0), p_cluster->BRAKET_OFFSET),
                   p_cluster->AIc(v, dir1),
                   prime(p_cluster->AIc(v, dir1), p_cluster->BRAKET_OFFSET));
  };

  auto readyToContract = [this](ITensor& t, DIRECTION direction,
                                Vertex const& v0, int dir0, Vertex const& v1,
                                int dir1) {
    // relabel env auxiliary indices
    t *= delta(p_ctmEnv->tauxByVertex(direction, v0, dir0),
               p_ctmEnv->tauxByVertex(direction, v1, dir1));
    // relabel site auxiliary indices
    t *= p_cluster->DContract(v0, dir0, v1, dir1);
    t *=
      prime(p_cluster->DContract(v0, dir0, v1, dir1), p_cluster->BRAKET_OFFSET);
  };

  auto getEdgeCombiners = [this](ITensor& cmb_v0, ITensor& cmb_v1,
                                 DIRECTION direction, Vertex const& v0,
                                 int dir0, Vertex const& v1, int dir1) {
    cmb_v0 = combiner(
      p_ctmEnv->tauxByVertex(direction, v0, dir0), p_cluster->AIc(v0, dir0),
      prime(p_cluster->AIc(v0, dir0), p_cluster->BRAKET_OFFSET));
    cmb_v1 = combiner(
      p_ctmEnv->tauxByVertex(direction, v1, dir1), p_cluster->AIc(v1, dir1),
      prime(p_cluster->AIc(v1, dir1), p_cluster->BRAKET_OFFSET));
  };

  if (DBG)
    std::cout << "===== EVBuilder::expVal2x2Diag11 called ====="
              << std::string(34, '=') << std::endl;

  // s1 sX
  // sY s2
  auto vX = v1 + Shift(1, 0);
  auto vY = v1 + Shift(0, 1);
  auto v2 = v1 + Shift(1, 1);

  // combiners
  ITensor cmb0, cmb1, cmb2, cmb3;

  // build upper left corner
  auto tN = p_ctmEnv->C_LU.at(vToId(v1)) * p_ctmEnv->T_U.at(vToId(v1));
  tN *= p_ctmEnv->T_L.at(vToId(v1));

  // get operator on site s1
  auto mpo1s = getTOT(op.first, vToId(v1), 0, DBG);
  get2dirSiteCombiner(cmb0, v1, DIRECTION::LEFT, DIRECTION::UP);
  tN = (tN * cmb0) * (mpo1s.mpo[0] * cmb0);
  // offset in case of on-site itensor being identical ( 1 site-TI iPEPS )
  tN.mapprime((int)DIRECTION::DOWN, DIRECTION::DOWN + AUXLINK_OFFSET, AUXLINK);
  tN.mapprime(DIRECTION::DOWN + p_cluster->BRAKET_OFFSET,
              DIRECTION::DOWN + p_cluster->BRAKET_OFFSET + AUXLINK_OFFSET,
              AUXLINK);

  if (DBG) {
    std::cout << ">>>>> 1) Upper Left corner done <<<<<" << std::endl;
    Print(tN);
  }

  // build right upper corner
  auto tmp = p_ctmEnv->C_RU.at(vToId(vX)) * p_ctmEnv->T_U.at(vToId(vX));
  tmp *= p_ctmEnv->T_R.at(vToId(vX));

  get2dirSiteCombiner(cmb0, vX, DIRECTION::UP, DIRECTION::RIGHT);
  tmp = (tmp * cmb0) * (getSiteBraKet(vX) * cmb0);

  if (DBG)
    Print(tmp);

  // TODO use deltas instead of reindex
  // readyToContract(tN,DIRECTION::UP,v1,DIRECTION::RIGHT,v2,DIRECTION::LEFT);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::UP, v1, DIRECTION::RIGHT, vX,
                   DIRECTION::LEFT);
  tN *= cmb0;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1));
  tN *= (tmp * cmb1);

  if (DBG) {
    std::cout << ">>>>> 2) Upper Right corner done <<<<<" << std::endl;
    Print(tN);
  }

  // build right down corner
  tmp = p_ctmEnv->C_RD.at(vToId(v2)) * p_ctmEnv->T_R.at(vToId(v2));
  tmp *= p_ctmEnv->T_D.at(vToId(v2));

  // get operator on site s1
  mpo1s = getTOT(op.second, vToId(v2), 0, DBG);
  get2dirSiteCombiner(cmb0, v2, DIRECTION::RIGHT, DIRECTION::DOWN);
  tmp = (tmp * cmb0) * (mpo1s.mpo[0] * cmb0);

  // readyToContract(tN,DIRECTION::RIGHT,vX,DIRECTION::DOWN,v2,DIRECTION::UP);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::RIGHT, vX, DIRECTION::DOWN, v2,
                   DIRECTION::UP);
  tN *= cmb0;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1));
  tN *= (tmp * cmb1);

  if (DBG) {
    std::cout << ">>>>> 3) Down Right corner done <<<<<" << std::endl;
    Print(tN);
  }
  // revert back the 1-site TI iPEPS primeLevel guard
  tN.mapprime(DIRECTION::DOWN + AUXLINK_OFFSET, (int)DIRECTION::DOWN, AUXLINK);
  tN.mapprime(DIRECTION::DOWN + p_cluster->BRAKET_OFFSET + AUXLINK_OFFSET,
              DIRECTION::DOWN + p_cluster->BRAKET_OFFSET, AUXLINK);

  tmp = p_ctmEnv->C_LD.at(vToId(vY)) * p_ctmEnv->T_D.at(vToId(vY));
  tmp *= p_ctmEnv->T_L.at(vToId(vY));

  get2dirSiteCombiner(cmb0, vY, DIRECTION::DOWN, DIRECTION::LEFT);
  tmp = (tmp * cmb0) * (getSiteBraKet(vY) * cmb0);

  if (DBG)
    Print(tmp);

  // readyToContract(tN,DIRECTION::LEFT,v1,DIRECTION::DOWN,vY,DIRECTION::UP);
  // readyToContract(tN,DIRECTION::DOWN,v2,DIRECTION::LEFT,vY,DIRECTION::RIGHT);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::LEFT, v1, DIRECTION::DOWN, vY,
                   DIRECTION::UP);
  getEdgeCombiners(cmb2, cmb3, DIRECTION::DOWN, v2, DIRECTION::LEFT, vY,
                   DIRECTION::RIGHT);

  tN = (tN * cmb0) * cmb2;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1),
               combinedIndex(cmb2), combinedIndex(cmb3));
  tN *= (tmp * cmb1) * cmb3;
  if (DBG)
    std::cout << ">>>>> 4) Down Left corner done <<<<<" << std::endl;

  if (tN.r() > 0) {
    std::cout << "[contract2x2Diag11] Unexpected rank r=" << tN.r()
              << std::endl;
    throw std::runtime_error("[contract2x2Diag11] observable is not a scalar");
  }

  if (DBG)
    std::cout << "===== EVBuilder::expVal2x2Diag11 done ====="
              << std::string(36, '=') << std::endl;

  return sumels(tN);
}

// Diagonal s1, s1+[-1,+1]
double EVBuilder::eval2x2DiagN11(OP_2S op2s, Vertex const& v1, bool DBG) const {
  return contract2x2DiagN11(op2s, v1, DBG) /
         contract2x2DiagN11(OP2S_Id, v1, DBG);
}

// Diagonal s1, s1+[-1,+1]
double EVBuilder::eval2x2DiagN11(std::pair<ITensor, ITensor> const& Op,
                                 Vertex const& v1,
                                 bool DBG) const {
  return contract2x2DiagN11(Op, v1, DBG) / contract2x2DiagN11(OP2S_Id, v1, DBG);
}

double EVBuilder::contract2x2DiagN11(OP_2S op2s,
                                     Vertex const& v1,
                                     bool DBG) const {
  auto vToId = [this](Vertex const& v) { return p_cluster->vertexToId(v); };

  auto v2 = v1 + Shift(1, 1);
  auto pI1 = p_cluster->mphys.at(vToId(v1));
  auto pI2 = p_cluster->mphys.at(vToId(v2));

  // if the indices are identical, create a dummy index for creation of
  // the operator
  std::pair<ITensor, ITensor> op;
  if (pI1 == pI2) {
    auto p_dummy = Index("dummyPhys", pI2.m(), pI2.type(), pI2.primeLevel());
    op = get2SiteSpinOP(op2s, pI1, p_dummy, DBG);
    op.second *= delta(p_dummy, pI2);
    op.second *= delta(prime(p_dummy), prime(pI2));
  } else {
    op = get2SiteSpinOP(op2s, pI1, pI2, DBG);
  }

  return contract2x2DiagN11(op, v1, DBG);
}

double EVBuilder::contract2x2DiagN11(std::pair<ITensor, ITensor> const& op,
                                     Vertex const& v1,
                                     bool DBG) const {
  using DIRECTION = CtmEnv::DIRECTION;

  const int AUXLINK_OFFSET = 100;

  auto vToId = [this](Vertex const& v) { return p_cluster->vertexToId(v); };

  auto getSiteBraKet = [this, &vToId](Vertex const& v) {
    return p_cluster->sites.at(vToId(v)) *
           dag(p_cluster->sites.at(vToId(v)))
             .prime(AUXLINK, p_cluster->BRAKET_OFFSET);
  };

  auto readyToContract = [this](ITensor& t, DIRECTION direction,
                                Vertex const& v0, int dir0, Vertex const& v1,
                                int dir1) {
    // relabel env auxiliary indices
    t *= delta(p_ctmEnv->tauxByVertex(direction, v0, dir0),
               p_ctmEnv->tauxByVertex(direction, v1, dir1));
    // relabel site auxiliary indices
    t *= p_cluster->DContract(v0, dir0, v1, dir1);
    t *=
      prime(p_cluster->DContract(v0, dir0, v1, dir1), p_cluster->BRAKET_OFFSET);
  };

  auto get2dirSiteCombiner = [this](ITensor& cmb, Vertex const& v,
                                    DIRECTION dir0, DIRECTION dir1) {
    cmb = combiner(p_cluster->AIc(v, dir0),
                   prime(p_cluster->AIc(v, dir0), p_cluster->BRAKET_OFFSET),
                   p_cluster->AIc(v, dir1),
                   prime(p_cluster->AIc(v, dir1), p_cluster->BRAKET_OFFSET));
  };

  auto getEdgeCombiners = [this](ITensor& cmb_v0, ITensor& cmb_v1,
                                 DIRECTION direction, Vertex const& v0,
                                 int dir0, Vertex const& v1, int dir1) {
    cmb_v0 = combiner(
      p_ctmEnv->tauxByVertex(direction, v0, dir0), p_cluster->AIc(v0, dir0),
      prime(p_cluster->AIc(v0, dir0), p_cluster->BRAKET_OFFSET));
    cmb_v1 = combiner(
      p_ctmEnv->tauxByVertex(direction, v1, dir1), p_cluster->AIc(v1, dir1),
      prime(p_cluster->AIc(v1, dir1), p_cluster->BRAKET_OFFSET));
  };

  if (DBG)
    std::cout << "===== EVBuilder::expVal2x2DiagN11 called ====="
              << std::string(34, '=') << std::endl;

  // sX s1
  // s2 sY
  auto vX = v1 + Shift(-1, 0);
  auto vY = v1 + Shift(0, 1);
  auto v2 = v1 + Shift(-1, 1);

  ITensor cmb0, cmb1, cmb2, cmb3;

  // build right upper corner
  auto tN = p_ctmEnv->C_RU.at(vToId(v1)) * p_ctmEnv->T_U.at(vToId(v1));
  tN *= p_ctmEnv->T_R.at(vToId(v1));

  // get operator on site s1
  auto mpo1s = getTOT(op.first, vToId(v1), 0, DBG);
  get2dirSiteCombiner(cmb0, v1, DIRECTION::UP, DIRECTION::RIGHT);
  tN = (tN * cmb0) * (mpo1s.mpo[0] * cmb0);
  // offset in case of on-site itensor being identical ( 1 site-TI iPEPS )
  tN.mapprime((int)DIRECTION::LEFT, DIRECTION::LEFT + AUXLINK_OFFSET, AUXLINK);
  tN.mapprime(DIRECTION::LEFT + p_cluster->BRAKET_OFFSET,
              DIRECTION::LEFT + p_cluster->BRAKET_OFFSET + AUXLINK_OFFSET,
              AUXLINK);

  if (DBG) {
    std::cout << ">>>>> 1) right upper corner done <<<<<" << std::endl;
    Print(tN);
  }

  // build right down corner
  auto tmp = p_ctmEnv->C_RD.at(vToId(vY)) * p_ctmEnv->T_R.at(vToId(vY));
  tmp *= p_ctmEnv->T_D.at(vToId(vY));

  get2dirSiteCombiner(cmb0, vY, DIRECTION::RIGHT, DIRECTION::DOWN);
  tmp = (tmp * cmb0) * (getSiteBraKet(vY) * cmb0);
  if (DBG)
    Print(tmp);

  // readyToContract(tN,DIRECTION::RIGHT,v1,DIRECTION::DOWN,vY,DIRECTION::UP);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::RIGHT, v1, DIRECTION::DOWN, vY,
                   DIRECTION::UP);
  tN *= cmb0;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1));
  tN *= (tmp * cmb1);
  if (DBG) {
    std::cout << ">>>>> 2) right down corner done <<<<<" << std::endl;
    Print(tN);
  }

  // build left down corner
  tmp = p_ctmEnv->C_LD.at(vToId(v2)) * p_ctmEnv->T_D.at(vToId(v2));
  tmp *= p_ctmEnv->T_L.at(vToId(v2));

  // get operator on site s1
  mpo1s = getTOT(op.second, vToId(v2), 0, DBG);
  get2dirSiteCombiner(cmb0, v2, DIRECTION::LEFT, DIRECTION::DOWN);
  tmp = (tmp * cmb0) * (mpo1s.mpo[0] * cmb0);

  // readyToContract(tN,DIRECTION::DOWN,vY,DIRECTION::LEFT,v2,DIRECTION::RIGHT);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::DOWN, vY, DIRECTION::LEFT, v2,
                   DIRECTION::RIGHT);
  tN *= cmb0;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1));
  tN *= (tmp * cmb1);
  if (DBG) {
    std::cout << ">>>>> 3) left down corner done <<<<<" << std::endl;
    Print(tN);
  }
  // revert back the 1-site TI iPEPS primeLevel guard
  tN.mapprime(DIRECTION::LEFT + AUXLINK_OFFSET, (int)DIRECTION::LEFT, AUXLINK);
  tN.mapprime(DIRECTION::LEFT + p_cluster->BRAKET_OFFSET + AUXLINK_OFFSET,
              DIRECTION::LEFT + p_cluster->BRAKET_OFFSET, AUXLINK);

  tmp = p_ctmEnv->C_LU.at(vToId(vX)) * p_ctmEnv->T_L.at(vToId(vX));
  tmp *= p_ctmEnv->T_U.at(vToId(vX));

  get2dirSiteCombiner(cmb0, vX, DIRECTION::LEFT, DIRECTION::UP);
  tmp = (tmp * cmb0) * (getSiteBraKet(vX) * cmb0);

  if (DBG)
    Print(tmp);

  // readyToContract(tN,DIRECTION::UP,v1,DIRECTION::LEFT,vX,DIRECTION::RIGHT);
  // readyToContract(tN,DIRECTION::LEFT,v2,DIRECTION::UP,vX,DIRECTION::DOWN);
  getEdgeCombiners(cmb0, cmb1, DIRECTION::UP, v1, DIRECTION::LEFT, vX,
                   DIRECTION::RIGHT);
  getEdgeCombiners(cmb2, cmb3, DIRECTION::LEFT, v2, DIRECTION::UP, vX,
                   DIRECTION::DOWN);

  tN = (tN * cmb0) * cmb2;
  tN = reindex(tN, combinedIndex(cmb0), combinedIndex(cmb1),
               combinedIndex(cmb2), combinedIndex(cmb3));
  tN *= (tmp * cmb1) * cmb3;

  if (DBG)
    std::cout << ">>>>> 4) left upper corner done <<<<<" << std::endl;

  if (tN.r() > 0) {
    std::cout << "[contract2x2DiagN11] Unexpected rank r=" << tN.r()
              << std::endl;
    throw std::runtime_error("[contract2x2DiagN11] Invalid input");
  }

  if (DBG)
    std::cout << "===== EVBuilder::expVal2x2DiagN11 done ====="
              << std::string(36, '=') << std::endl;

  return sumels(tN);
}

double EVBuilder::eval2x2op4s(OP_4S op4s, Vertex const& v1, bool DBG) const {
  return 0.0;
}

// ITensor EVBuilder::getT(ITensor const& s, std::array<Index, 4> const&
// plToEnv,
//     bool dbg) const
// {

//     Index aS(noprime(findtype(s, AUXLINK)));
//     Index pS(noprime(findtype(s, PHYS)));

//     // build |ket> part
//     ITensor res = s;
//     res = res * prime(conj(s), AUXLINK, 4);
//     if (dbg) Print(res);

//     // contract given indices into ENV compatible indices
//     auto cmb = combiner(aS,prime(aS,4));
//     for (int i=0; i<=3; i++) {
//         if(plToEnv[i]) {
//             res = res * prime(cmb,i);
//             res = res * delta(plToEnv[i],commonIndex(res,prime(cmb,i)));
//             if (dbg) {
//                 Print(plToEnv[i]);
//                 Print(res);
//             }
//         }
//     }

//     return res;
// }

// double EVBuilder::contract3Smpo2x2(MPO_3site const& mpo3s,
//     std::vector< std::pair<int,int> > siteSeq, bool dbg) const
// {

//     if(siteSeq.size() != 4) {
//         std::cout<<"EVBuilder::contract3Smpo2x2: siteSeq.size() !=
//         4"<<std::endl; exit(EXIT_FAILURE);
//     }

//     std::vector< std::string > tn(4);
//     std::vector<int> pl(8);

//     for (int i=0; i<4; i++) {
//         // shift to unit cell
//         auto pos = std::make_pair(
//             (siteSeq[i].first + cls.sizeM * std::abs(siteSeq[i].first)) %
//             cls.sizeM, (siteSeq[i].second + cls.sizeN *
//             std::abs(siteSeq[i].second)) % cls.sizeN
//         );

//         tn[i] = cls.cToS.at(pos);

//         int j = (i + 4 - 1) % 4;
//         auto sv = std::make_pair(siteSeq[i].first - siteSeq[j].first,
//             siteSeq[i].second - siteSeq[j].second);

//         int plI0 = 2*j+1;
//         int plI1 = 2*i;
//         if (sv == std::make_pair(1,0)) {
//             pl[plI0] = 2;
//             pl[plI1] = 0;
//         } else if (sv == std::make_pair(-1,0)) {
//             pl[plI0] = 0;
//             pl[plI1] = 2;
//         } else if (sv == std::make_pair(0,1)) {
//             pl[plI0] = 3;
//             pl[plI1] = 1;
//         } else {
//             pl[plI0] = 1;
//             pl[plI1] = 3;
//         }
//     }

//     return contract3Smpo2x2(mpo3s, tn, pl, dbg);
// }

// double EVBuilder::contract3Smpo2x2(MPO_3site const& mpo3s,
//     std::vector<std::string> tn, std::vector<int> pl, bool dbg) const
// {
//     int dbgLvl = 3;

//     std::chrono::steady_clock::time_point t_begin_int, t_end_int;

//     if (dbg) {
//         std::cout<<"GATE: ";
//         for(int i=0; i<=3; i++) {
//             std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->";
//         }
//         std::cout<< std::endl;
//     }

//     if(dbg && (dbgLvl >= 2)) {
//         std::cout<< mpo3s;
//         PrintData(mpo3s.H1);
//         PrintData(mpo3s.H2);
//         PrintData(mpo3s.H3);
//     }

//     // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS
//     ************************

//     // map MPOs
//     ITensor dummyMPO = ITensor();
//     std::array<const ITensor *, 4> mpo({&mpo3s.H1, &mpo3s.H2, &mpo3s.H3,
//     &dummyMPO});

//     // find integer identifier of on-site tensors within CtmEnv
//     std::vector<int> si;
//     for (int i=0; i<=3; i++) {
//         si.push_back(std::distance(cls.siteIds.begin(),
//                 std::find(std::begin(cls.siteIds),
//                     std::end(cls.siteIds), tn[i])));
//     }
//     if(dbg) {
//         std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
//         for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] <<
//         std::endl; }
//     }

//     // read off auxiliary and physical indices of the cluster sites
//     std::array<Index, 4> aux({
//         noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
//         noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
//         noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
//         noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

//     std::array<Index, 4> auxRT({ aux[0], aux[1], aux[1], aux[2] });
//     std::array<int, 4> plRT({ pl[1], pl[2], pl[3], pl[4] });

//     std::array<Index, 4> phys({
//         noprime(findtype(cls.sites.at(tn[0]), PHYS)),
//         noprime(findtype(cls.sites.at(tn[1]), PHYS)),
//         noprime(findtype(cls.sites.at(tn[2]), PHYS)),
//         noprime(findtype(cls.sites.at(tn[3]), PHYS)) });

//     std::array<Index, 3> opPI({
//         noprime(findtype(mpo3s.H1, PHYS)),
//         noprime(findtype(mpo3s.H2, PHYS)),
//         noprime(findtype(mpo3s.H3, PHYS)) });

//     // prepare map from on-site tensor aux-indices to half row/column T
//     // environment tensors
//     std::array<const std::vector<ITensor> * const, 4> iToT(
//         {&cd_f.T_L, &cd_f.T_U, &cd_f.T_R ,&cd_f.T_D});

//     // prepare map from on-site tensor aux-indices pair to half corner T-C-T
//     // environment tensors
//     const std::map<int, const std::vector<ITensor> * const > iToC(
//         {{23, &cd_f.C_LU}, {32, &cd_f.C_LU},
//         {21, &cd_f.C_LD}, {12, &cd_f.C_LD},
//         {3, &cd_f.C_RU}, {30, &cd_f.C_RU},
//         {1, &cd_f.C_RD}, {10, &cd_f.C_RD}});

//     // for every on-site tensor point from primeLevel(index) to ENV index
//     // eg. I_XH or I_XV (with appropriate prime level).
//     std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

//     // Find for site 0 through 3 which are connected to ENV
//     std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of
//     on-site tensor

//     Index iQA, iQD, iQB;
//     ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
//     ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
//     ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);

//     ITensor eRE;
//     ITensor deltaBra, deltaKet;

//     {
//         // precompute 4 (proto)corners of 2x2 environment
//         std::vector<ITensor> pc(4);
//         for (int s=0; s<=3; s++) {
//             // aux-indices connected to sites
//             std::vector<int> connected({pl[s*2], pl[s*2+1]});
//             // set_difference gives aux-indices connected to ENV
//             std::sort(connected.begin(), connected.end());
//             std::vector<int> tmp_iToE;
//             std::set_difference(plOfSite.begin(), plOfSite.end(),
//                 connected.begin(), connected.end(),
//                 std::inserter(tmp_iToE, tmp_iToE.begin()));
//             tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV
//             tensor if(dbg) {
//                 std::cout <<"primeLevels (pl) of indices connected to ENV -
//                 site: "
//                     << tn[s] << std::endl;
//                 std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<<
//                 tmp_iToE[2] << std::endl;
//             }

//             // Assign indices by which site is connected to ENV
//             if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
//                 iToE[s][tmp_iToE[0]] = findtype(
//                 (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
//                 iToE[s][tmp_iToE[1]] = findtype(
//                 (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
//             } else {
//                 iToE[s][tmp_iToE[0]] = findtype(
//                 (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
//                 iToE[s][tmp_iToE[1]] = findtype(
//                 (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
//             }

//             pc[s] =
//             (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
//                 (*iToT.at(tmp_iToE[1]))[si[s]];
//             if(dbg) Print(pc[s]);
//             // set primeLevel of ENV indices between T's to 0 to be ready for
//             contraction pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
//         }
//         if(dbg) {
//             for(int i=0; i<=3; i++) {
//                 std::cout <<"Site: "<< tn[i] <<" ";
//                 for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<"
//                 "; std::cout << std::endl;
//             }
//         }
//         // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE
//         *******************

//         // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT
//         *************************** t_begin_int =
//         std::chrono::steady_clock::now();

//         // C  D
//         //    |
//         // A--B
//         // ITensor eRE;
//         // ITensor deltaBra, deltaKet;

//         // Decompose A tensor on which the gate is applied
//         //ITensor QA, tempSA, eA(prime(aux[0],pl[1]), phys[0]);
//         ITensor tempSA;
//         svd(cls.sites.at(tn[0]), eA, tempSA, QA);
//         iQA = Index("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
//         eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
//         QA *= delta(commonIndex(QA,tempSA), iQA);

//         // Prepare corner of A
//         ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
//         if(dbg && (dbgLvl >=3)) Print(tempC);

//         deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
//         deltaBra = prime(deltaKet,4);
//         tempC = (tempC * deltaBra) * deltaKet;
//         if(dbg && (dbgLvl >=3)) Print(tempC);

//         eRE = tempC;

//         // Prepare corner of C
//         tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >=
//         3))); if(dbg && (dbgLvl >=3)) Print(tempC);

//         deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
//         deltaBra = prime(deltaKet,4);
//         tempC = (tempC * deltaBra) * deltaKet;
//         if(dbg && (dbgLvl >=3)) Print(tempC);

//         eRE = eRE * tempC;

//         // Decompose D tensor on which the gate is applied
//         //ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
//         ITensor tempSD;
//         svd(cls.sites.at(tn[2]), eD, tempSD, QD);
//         iQD = Index("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
//         eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
//         QD *= delta(commonIndex(QD,tempSD), iQD);

//         // Prepare corner of D
//         tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
//         if(dbg && (dbgLvl >=3)) Print(tempC);

//         eRE = eRE * tempC;

//         // Decompose B tensor on which the gate is applied
//         //ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]),
//         phys[1]); ITensor tempSB; svd(cls.sites.at(tn[1]), eB, tempSB, QB);
//         iQB = Index("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
//         eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
//         QB *= delta(commonIndex(QB,tempSB), iQB);

//         tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
//         if(dbg && (dbgLvl >=3)) Print(tempC);

//         eRE = eRE * tempC;

//         t_end_int = std::chrono::steady_clock::now();
//         if (dbg) std::cout<<"Constructed reduced Env - T: "<<
//             std::chrono::duration_cast<std::chrono::microseconds>(t_end_int -
//             t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
//         if(dbg && (dbgLvl >=3)) Print(eRE);
//         // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE
//         **********************
//     }

//     // ***** FORM "PROTO" ENVIRONMENTS FOR M and K
//     ***************************** t_begin_int =
//     std::chrono::steady_clock::now();

//     ITensor protoK = (eRE * eA) * delta(prime(aux[0],pl[1]),
//     prime(aux[1],pl[2])); protoK = (protoK * eB) * delta(prime(aux[1],pl[3]),
//     prime(aux[2],pl[4])); protoK = (protoK * eD); if(dbg && (dbgLvl >=3))
//     Print(protoK);

//     auto protoKId = protoK;
//     protoK = ( protoK * delta(opPI[0],phys[0]) ) * mpo3s.H1;
//     protoK = ( protoK * delta(opPI[1],phys[1]) ) * mpo3s.H2;
//     protoK = ( protoK * delta(opPI[2],phys[2]) ) * mpo3s.H3;
//     protoK = (( protoK * delta(prime(opPI[0],1),phys[0]) ) *
//         delta(prime(opPI[1],1),phys[1]) ) * delta(prime(opPI[2],1),phys[2]);
//     if(dbg && (dbgLvl >=3)) Print(protoK);

//     // PROTOK - VARIANT 1
//     protoKId *= conj(eA).prime(AUXLINK,4);
//     protoKId *= delta(prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4));
//     protoK = protoK * conj(eA).prime(AUXLINK,4);
//     protoK *= delta(prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4));
//     if(dbg && (dbgLvl >=3)) Print(protoK);

//     protoKId *= conj(eB).prime(AUXLINK,4);
//     protoKId *= delta(prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4));
//     protoK = protoK * conj(eB).prime(AUXLINK,4);
//     protoK *= delta(prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4));
//     if(dbg && (dbgLvl >=3)) Print(protoK);

//     protoKId *= conj(eD).prime(AUXLINK,4);
//     protoK = protoK * conj(eD).prime(AUXLINK,4);
//     if(dbg && (dbgLvl >=3)) Print(protoK);

//     if (rank(protoK) > 0) std::cout<<"ERROR - protoK not a
//     scalar"<<std::endl; if (rank(protoKId) > 0) std::cout<<"ERROR - protoKId
//     not a scalar"<<std::endl; std::complex<double> ev   = sumelsC(protoK);
//     std::complex<double> evId = sumelsC(protoKId);
//     if (isComplex(protoK)) {
//         std::cout<<"Expectation value is Complex: imag(ev)="<< ev.imag() <<
//         std::endl;
//     }
//     if (isComplex(protoKId)) {
//         std::cout<<"evId is Complex: imag(evId)="<< evId.imag() << std::endl;
//     }

//     t_end_int = std::chrono::steady_clock::now();
//     if (dbg) {
//         std::cout<<"ev: "<< ev <<" evId: "<< evId << std::endl;
//         std::cout<<"EVBuilder::contract3Smpo2x2 - T: "<<
//         std::chrono::duration_cast<std::chrono::microseconds>(t_end_int -
//         t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
//     }

//     return ev.real()/evId.real();
// }

// std::complex<double> ExpValBuilder::expVal_1sO1sO_V(int dist,
//         itensor::ITensor const& op1, itensor::ITensor const& op2)
// {

//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);
//

//     auto U   = env.C_lu*env.T_u*env.C_ru;
//     U = U*env.T_l*op1*env.T_r;
//     auto UId = env.C_lu*env.T_u*env.C_ru;
//     UId = UId*env.T_l*X*env.T_r;
//     //DEBUG Print(L);

//     /*
//      * Contract U with "dist" copies of a row
//      *
//      * I(Tl)   I(Xv)   I(Tr)
//      *  |       |       |
//      * |T|-----|X|-----|T|
//      *  |       |       |
//      * I(Tl)'  I(Xv)'  I(Tr)'
//      *
//      */
//     //DEBUG std::cout << "Inserting "<< dist << " T_l--X--T_r row"
//     //DEBUG          << "\n";
//     for(int i=0;i<dist;i++) {
//         U.noprime();
//         UId.noprime();
//         U = U*env.T_l*X*env.T_r;
//         UId = UId*env.T_l*X*env.T_r;
//     }

//     /*
//      * Construct the "down" part tensor D
//      *
//      * I(Tl)  I(Xv)  I(Tr)
//      *  |      |       |   ==>
//      * |T|----|O2|----|T|  ==>   I(Tl) I(Xv)  I(Tr)
//      *  |      |       |   ==>    _|____|_____|_
//      * |C|----|T |----|C|        |_____D________|
//      *
//      */
//     auto D = env.C_ld*env.T_d*env.C_rd;
//     D = D*env.T_l*op2*env.T_r;
//     auto DId = env.C_ld*env.T_d*env.C_rd;
//     DId = DId*env.T_l*X*env.T_r;
//     //DEBUG Print(R);

//     U.noprime();
//     UId.noprime();
//     // Contract (L*col^dist)*R
//     auto ccBare = U*D;
//     auto ccNorm = UId*DId;
//     //DEBUG PrintData(ccBare);
//     //DEBUG PrintData(ccNorm);

//     return sumelsC(ccBare)/sumelsC(ccNorm);
// }

// std::complex<double> ExpValBuilder::expVal_2sOV2sOV_H(int dist,
//             ExpValBuilder::Mpo2S const& op1,
//             ExpValBuilder::Mpo2S const& op2)
// {
//     //std::cout << "DEBUG ##### expVal 2sOV2sOV_H #####" << "\n";

//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);

//     /*
//      * Construct the "left" part tensor L
//      *  _
//      * |C|--I(T_u)
//      *  |
//      * |T|--I(XH)
//      *  |
//      * |T|--I(XH)''
//      *  |
//      * |C|--I(T_d)
//      *
//      *
//      */
//     auto LA  = env.C_lu*env.T_l;
//     LA = LA.prime(prime(env.i_Tl,1),-1)*prime(env.T_l, env.i_Xh, 2)*env.C_ld;
//     auto LId = LA;

//     /*
//      * Construct the "left" part tensor L
//      *  _    __               _    __                 __
//      * |C|--|T |--I(T_u)'    |C|--|T |--I(T_u)'      |  |--I(T_u)
//      *  |    |                |    |                 |  |
//      * |T|--|O1B|-I(XH)'  => |T|--|O1B|-I(XH)'   ==> |  |--I(Xh)
//      *  |    v''          =>  |    |             ==> |L |
//      * |T|-h''            => |T|--|O1A|-I(XH)'3  ==> |  |--I(Xh)''
//      *  |                     |    |                 |  |
//      * |C|--I(T_d)           |C|--|T |--I(T_d)'      |__|--I(T_d)
//      *
//      */
//     LA = LA*env.T_u*prime(op1.opB, prime(env.i_Xv,1), 1);
//     LA = LA*prime(op1.opA, env.i_Xh,prime(env.i_Xh,1),env.i_Xv,
//         prime(env.i_Xv,1), 2)*prime(env.T_d, prime(env.i_Xv,1), 2);
//     LA.prime(-1);

//     LId = LId*env.T_u*prime(X, prime(env.i_Xv,1), 1);
//     LId = LId*prime(X,2)*prime(env.T_d, prime(env.i_Xv,1), 2);
//     LId.prime(-1);

//     for (int i=0; i<dist; i++) {
//         LA = LA*env.T_u*prime(X, prime(env.i_Xv,1), 1);
//         LA = LA*prime(X,2)*prime(env.T_d, prime(env.i_Xv,1), 2);
//         LA.prime(-1);

//         LId = LId*env.T_u*prime(X, prime(env.i_Xv,1), 1);
//         LId = LId*prime(X,2)*prime(env.T_d, prime(env.i_Xv,1), 2);
//         LId.prime(-1);
//     }

//     /*
//      * Construct the "left" part tensor L
//      *           _
//      * I(T_u)'--|C|
//      *           |
//      *  I(XH)'--|T|
//      *           |
//      * I(XH)'3--|T|
//      *           |
//      * I(T_d)'--|C|
//      *
//      */
//     auto RB = env.C_ru*env.T_r;
//     RB = RB.prime(prime(env.i_Tr,1),-1)*prime(env.T_r, prime(env.i_Xh,1), 2)
//         *env.C_rd;
//     auto RId = RB;

//     /*
//      * Construct the "right" part tensor R
//      *         __    _               __     _                __
//      * I(T_u)-|T |--|C|     I(T_u)--|T |---|C|      I(T_u)--|  |
//      *         |     |               |      |               |  |
//      * I(XH)-|O2B|--|T|  =>  I(XH)--|O2B|--|T|  ==>  I(XH)--|  |
//      *         v''   |   =>          |      |   ==>         |R |
//      *          h'2-|T|  => I(XH)''-|O1A|--|T|  ==> I(XH)''-|  |
//      *               |               |      |               |  |
//      *     I(T_d)'--|C|     I(T_d)--|T |---|C|      I(T_d)--|__|
//      *
//      */
//     RB = RB*env.T_u*prime(op2.opB, prime(env.i_Xv,1), 1);
//     RB = RB*prime(op2.opA, env.i_Xh,prime(env.i_Xh,1),env.i_Xv,
//         prime(env.i_Xv,1), 2)*prime(env.T_d, prime(env.i_Xv,1), 2);

//     RId = RId*env.T_u*prime(X, prime(env.i_Xv,1), 1);
//     RId = RId*prime(X,2)*prime(env.T_d, prime(env.i_Xv,1), 2);

//     // Contract L*R
//     auto ccBare = LA*RB;
//     auto ccNorm = LId*RId;
//     //DEBUG PrintData(ccBare);
//     //DEBUG PrintData(ccNorm);

//     return sumelsC(ccBare)/sumelsC(ccNorm);
// }

// std::complex<double> ExpValBuilder::expVal_2sOH2sOH_H(int dist,
//             ExpValBuilder::Mpo2S const& op1,
//             ExpValBuilder::Mpo2S const& op2)
// {
//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);
//     /*
//      * Construct the "left" part tensor L
//      *  _    __    __                 __
//      * |C|--|T |--|T |--I(T_u)'      |  |--I(T_u)'
//      *  |    |     |             ==> |  |
//      * |T|--|O1A|=|O1B|--I(XH)'  ==> |L |--I(Xh)'
//      *  |    |     |             ==> |  |
//      * |C|--|T |--|T |--I(T_d)'      |__|--I(T_d)'
//      *
//      */
//     auto L   = env.C_lu*env.T_l*env.C_ld;
//     L = L*env.T_u*op1.opA*env.T_d;
//     L.noprime();
//     L = L*env.T_u*op1.opB*env.T_d;

//     auto LId = env.C_lu*env.T_l*env.C_ld;
//     LId = LId*env.T_u*X*env.T_d;
//     LId.noprime();
//     LId = LId*env.T_u*X*env.T_d;
//     //DEBUG Print(L);

//     /*
//      * Contract L with "dist" copies of a column
//      *
//      * I(T_u)--|T|--I(T_u)'
//      *          |
//      *  I(Xh)--|X|--I(Xh)'
//      *          |
//      * I(T_d)--|T|--I(T_d)'
//      *
//      */
//     //DEBUG std::cout << "Inserting "<< dist <<" T_u--X--T_d column"
//     //DEBUG         << "\n";
//     for(int i=0;i<dist;i++) {
//         L.noprime();
//         LId.noprime();
//         L = L*env.T_u*X*env.T_d;
//         LId = LId*env.T_u*X*env.T_d;
//     }

//     /*
//      * Construct the "right" part tensor R
//      *          __    __     _                 __
//      * I(T_u)--|T |--|T |---|C|       I(T_u)--|  |
//      *          |     |      |   ==>          |  |
//      *  I(Xh)--|O2A|=|O2B|--|T|  ==>   I(Xh)--|R |
//      *          |     |      |   ==>          |  |
//      * I(T_d)--|T |--|T |---|C|       I(T_d)--|__|
//      *
//      * for "dist" even, otherwise A and B are exchanged
//      *
//      */
//     auto R = env.C_ru*env.T_r*env.C_rd;
//     if (dist % 2 == 0) {
//         R = R*env.T_u*op2.opB*env.T_d;
//         R.prime();
//         R = R*env.T_u*op2.opA*env.T_d;
//     } else {
//         R = R*env.T_u*op2.opA*env.T_d;
//         R.prime();
//         R = R*env.T_u*op2.opB*env.T_d;
//     }

//     auto RId = env.C_ru*env.T_r*env.C_rd;
//     RId = RId*env.T_u*X*env.T_d;
//     RId.prime();
//     RId = RId*env.T_u*X*env.T_d;
//     //Print(R);

//     L.noprime();
//     LId.noprime();
//     // Contract (L*col^dist)*R
//     auto ccBare = L*R;
//     auto ccNorm = LId*RId;
//     //DEBUG PrintData(ccBare);
//     //DEBUG PrintData(ccNorm);

//     return sumelsC(ccBare)/sumelsC(ccNorm);
// }

MPO_3site EVBuilder::get3Smpo(std::string mpo3s, bool DBG) const {
  int physDim = 2;
  Index s1 = Index("S1", physDim, PHYS);
  Index s2 = Index("S2", physDim, PHYS);
  Index s3 = Index("S3", physDim, PHYS);
  Index s1p = prime(s1);
  Index s2p = prime(s2);
  Index s3p = prime(s3);

  ITensor h123 = ITensor(s1, s2, s3, s1p, s2p, s3p);
  if (mpo3s == "3SZ") {
    h123 = SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2) *
           SU2_getSpinOp(SU2_S_Z, s3);
  } else {
    std::cout << "Invalid MPO selection mpo3s: " << mpo3s << std::endl;
    exit(EXIT_FAILURE);
  }

  return symmMPO3Sdecomp(h123, s1, s2, s3);
}

std::pair<ITensor, ITensor> EVBuilder::get2SiteSpinOP(OP_2S op2s,
                                                      Index const& sA,
                                                      Index const& sB,
                                                      bool dbg) {
  /*
   * 2-site operator acts on 2 physical indices
   *
   *           A      B
   *   <bar|   s'     s'''
   *          _|______|_
   *         |____OP____|
   *           |      |
   *           s      s''  |ket>
   *
   */
  // Assume s0 is different then s2
  Index tmp_sB = sB;
  if (sA == sB) {
    if (dbg)
      std::cout << "On-site PHYS indices sA and sB are identitcal" << std::endl;
    tmp_sB = Index(sB.rawname() + "_copy", sB.m(), sB.type(), sB.primeLevel());
  }

  // auto s0 = findtype(TA.inds(), PHYS);
  auto s0 = sA;
  auto s1 = prime(s0, 1);
  // auto s2 = prime(findtype(TB.inds(), PHYS), 2);
  auto s2 = tmp_sB;
  auto s3 = prime(s2, 1);

  // check dimensions of phys indices on TA and TB
  if (s0.m() != s2.m()) {
    std::cout << "On-site tensors TA and TB have different dimension of"
              << " phys index" << std::endl;
    exit(EXIT_FAILURE);
  }
  int dimS = s0.m();

  auto Op = ITensor(s0, s1, s2, s3);
  switch (op2s) {
    case OP2S_Id: {  // Identity operator
      if (dbg)
        std::cout << ">>>>> 2) Constructing OP2S_Id <<<<<" << std::endl;
      for (int i = 1; i <= dimS; i++) {
        for (int j = 1; j <= dimS; j++) {
          Op.set(s0(i), s2(j), s1(i), s3(j), 1. + 0._i);
        }
      }
      break;
    }
    case OP2S_AKLT_S2_H: {  // H of AKLT-S2 on square lattice
      if (dbg)
        std::cout << ">>>>> 2) Constructing OP2S_AKLT-S2-H <<<<<" << std::endl;
      // Loop over <bra| indices
      int rS = dimS - 1;  // Label of SU(2) irrep in Dyknin notation
      int mbA, mbB, mkA, mkB;
      double hVal;
      for (int bA = 1; bA <= dimS; bA++) {
        for (int bB = 1; bB <= dimS; bB++) {
          // Loop over |ket> indices
          for (int kA = 1; kA <= dimS; kA++) {
            for (int kB = 1; kB <= dimS; kB++) {
              // Use Dynkin notation to specify irreps
              mbA = -(rS) + 2 * (bA - 1);
              mbB = -(rS) + 2 * (bB - 1);
              mkA = -(rS) + 2 * (kA - 1);
              mkB = -(rS) + 2 * (kB - 1);
              // Loop over possible values of m given by tensor product
              // of 2 spin (dimS-1) irreps (In Dynkin notation)
              hVal = 0.0;
              for (int m = -2 * (rS); m <= 2 * (rS); m = m + 2) {
                if ((mbA + mbB == m) && (mkA + mkB == m)) {
                  // DEBUG
                  if (dbg)
                    std::cout
                      << "<" << mbA << "," << mbB << "|" << m << "> x <" << m
                      << "|" << mkA << "," << mkB
                      << "> = " << SU2_getCG(rS, rS, 2 * rS, mbA, mbB, m)
                      << " x " << SU2_getCG(rS, rS, 2 * rS, mkA, mkB, m)
                      << std::endl;

                  hVal += SU2_getCG(rS, rS, 2 * rS, mbA, mbB, m) *
                          SU2_getCG(rS, rS, 2 * rS, mkA, mkB, m);
                }
              }
              if ((bA == kA) && (bB == kB)) {
                // add 2*Id(bA,kA;bB,kB) ==
                //    sqrt(2)*Id(bA,kA)(x)sqrt(2)*Id(bB,kB)
                Op.set(s0(kA), s2(kB), s1(bA), s3(bB), hVal + sqrt(2.0));
              } else {
                Op.set(s0(kA), s2(kB), s1(bA), s3(bB), hVal);
              }
            }
          }
        }
      }
      break;
    }
    case OP2S_SS: {
      // S^vec_i * S^vec_i+1 =
      // = s^z_i*s^z_i+1 + 1/2(s^+_i*s^-_i+1 + s^-_i*s^+_i+1)
      if (dbg)
        std::cout << ">>>>> 2) Constructing OP2S_SS <<<<<" << std::endl;

      Index sBra = Index("sBra", dimS);
      Index sKet = prime(sBra);
      ITensor Sz = getSpinOp(MPO_S_Z, sBra);
      ITensor Sp = getSpinOp(MPO_S_P, sBra);
      ITensor Sm = getSpinOp(MPO_S_M, sBra);

      double hVal;
      // Loop over <bra| indices
      for (int bA = 1; bA <= dimS; bA++) {
        for (int bB = 1; bB <= dimS; bB++) {
          // Loop over |ket> indices
          for (int kA = 1; kA <= dimS; kA++) {
            for (int kB = 1; kB <= dimS; kB++) {
              hVal =
                Sz.real(sBra(bA), sKet(kA)) * Sz.real(sBra(bB), sKet(kB)) +
                0.5 *
                  (Sp.real(sBra(bA), sKet(kA)) * Sm.real(sBra(bB), sKet(kB)) +
                   Sm.real(sBra(bA), sKet(kA)) * Sp.real(sBra(bB), sKet(kB)));

              Op.set(s0(kA), s2(kB), s1(bA), s3(bB), hVal);
            }
          }
        }
      }
      break;
    }
    case OP2S_SZSZ: {
      if (dbg)
        std::cout << ">>>>> 2) Constructing OP2S_SZSZ <<<<<" << std::endl;

      Index sBra = Index("sBra", dimS);
      Index sKet = prime(sBra);
      ITensor Sz = getSpinOp(MPO_S_Z, sBra);

      double hVal;
      // Loop over <bra| indices
      for (int bA = 1; bA <= dimS; bA++) {
        for (int bB = 1; bB <= dimS; bB++) {
          // Loop over |ket> indices
          for (int kA = 1; kA <= dimS; kA++) {
            for (int kB = 1; kB <= dimS; kB++) {
              hVal = Sz.real(sBra(bA), sKet(kA)) * Sz.real(sBra(bB), sKet(kB));

              Op.set(s0(kA), s2(kB), s1(bA), s3(bB), hVal);
            }
          }
        }
      }

      break;
    }
    default: {
      if (dbg)
        std::cout << "Invalid OP_2S selection" << std::endl;
      exit(EXIT_FAILURE);
      break;
    }
  }

  // Perform SVD
  /*         __
   * I(s)---|  |--I(s)''    =>
   *        |OP|            =>
   * I(s)'--|__|--I(s)'''   =>
   *            ___                      ___
   * => I(s)---|   |         _          |   |--I(s)''
   * =>        |OpA|--I(o)--|S|--I(o)'--|OpB|
   * => I(s)'--|___|                    |___|--I(s)'''
   *
   */
  auto OpA = ITensor(s0, s1);
  ITensor OpB, S;

  if (dbg)
    std::cout << ">>>>> 3) Performing SVD OP2S -> OpA * S * OpB <<<<<"
              << std::endl;
  svd(Op, OpA, S, OpB, {"Cutoff", 1.0e-16});

  if (dbg) {
    Print(OpA);
    PrintData(S);
    Print(OpB);
  }

  // create a lambda function
  // which returns the square of its argument
  auto sqrt_T = [](Real r) { return sqrt(r); };
  S.apply(sqrt_T);

  // Absorb singular values (symmetrically) into OpA, OpB
  OpA = (OpA * S) * delta(commonIndex(S, OpB), commonIndex(OpA, S));
  OpB = S * OpB;

  if (sA == sB) {
    OpB *= delta(tmp_sB, sB);
    OpB *= delta(tmp_sB, sB).prime(1);
  }

  if (dbg) {
    std::cout << ">>>>> 4) Absorbing sqrt(S) to both OpA and OpB <<<<<"
              << std::endl;
    PrintData(OpA);
    PrintData(OpB);
  }

  return std::make_pair(OpA, OpB);
}

// TODO use getSpinOp defined in pi-peps/su2.h to get spin operator
ITensor EVBuilder::getSpinOp(MPO_1S mpo, Index const& s, bool DBG) {
  SU2O su2o;
  switch (mpo) {
    case MPO_Id: {
      su2o = SU2_Id;
      break;
    }
    case MPO_S_Z: {
      su2o = SU2_S_Z;
      break;
    }
    case MPO_S_Z2: {
      su2o = SU2_S_Z2;
      break;
    }
    case MPO_S_P: {
      su2o = SU2_S_P;
      break;
    }
    case MPO_S_M: {
      su2o = SU2_S_M;
      break;
    }
    default: {
      std::cout << "Invalid MPO selection" << std::endl;
      exit(EXIT_FAILURE);
      break;
    }
  }

  return SU2_getSpinOp(su2o, s, DBG);
}

std::ostream& EVBuilder::print(std::ostream& s) const {
  s << "ExpValBuilder(" << std::endl;
  s << "name: " << name << std::endl
    << "cluster_type: " << p_cluster->cluster_type << std::endl
    << ")" << std::endl;
  return s;
}

std::ostream& operator<<(std::ostream& s, EVBuilder const& ev) {
  return ev.print(s);
}