#include "pi-peps/config.h"
#include "pi-peps/cluster-ev-builder.h"

// namespace itensor {
using namespace itensor;

std::vector<std::complex<double>> EVBuilder::corrf_SS(Vertex const& v1,
                                                      CtmEnv::DIRECTION dir,
                                                      int dist,
                                                      std::vector<double> coefs,
                                                      bool DBG) const {
  std::vector<std::complex<double>> szsz, spsm, smsp;
  if (dir == CtmEnv::DIRECTION::RIGHT) {
    szsz = expVal_1sO1sO_H(
      v1, dist, std::make_pair(EVBuilder::MPO_S_Z, EVBuilder::MPO_S_Z), DBG);
    spsm = expVal_1sO1sO_H(
      v1, dist, std::make_pair(EVBuilder::MPO_S_P, EVBuilder::MPO_S_M), DBG);
    smsp = expVal_1sO1sO_H(
      v1, dist, std::make_pair(EVBuilder::MPO_S_M, EVBuilder::MPO_S_P), DBG);
  } else if (dir == CtmEnv::DIRECTION::DOWN) {
    szsz = expVal_1sO1sO_V(
      v1, dist, std::make_pair(EVBuilder::MPO_S_Z, EVBuilder::MPO_S_Z), DBG);
    spsm = expVal_1sO1sO_V(
      v1, dist, std::make_pair(EVBuilder::MPO_S_P, EVBuilder::MPO_S_M), DBG);
    smsp = expVal_1sO1sO_V(
      v1, dist, std::make_pair(EVBuilder::MPO_S_M, EVBuilder::MPO_S_P), DBG);
  }

  for (int i = 0; i < dist; i++)
    szsz[i] = coefs[2] * szsz[i] + 0.5 * (spsm[i] + smsp[i]);

  return szsz;
}

// std::vector< std::complex<double> > EVBuilder::corrf_SS(Vertex const& v1,
//                 CtmEnv::DIRECTION dir,
//                 int dist, bool DBG) const {
// 	return corrf_SS(v1, dir, dist, {1.0, 1.0, 1.0}, DBG);
// }

std::vector<std::complex<double>> EVBuilder::expVal_1sO1sO_H(
  Vertex const& v1,
  int dist,
  std::pair<MPO_1S, MPO_1S> const& Op,
  bool DBG) const {
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

  Vertex v = v1;
  ITensor tN, nN, LTdistR;

  std::vector<std::complex<double>> ccVal;

  // 1) ##### Construct LEFT edge
  // ########################################################
  /*
   * Construct the "left" part tensor L
   *  _          __
   * |C|--      |  |--I(T_u)
   *  |     ==> |  |
   * |T|--  ==> |L |--I(Xh)
   *  |     ==> |  |
   * |C|--'     |__|--I(T_d)
   *
   */
  if (DBG)
    std::cout << "C_LU[" << v << " => " << vToId(v) << "]" << std::endl;
  tN = p_ctmEnv->C_LU.at(vToId(v));
  if (DBG)
    std::cout << "T_L[" << v << " =>" << vToId(v) << "]" << std::endl;
  tN *= prime(p_ctmEnv->T_L.at(vToId(v)), AUXLINK, tmp_prime_offset * 0);
  if (DBG)
    std::cout << "C_LD[" << v << " => " << vToId(v) << "]" << std::endl;
  tN *= p_ctmEnv->C_LD.at(vToId(v));
  if (DBG)
    Print(tN);
  nN = tN;
  // ##### Construct LEFT edge DONE
  // ###################################################### 2) ##### APPEND
  // COLUMNS #############################################################
  /*
   * Contract L with "dist" copies of a column
   *
   * I(T_u)--|T|--I(T_u)'
   *          |
   *  I(Xh)--|X|--I(Xh)'
   *          |
   * I(T_d)--|T|--I(T_d)'
   *
   */
  for (int col = 0; col <= dist; col++) {
    v = v1 + col * Shift(1, 0);

    if (DBG)
      std::cout << "T_U[" << v << " => " << vToId(v) << "]" << std::endl;

    if (col > 0) {
      applyDeltaEdge(tN, v, DIRECTION::UP, DIRECTION::RIGHT);
      applyDeltaEdge(nN, v, DIRECTION::UP, DIRECTION::RIGHT);
    }
    tN *= p_ctmEnv->T_U.at(vToId(v));
    nN *= p_ctmEnv->T_U.at(vToId(v));

    if (DBG)
      Print(tN);

    if (DBG)
      std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

    Index tmp_down, tmp_right;
    tmp_down = p_cluster->AIc(v, DIRECTION::UP);

    if (col > 0) {
      //     applyDeltaSite(tN,v,DIRECTION::LEFT)
      tmp_right = p_cluster->AIc(v + Shift(-1, 0), DIRECTION::RIGHT);
    } else {
      tmp_right = p_cluster->AIc(v, DIRECTION::LEFT);
    };

    auto tmp_cmb0 =
      combiner(tmp_down, prime(tmp_down, p_cluster->BRAKET_OFFSET), tmp_right,
               prime(tmp_right, p_cluster->BRAKET_OFFSET));
    auto tmp_cmb1 = combiner(
      p_cluster->AIc(v, DIRECTION::UP),
      prime(p_cluster->AIc(v, DIRECTION::UP), p_cluster->BRAKET_OFFSET),
      p_cluster->AIc(v, DIRECTION::LEFT),
      prime(p_cluster->AIc(v, DIRECTION::LEFT), p_cluster->BRAKET_OFFSET));

    tN *= tmp_cmb0;
    nN *= tmp_cmb0;
    // TODO use delta instead of reindex
    tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
    nN = reindex(nN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));

    // find corresponding site in elementary cluster
    auto id = p_cluster->vertexToId(v);
    // find the index of site given its elem position within cluster
    auto pI = p_cluster->mphys.at(id);
    if (v == v1) {
      auto op = getSpinOp(Op.first, pI, DBG);

      auto siteop = p_cluster->sites.at(vToId(v)) * op;
      siteop.noprime(PHYS);
      siteop *= dag(p_cluster->sites.at(vToId(v)))
                  .prime(AUXLINK, p_cluster->BRAKET_OFFSET);

      tN *= siteop * tmp_cmb1;
    } else {
      auto op = getSpinOp(Op.second, pI, DBG);

      auto siteop = p_cluster->sites.at(vToId(v)) * op;
      siteop.noprime(PHYS);
      siteop *= dag(p_cluster->sites.at(vToId(v)))
                  .prime(AUXLINK, p_cluster->BRAKET_OFFSET);

      LTdistR = tN * (siteop * tmp_cmb1);
      // tN *= getSiteBraKet(v);
      tN *= getSiteBraKet(v) * tmp_cmb1;
    }
    nN *= getSiteBraKet(v) * tmp_cmb1;

    if (DBG)
      std::cout << "T_D[" << v << " => " << vToId(v) << "]" << std::endl;

    if (col > 0) {
      applyDeltaEdge(tN, v, DIRECTION::DOWN, DIRECTION::RIGHT);
      applyDeltaEdge(LTdistR, v, DIRECTION::DOWN, DIRECTION::RIGHT);
      applyDeltaEdge(nN, v, DIRECTION::DOWN, DIRECTION::RIGHT);
    }
    tN *= p_ctmEnv->T_D.at(vToId(v));
    nN *= p_ctmEnv->T_D.at(vToId(v));
    if (col > 0)
      LTdistR *= p_ctmEnv->T_D.at(vToId(v));

    if (DBG)
      std::cout << ">>>>> Appended col X= " << col << " <<<<<" << std::endl;
    if (DBG) {
      Print(tN);
      Print(nN);
      Print(LTdistR);
    }

    if (col > 0) {
      // TODO ? normalize
      // auto sqrtN = std::sqrt(norm(N));
      // N   = N / sqrtN;
      // NId = NId / sqrtN;

      // 2) ##### APPEND COLUMNS DONE
      // ######################################################## 3) #####
      // Construct RIGHT edge
      // #######################################################
      /*
       * Construct the "right" part tensor R
       *                _                 __
       *             --|C|       I(T_u)--|  |
       *                |   ==>          |  |
       *             --|T|  ==>   I(Xh)--|R |
       *                |   ==>          |  |
       *             --|C|       I(T_d)--|__|
       */
      if (DBG)
        std::cout << "C_RU[" << v << " => " << vToId(v) << "]" << std::endl;
      LTdistR *= p_ctmEnv->C_RU.at(vToId(v));
      auto LTdistRNN = nN * p_ctmEnv->C_RU.at(vToId(v));
      if (DBG)
        std::cout << "T_R[" << v << " =>" << vToId(v) << "]" << std::endl;
      LTdistR *= p_ctmEnv->T_R.at(vToId(v));
      LTdistRNN *= p_ctmEnv->T_R.at(vToId(v));
      if (DBG)
        std::cout << "C_RD[" << v << " => " << vToId(v) << "]" << std::endl;
      LTdistR *= p_ctmEnv->C_RD.at(vToId(v));
      LTdistRNN *= p_ctmEnv->C_RD.at(vToId(v));
      // 3) ##### Construct RIGHT edge DONE
      // ##################################################

      // Assign value
      ccVal.push_back(sumelsC(LTdistR) / sumelsC(LTdistRNN));
    }
  }

  return ccVal;
}

std::vector<std::complex<double>> EVBuilder::expVal_1sO1sO_V(
  Vertex const& v1,
  int dist,
  std::pair<MPO_1S, MPO_1S> const& Op,
  bool DBG) const {
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

  Vertex v = v1;
  ITensor tN, nN, UTdistD;
  std::vector<std::complex<double>> ccVal;

  // 1) ##### Construct UP edge
  // ########################################################

  //      * Construct the "Up" part tensor U
  //      *  _      __      _            ________________
  //      * |C|----|T |----|C|          |_______U________|
  //      *  |      |       |      ==>    |      |     |
  //                          ==>  I(Tl)' I(Xv)'  I(Tr)'

  if (DBG)
    std::cout << "C_LU[" << v << " => " << vToId(v) << "]" << std::endl;
  tN = p_ctmEnv->C_LU.at(vToId(v));
  if (DBG)
    std::cout << "T_U[" << v << " =>" << vToId(v) << "]" << std::endl;
  tN *= p_ctmEnv->T_U.at(vToId(v));
  if (DBG)
    std::cout << "C_RU[" << v << " => " << vToId(v) << "]" << std::endl;
  tN *= p_ctmEnv->C_RU.at(vToId(v));
  if (DBG)
    Print(tN);
  nN = tN;
  // ##### Construct LEFT edge DONE
  // ###################################################### 2) ##### APPEND
  // ROWS #############################################################
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
  for (int row = 0; row <= dist; row++) {
    v = v1 + row * Shift(0, 1);

    if (DBG)
      std::cout << "T_L[" << v << " => " << vToId(v) << "]" << std::endl;

    if (row > 0) {
      applyDeltaEdge(tN, v, DIRECTION::LEFT, DIRECTION::DOWN);
      applyDeltaEdge(nN, v, DIRECTION::LEFT, DIRECTION::DOWN);
    }
    tN *= p_ctmEnv->T_L.at(vToId(v));
    nN *= p_ctmEnv->T_L.at(vToId(v));

    if (DBG)
      Print(tN);

    if (DBG)
      std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

    Index tmp_down, tmp_right;

    if (row > 0) {
      //     applyDeltaSite(tN,v,DIRECTION::UP)
      tmp_down = p_cluster->AIc(v + Shift(0, -1), DIRECTION::DOWN);
    } else {
      tmp_down = p_cluster->AIc(v, DIRECTION::UP);
    }

    tmp_right = p_cluster->AIc(v, DIRECTION::LEFT);

    auto tmp_cmb0 =
      combiner(tmp_down, prime(tmp_down, p_cluster->BRAKET_OFFSET), tmp_right,
               prime(tmp_right, p_cluster->BRAKET_OFFSET));
    auto tmp_cmb1 = combiner(
      p_cluster->AIc(v, DIRECTION::UP),
      prime(p_cluster->AIc(v, DIRECTION::UP), p_cluster->BRAKET_OFFSET),
      p_cluster->AIc(v, DIRECTION::LEFT),
      prime(p_cluster->AIc(v, DIRECTION::LEFT), p_cluster->BRAKET_OFFSET));

    tN *= tmp_cmb0;
    nN *= tmp_cmb0;
    // TODO use delta instead of reindex
    tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
    nN = reindex(nN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));

    // find corresponding site in elementary cluster
    auto id = p_cluster->vertexToId(v);
    // find the index of site given its elem position within cluster
    auto pI = p_cluster->mphys.at(id);
    if (v == v1) {
      auto op = getSpinOp(Op.first, pI, DBG);

      auto siteop = p_cluster->sites.at(vToId(v)) * op;
      siteop.noprime(PHYS);
      siteop *= dag(p_cluster->sites.at(vToId(v)))
                  .prime(AUXLINK, p_cluster->BRAKET_OFFSET);

      tN *= siteop * tmp_cmb1;
    } else {
      auto op = getSpinOp(Op.second, pI, DBG);

      auto siteop = p_cluster->sites.at(vToId(v)) * op;
      siteop.noprime(PHYS);
      siteop *= dag(p_cluster->sites.at(vToId(v)))
                  .prime(AUXLINK, p_cluster->BRAKET_OFFSET);

      UTdistD = tN * (siteop * tmp_cmb1);
      // tN *= getSiteBraKet(v);
      tN *= getSiteBraKet(v) * tmp_cmb1;
    }
    nN *= getSiteBraKet(v) * tmp_cmb1;

    if (DBG)
      std::cout << "T_R[" << v << " => " << vToId(v) << "]" << std::endl;

    if (row > 0) {
      applyDeltaEdge(tN, v, DIRECTION::RIGHT, DIRECTION::DOWN);
      applyDeltaEdge(UTdistD, v, DIRECTION::RIGHT, DIRECTION::DOWN);
      applyDeltaEdge(nN, v, DIRECTION::RIGHT, DIRECTION::DOWN);
    }
    tN *= p_ctmEnv->T_R.at(vToId(v));
    nN *= p_ctmEnv->T_R.at(vToId(v));
    if (row > 0)
      UTdistD *= p_ctmEnv->T_R.at(vToId(v));

    if (DBG)
      std::cout << ">>>>> Appended row X= " << row << " <<<<<" << std::endl;
    if (DBG) {
      Print(tN);
      Print(nN);
      Print(UTdistD);
    }

    // 3) ##### Construct RIGHT edge
    // #######################################################
    //     /*
    //      * Construct the "down" part tensor D
    //      *
    //                                   I(Tl) I(Xv) I(Tr)
    //      *  |      |       |   ==>    _|____|_____|_
    //      * |C|----|T |----|C|        |_____D________|
    //      *
    //      */
    if (row > 0) {
      if (DBG)
        std::cout << "C_LD[" << v << " => " << vToId(v) << "]" << std::endl;
      UTdistD *= p_ctmEnv->C_LD.at(vToId(v));
      auto UTdistDNN = nN * p_ctmEnv->C_LD.at(vToId(v));
      if (DBG)
        std::cout << "T_D[" << v << " =>" << vToId(v) << "]" << std::endl;

      UTdistD *= p_ctmEnv->T_D.at(vToId(v));
      UTdistDNN *= p_ctmEnv->T_D.at(vToId(v));
      if (DBG)
        std::cout << "C_RD[" << v << " => " << vToId(v) << "]" << std::endl;
      UTdistD *= p_ctmEnv->C_RD.at(vToId(v));
      UTdistDNN *= p_ctmEnv->C_RD.at(vToId(v));
      // 3) ##### Construct RIGHT edge DONE
      // ##################################################

      // Assign value
      ccVal.push_back(sumelsC(UTdistD) / sumelsC(UTdistDNN));
    }
  }

  return ccVal;
}

// } // namespace itensor