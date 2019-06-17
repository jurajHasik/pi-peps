#include "pi-peps/transfer-op.h"

#ifdef PEPS_WITH_ARPACK

#  define pow_2(a) ((a) * (a))

namespace itensor {

  TransferOpVecProd::TransferOpVecProd(EVBuilder const& ev_,
                                       Vertex const& v_,
                                       CtmEnv::DIRECTION dir_)
    : dir(dir_), v_ref(v_), ev(ev_) {}

  void TransferOpVecProd::operator()(double const* const x,
                                     double* const y,
                                     bool DBG) {
    using DIRECTION = CtmEnv::DIRECTION;

    auto vToId = [this](Vertex const& v) {
      return ev.p_cluster->vertexToId(v);
    };

    auto getSiteBraKet = [this, &vToId](Vertex const& v) {
      return ev.p_cluster->sites.at(vToId(v)) *
             dag(ev.p_cluster->sites.at(vToId(v)))
               .prime(AUXLINK, ev.p_cluster->BRAKET_OFFSET);
    };

    auto applyDeltaEdge = [this](ITensor& t, Vertex const& v, DIRECTION edge,
                                 DIRECTION dir) {
      // (edge = LEFT or RIGHT => dir = UP or DOWN) or
      // (edge = UP or DOWN => dir = LEFT or RIGHT)
      if ((edge == dir) || ((edge + 2) % 4 == dir)) {
        std::cout << "[TransferOpVecProd::operator()] Invalid input: edge= "
                  << edge << " dir: " << dir << std::endl;
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

      t *= delta(ev.p_ctmEnv->tauxByVertex(edge, v + s, dir),
                 ev.p_ctmEnv->tauxByVertex(edge, v, (dir + 2) % 4));
    };

    // Depending on a direction, get the dimension of the TransferOp
    int N = pow_2(ev.p_cluster->AIc(v_ref, dir).m()) * pow_2(ev.p_ctmEnv->x);

    auto i = Index("i", N);
    auto ip = prime(i);

    // copy x
    std::vector<double> cpx(N);
    std::copy(x, x + N, cpx.data());

    // auto vecRefX = makeVecRef(cpx.data(),cpx.size());

    auto isX = IndexSet(i);
    auto tN = ITensor(isX, Dense<double>(std::move(cpx)));

    ITensor cmbX;
    if (dir == DIRECTION::RIGHT) {
      cmbX = combiner(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::UP, v_ref, DIRECTION::LEFT),
        ev.p_cluster->AIc(v_ref, DIRECTION::LEFT),
        prime(ev.p_cluster->AIc(v_ref, DIRECTION::LEFT),
              ev.p_cluster->BRAKET_OFFSET),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::DOWN, v_ref, DIRECTION::LEFT));
    } else if (dir == DIRECTION::DOWN) {
      cmbX = combiner(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::LEFT, v_ref, DIRECTION::UP),
        ev.p_cluster->AIc(v_ref, DIRECTION::UP),
        prime(ev.p_cluster->AIc(v_ref, DIRECTION::UP),
              ev.p_cluster->BRAKET_OFFSET),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::RIGHT, v_ref, DIRECTION::UP));
    } else {
      std::cout << "[TransferOpVecProd] Unsupported option: " << dir
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // TODO check if dimension of combindedIndex and i is the same
    tN *= delta(combinedIndex(cmbX), i);
    tN *= cmbX;

    // Apply lY rows or lX columns depending on chosen direction
    auto v = v_ref;
    if (dir == DIRECTION::RIGHT) {
      for (int col = 0; col < ev.p_cluster->lX; col++) {
        v = v_ref + col * Shift(1, 0);

        if (DBG)
          std::cout << "T_U[" << v << " => " << vToId(v) << "]" << std::endl;

        if (col > 0)
          applyDeltaEdge(tN, v, DIRECTION::UP, DIRECTION::RIGHT);
        tN *= ev.p_ctmEnv->T_U.at(vToId(v));

        if (DBG)
          Print(tN);
        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        Index tmp_down, tmp_right;

        tmp_down = ev.p_cluster->AIc(v, DIRECTION::UP);
        if (col > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::LEFT)
          tmp_right = ev.p_cluster->AIc(v + Shift(-1, 0), DIRECTION::RIGHT);
        } else {
          tmp_right = ev.p_cluster->AIc(v, DIRECTION::LEFT);
        };

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, ev.p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, ev.p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 = combiner(ev.p_cluster->AIc(v, DIRECTION::UP),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::UP),
                                       ev.p_cluster->BRAKET_OFFSET),
                                 ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                       ev.p_cluster->BRAKET_OFFSET));

        tN *= tmp_cmb0;
        // TODO use delta instead of reindex
        tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        tN *= getSiteBraKet(v) * tmp_cmb1;

        if (DBG)
          std::cout << "T_D[" << v << " => " << vToId(v) << "]" << std::endl;

        if (col > 0)
          applyDeltaEdge(tN, v, DIRECTION::DOWN, DIRECTION::RIGHT);
        tN *= ev.p_ctmEnv->T_D.at(vToId(v));

        if (DBG)
          std::cout << ">>>>> Appended col X= " << col << " <<<<<" << std::endl;
        if (DBG)
          Print(tN);
      }
    } else if (dir == DIRECTION::DOWN) {
      for (int row = 0; row < ev.p_cluster->lY; row++) {
        v = v_ref + row * Shift(0, 1);

        if (DBG)
          std::cout << "T_L[" << v << " => " << vToId(v) << "]" << std::endl;

        if (row > 0)
          applyDeltaEdge(tN, v, DIRECTION::LEFT, DIRECTION::DOWN);
        tN *= ev.p_ctmEnv->T_L.at(vToId(v));

        if (DBG)
          Print(tN);
        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        Index tmp_down, tmp_right;

        if (row > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::UP)
          tmp_down = ev.p_cluster->AIc(v + Shift(0, -1), DIRECTION::DOWN);
        } else {
          tmp_down = ev.p_cluster->AIc(v, DIRECTION::UP);
        }
        tmp_right = ev.p_cluster->AIc(v, DIRECTION::LEFT);

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, ev.p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, ev.p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 = combiner(ev.p_cluster->AIc(v, DIRECTION::UP),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::UP),
                                       ev.p_cluster->BRAKET_OFFSET),
                                 ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                       ev.p_cluster->BRAKET_OFFSET));

        tN *= tmp_cmb0;
        // TODO use delta instead of reindex
        tN = reindex(tN, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        tN *= getSiteBraKet(v) * tmp_cmb1;

        if (DBG)
          std::cout << "T_R[" << v << " => " << vToId(v) << "]" << std::endl;

        if (row > 0)
          applyDeltaEdge(tN, v, DIRECTION::RIGHT, DIRECTION::DOWN);
        tN *= ev.p_ctmEnv->T_R.at(vToId(v));

        if (DBG)
          std::cout << ">>>>> Appended row X= " << row << " <<<<<" << std::endl;
        if (DBG)
          Print(tN);
      }
    }

    if (dir == DIRECTION::RIGHT) {
      cmbX = combiner(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::UP, v, DIRECTION::RIGHT),
        ev.p_cluster->AIc(v, DIRECTION::RIGHT),
        prime(ev.p_cluster->AIc(v, DIRECTION::RIGHT),
              ev.p_cluster->BRAKET_OFFSET),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::DOWN, v, DIRECTION::RIGHT));
    } else if (dir == DIRECTION::DOWN) {
      cmbX = combiner(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::LEFT, v, DIRECTION::DOWN),
        ev.p_cluster->AIc(v, DIRECTION::DOWN),
        prime(ev.p_cluster->AIc(v, DIRECTION::DOWN),
              ev.p_cluster->BRAKET_OFFSET),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::RIGHT, v, DIRECTION::DOWN));
    }

    tN *= cmbX;
    tN *= delta(combinedIndex(cmbX), i);

    // TODO possibly redundant
    tN.scaleTo(1.0);

    auto extractReal = [](Dense<Real> const& d) { return d.store; };

    auto xData = applyFunc(extractReal, tN.store());
    std::copy(xData.data(), xData.data() + N, y);
  }

  void analyzeTransferMatrix(EVBuilder const& ev,
                             Vertex const& v,
                             CtmEnv::DIRECTION dir,
                             int num_eigs,
                             std::string alg_type) {
    if (alg_type == "ARPACK") {
      TransferOpVecProd tvp(ev, v, dir);

      int N = pow_2(ev.p_cluster->AIc(v, dir).m()) * pow_2(ev.p_ctmEnv->x);
      ARDNS<TransferOpVecProd> ardns(tvp);

      std::vector<std::complex<double>> eigv;
      std::vector<double> V;
      ardns.real_nonsymm(N, num_eigs, 100, 0.0, N * 10, eigv, V);

      // sort
      std::sort(
        eigv.begin(), eigv.end(),
        [](std::complex<double> const& a, std::complex<double> const& b) {
          return std::abs(a) > std::abs(b);
        });
      for (auto const& val : eigv) {
        std::cout << val / std::abs(eigv[0]) << std::endl;
      }

    } else {
      std::cout << "[EVBuilder::analyzeTransferMatrix] Unsupported option: "
                << alg_type << std::endl;
    }
  }

}  // namespace itensor

#endif

namespace itensor {

  TransferOpVecProd_itensor::TransferOpVecProd_itensor(EVBuilder const& ev_,
                                                       Vertex const& v_,
                                                       CtmEnv::DIRECTION dir_)
    : dir(dir_), v_ref(v_), ev(ev_) {}

  void TransferOpVecProd_itensor::operator()(ITensor& bT, bool DBG) {
    using DIRECTION = CtmEnv::DIRECTION;

    auto vToId = [this](Vertex const& v) {
      return ev.p_cluster->vertexToId(v);
    };

    auto getSiteBraKet = [this, &vToId](Vertex const& v) {
      return ev.p_cluster->sites.at(vToId(v)) *
             dag(ev.p_cluster->sites.at(vToId(v)))
               .prime(AUXLINK, ev.p_cluster->BRAKET_OFFSET);
    };

    auto applyDeltaEdge = [this](ITensor& t, Vertex const& v, DIRECTION edge,
                                 DIRECTION dir) {
      // (edge = LEFT or RIGHT => dir = UP or DOWN) or
      // (edge = UP or DOWN => dir = LEFT or RIGHT)
      if ((edge == dir) || ((edge + 2) % 4 == dir)) {
        std::cout << "[TransferOpVecProd::operator()] Invalid input: edge= "
                  << edge << " dir: " << dir << std::endl;
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

      t *= delta(ev.p_ctmEnv->tauxByVertex(edge, v + s, dir),
                 ev.p_ctmEnv->tauxByVertex(edge, v, (dir + 2) % 4));
    };

    // TODO assume(check?) the bT has the correct indices of the boundary C-T-C

    // Apply lY rows or lX columns depending on chosen direction
    auto v = v_ref;
    if (dir == DIRECTION::RIGHT) {
      for (int col = 0; col < ev.p_cluster->lX; col++) {
        v = v_ref + col * Shift(1, 0);

        if (DBG)
          std::cout << "T_U[" << v << " => " << vToId(v) << "]" << std::endl;

        if (col > 0)
          applyDeltaEdge(bT, v, DIRECTION::UP, DIRECTION::RIGHT);
        bT *= ev.p_ctmEnv->T_U.at(vToId(v));

        if (DBG)
          Print(bT);
        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        Index tmp_down, tmp_right;

        tmp_down = ev.p_cluster->AIc(v, DIRECTION::UP);
        if (col > 0) {
          //     applyDeltaSite(tN,v,DIRECTION::LEFT)
          tmp_right = ev.p_cluster->AIc(v + Shift(-1, 0), DIRECTION::RIGHT);
        } else {
          tmp_right = ev.p_cluster->AIc(v, DIRECTION::LEFT);
        };

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, ev.p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, ev.p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 = combiner(ev.p_cluster->AIc(v, DIRECTION::UP),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::UP),
                                       ev.p_cluster->BRAKET_OFFSET),
                                 ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                       ev.p_cluster->BRAKET_OFFSET));

        bT *= tmp_cmb0;
        // TODO use delta instead of reindex
        bT = reindex(bT, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        bT *= (getSiteBraKet(v) * tmp_cmb1);

        if (DBG)
          std::cout << "T_D[" << v << " => " << vToId(v) << "]" << std::endl;

        if (col > 0)
          applyDeltaEdge(bT, v, DIRECTION::DOWN, DIRECTION::RIGHT);
        bT *= ev.p_ctmEnv->T_D.at(vToId(v));

        if (DBG)
          std::cout << ">>>>> Appended col X= " << col << " <<<<<" << std::endl;
        if (DBG)
          Print(bT);
      }
    } else if (dir == DIRECTION::DOWN) {
      for (int row = 0; row < ev.p_cluster->lY; row++) {
        v = v_ref + row * Shift(0, 1);

        if (DBG)
          std::cout << "T_L[" << v << " => " << vToId(v) << "]" << std::endl;

        if (row > 0)
          applyDeltaEdge(bT, v, DIRECTION::LEFT, DIRECTION::DOWN);
        bT *= ev.p_ctmEnv->T_L.at(vToId(v));

        if (DBG)
          Print(bT);
        if (DBG)
          std::cout << "[" << v << " => " << vToId(v) << "]" << std::endl;

        Index tmp_down, tmp_right;

        if (row > 0) {
          //     applyDeltaSite(bT,v,DIRECTION::UP)
          tmp_down = ev.p_cluster->AIc(v + Shift(0, -1), DIRECTION::DOWN);
        } else {
          tmp_down = ev.p_cluster->AIc(v, DIRECTION::UP);
        }
        tmp_right = ev.p_cluster->AIc(v, DIRECTION::LEFT);

        auto tmp_cmb0 =
          combiner(tmp_down, prime(tmp_down, ev.p_cluster->BRAKET_OFFSET),
                   tmp_right, prime(tmp_right, ev.p_cluster->BRAKET_OFFSET));
        auto tmp_cmb1 = combiner(ev.p_cluster->AIc(v, DIRECTION::UP),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::UP),
                                       ev.p_cluster->BRAKET_OFFSET),
                                 ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                 prime(ev.p_cluster->AIc(v, DIRECTION::LEFT),
                                       ev.p_cluster->BRAKET_OFFSET));

        bT *= tmp_cmb0;
        // TODO use delta instead of reindex
        bT = reindex(bT, combinedIndex(tmp_cmb0), combinedIndex(tmp_cmb1));
        bT *= getSiteBraKet(v) * tmp_cmb1;

        if (DBG)
          std::cout << "T_R[" << v << " => " << vToId(v) << "]" << std::endl;

        if (row > 0)
          applyDeltaEdge(bT, v, DIRECTION::RIGHT, DIRECTION::DOWN);
        bT *= ev.p_ctmEnv->T_R.at(vToId(v));

        if (DBG)
          std::cout << ">>>>> Appended row X= " << row << " <<<<<" << std::endl;
        if (DBG)
          Print(bT);
      }
    }

    // reset indices back to originals to reflect new bT
    if (dir == DIRECTION::RIGHT) {
      applyDeltaEdge(bT, v_ref, DIRECTION::UP, DIRECTION::RIGHT);
      applyDeltaEdge(bT, v_ref, DIRECTION::DOWN, DIRECTION::RIGHT);
      auto tmp_del =
        delta(ev.p_cluster->AIc(v_ref + Shift(-1, 0), DIRECTION::RIGHT),
              ev.p_cluster->AIc(v_ref, DIRECTION::LEFT));
      bT *= tmp_del;
      bT *= prime(tmp_del, ev.p_cluster->BRAKET_OFFSET);
    } else if (dir == DIRECTION::DOWN) {
      applyDeltaEdge(bT, v_ref, DIRECTION::LEFT, DIRECTION::DOWN);
      applyDeltaEdge(bT, v_ref, DIRECTION::RIGHT, DIRECTION::DOWN);
      auto tmp_del =
        delta(ev.p_cluster->AIc(v_ref + Shift(0, -1), DIRECTION::DOWN),
              ev.p_cluster->AIc(v_ref, DIRECTION::UP));
      bT *= tmp_del;
      bT *= prime(tmp_del, ev.p_cluster->BRAKET_OFFSET);
    }
  }

  double analyzeBoundaryVariance(EVBuilder const& ev,
                                 Vertex const& v,
                                 CtmEnv::DIRECTION dir,
                                 bool DBG) {
    using DIRECTION = CtmEnv::DIRECTION;

    auto vToId = [ev](Vertex const& v) { return ev.p_cluster->vertexToId(v); };

    auto siteId = vToId(v);
    Vertex v_other;
    std::string siteId_other;
    TransferOpVecProd_itensor tvp(ev, v, dir);
    ITensor bT, bTother;

    // construct boundaries
    if (dir == DIRECTION::RIGHT) {
      bT = ev.p_ctmEnv->C_LU.at(siteId);
      bT *= ev.p_ctmEnv->T_L.at(siteId);
      bT *= ev.p_ctmEnv->C_LD.at(siteId);

      v_other = v + Shift(-1, 0);
      siteId_other = vToId(v_other);
      bTother = ev.p_ctmEnv->C_RU.at(siteId_other);
      bTother *= ev.p_ctmEnv->T_R.at(siteId_other);
      bTother *= ev.p_ctmEnv->C_RD.at(siteId_other);

      bTother *= delta(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::UP, v_other, DIRECTION::RIGHT),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::UP, v, DIRECTION::LEFT));
      auto tmp_del = delta(ev.p_cluster->AIc(v_other, DIRECTION::RIGHT),
                           ev.p_cluster->AIc(v, DIRECTION::LEFT));
      bTother *= tmp_del;
      bTother *= prime(tmp_del, ev.p_cluster->BRAKET_OFFSET);
      bTother *= delta(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::DOWN, v_other, DIRECTION::RIGHT),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::DOWN, v, DIRECTION::LEFT));
    } else if (dir == DIRECTION::DOWN) {
      bT = ev.p_ctmEnv->C_LU.at(siteId);
      bT *= ev.p_ctmEnv->T_U.at(siteId);
      bT *= ev.p_ctmEnv->C_RU.at(siteId);

      v_other = v + Shift(0, -1);
      siteId_other = vToId(v_other);
      bTother = ev.p_ctmEnv->C_LD.at(siteId_other);
      bTother *= ev.p_ctmEnv->T_D.at(siteId_other);
      bTother *= ev.p_ctmEnv->C_RD.at(siteId_other);

      bTother *= delta(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::LEFT, v_other, DIRECTION::DOWN),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::LEFT, v, DIRECTION::UP));
      auto tmp_del = delta(ev.p_cluster->AIc(v_other, DIRECTION::DOWN),
                           ev.p_cluster->AIc(v, DIRECTION::UP));
      bTother *= tmp_del;
      bTother *= prime(tmp_del, ev.p_cluster->BRAKET_OFFSET);
      bTother *= delta(
        ev.p_ctmEnv->tauxByVertex(DIRECTION::RIGHT, v_other, DIRECTION::DOWN),
        ev.p_ctmEnv->tauxByVertex(DIRECTION::RIGHT, v, DIRECTION::UP));
    }

    auto bb = bT * bTother;
    if (bb.r() > 0)
      throw std::runtime_error("[analyzeBoundaryVariance] bTb rank > 0");

    // apply transfer operator once
    tvp(bT, DBG);

    // compute <boundary|T|boundary>
    auto bTb = bT * bTother;
    if (bTb.r() > 0)
      throw std::runtime_error("[analyzeBoundaryVariance] bTb rank > 0");

    // apply transfer operator second time
    tvp(bT, DBG);

    // compute <boundary|T^2|boundary>
    auto bTTb = bT * bTother;
    if (bTTb.r() > 0)
      throw std::runtime_error("[analyzeBoundaryVariance] bTTb rank > 0");

    double bndr_var = sumels(bTTb) / sumels(bb) -
                      (sumels(bTb) / sumels(bb)) * (sumels(bTb) / sumels(bb));

    if (sumels(bb) < 10e-4 || sumels(bTb) < 10e-4 || sumels(bTTb) < 10e-4) {
      std::cout << "[analyzeBoundaryVariance] bb= " << sumels(bb)
                << " bTb= " << sumels(bTb) << " bTTb= " << sumels(bTTb)
                << std::endl;
    }

    if (DBG)
      std::cout << "bTTb: " << sumels(bTTb) << " bTb: " << sumels(bTb)
                << " bb: " << sumels(bb) << " bTTb/bb-(bTb/bb)^2: " << bndr_var
                << std::endl;

    return bndr_var;
  }

}  // namespace itensor