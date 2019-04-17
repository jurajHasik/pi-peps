#include "pi-peps/config.h"
#include "pi-peps/full-update.h"

using namespace itensor;

ITensor pseudoInverse(ITensor const& M, Args const& args) {
  auto dbg = args.getBool("dbg", false);
  auto dbgLvl = args.getInt("dbgLevel", 0);
  auto svd_cutoff = args.getReal("pseudoInvCutoff", 1.0e-15);
  auto svd_cutoff_insert = args.getReal("pseudoInvCutoffInsert", 0.0);

  double machine_eps = std::numeric_limits<double>::epsilon();
  if (dbg && (dbgLvl >= 3)) {
    std::cout << "M EPS: " << machine_eps << std::endl;
    std::cout << "pseudoInverse: svd_cutoff = " << svd_cutoff << std::endl;
  }

  // TODO implement check on index structure of M - expect two different indices

  auto i0 = M.inds()[0];
  auto i1 = M.inds()[1];

  ITensor U(i0), dM, Vt;
  svd(M, U, dM, Vt, {"Truncate", false});

  std::vector<double> dM_elems;
  for (int i = 1; i <= dM.inds().front().m(); i++) {
    if (dM.real(dM.inds().front()(i), dM.inds().back()(i)) /
          dM.real(dM.inds().front()(1), dM.inds().back()(1)) >
        svd_cutoff) {
      dM_elems.push_back(1.0 /
                         dM.real(dM.inds().front()(i), dM.inds().back()(i)));
    } else {
      dM_elems.push_back(svd_cutoff_insert);
    }
  }

  auto inv_dM = diagTensor(dM_elems, dM.inds().front(), dM.inds().back());

  auto InvM = (conj(U) * inv_dM) * conj(Vt);
  InvM *= delta(prime(i0, 1), i1);
  InvM *= delta(prime(i1, 1), i0);
  InvM.prime(-1);

  return InvM;
}

// TODO handle inputs to linsystem with multiple indices
Args fullUpdate_ALS2S_IT(MPO_2site const& mpo,
                         Cluster& cls,
                         CtmEnv const& ctmEnv,
                         std::vector<std::string> const& tn,
                         std::vector<int> pl,
                         LinSysSolver const& ls,
                         Args const& args) {
  using DIRECTION = CtmEnv::DIRECTION;

  auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter", 50);
  auto dbg = args.getBool("fuDbg", false);
  auto dbgLvl = args.getInt("fuDbgLevel", 0);
  auto symmProtoEnv = args.getBool("symmetrizeProtoEnv", true);
  auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv", true);
  auto fuTrialInit = args.getBool("fuTrialInit", false);
  auto epsdistf = args.getReal("epsdistf", 1.0e-8);
  auto epsregularisation = args.getReal("epsregularisation", 0.0);
  auto svd_cutoff = args.getReal("pseudoInvCutoff", 1.0e-15);
  auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap", 0.0);
  auto otNormType = args.getString("otNormType");

  double machine_eps = std::numeric_limits<double>::epsilon();
  if (dbg && (dbgLvl >= 1))
    std::cout << "M EPS: " << machine_eps << std::endl;

  std::chrono::steady_clock::time_point t_begin_int, t_end_int;

  // prepare to hold diagnostic data
  Args diag_data = Args::global();
  std::vector<ITensor> orig_tensors = {cls.sites.at(tn[0]),
                                       cls.sites.at(tn[1])};

  if (dbg) {
    std::cout << "GATE: " << mpo.uuid << " | ";
    std::cout << tn[0] << " -> " << pl[0] << " -> " << pl[1] << " -> " << tn[1]
              << std::endl;

    if (dbg && (dbgLvl >= 2)) {
      PrintData(mpo.H1);
      PrintData(mpo.H2);
    }
  }

  // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
  double m = 0.;
  auto max_m = [&m](double d) {
    if (std::abs(d) > m)
      m = std::abs(d);
  };

  // read off auxiliary and physical indices of the cluster sites
  // std::array<Index, 2> aux;
  // for (int i=0; i<2; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

  std::array<Index, 2> phys;
  for (int i = 0; i < 2; i++)
    phys[i] = cls.mphys.at(tn[i]);

  std::array<Index, 2> opPI({mpo.Is1, mpo.Is2});

  if (dbg) {
    std::cout << "On-site indices:" << std::endl;
    for (int i = 0; i < 2; i++) {
      // std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
      std::cout << tn[i] << " : "
                << " " << phys[i] << std::endl;
    }
  }

  // reduced on-site tensors over which the gate is applied
  Index iQA, iQB;
  // ITensor QA, eA(prime(aux[0],pl[0]), phys[0]);
  // ITensor QB, eB(prime(aux[1],pl[1]), phys[1]);
  ITensor QA, eA(cls.AIc(tn[0], pl[0]), phys[0]);
  ITensor QB, eB(cls.AIc(tn[1], pl[1]), phys[1]);

  // tensor holding the reduced environment of 2 sites
  ITensor eRE;
  {
    t_begin_int = std::chrono::steady_clock::now();

    // find integer identifier of on-site tensors within CtmEnv
    std::vector<int> si;
    for (int i = 0; i < 2; i++) {
      si.push_back(std::distance(ctmEnv.siteIds.begin(),
                                 std::find(std::begin(ctmEnv.siteIds),
                                           std::end(ctmEnv.siteIds), tn[i])));
    }
    if (dbg) {
      std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
      for (int i = 0; i < 2; ++i) {
        std::cout << tn[i] << " -> " << si[i] << std::endl;
      }
    }

    // prepare map from on-site tensor aux-indices to half row/column T
    // environment tensors
    std::array<const std::map<std::string, ITensor>* const, 4> iToT(
      {&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R, &ctmEnv.T_D});

    // prepare map from on-site tensor aux-indices pair to half corner T-C-T
    // environment tensors
    const std::map<int, const std::map<std::string, ITensor>* const> iToC(
      {{23, &ctmEnv.C_RD},
       {32, &ctmEnv.C_RD},
       {21, &ctmEnv.C_RU},
       {12, &ctmEnv.C_RU},
       {3, &ctmEnv.C_LD},
       {30, &ctmEnv.C_LD},
       {1, &ctmEnv.C_LU},
       {10, &ctmEnv.C_LU}});

    // for every on-site tensor point from primeLevel(index) to ENV index
    // eg. I_XH or I_XV (with appropriate prime level).
    std::array<std::array<Index, 3>, 2> iToE;  // indexToENVIndex => iToE

    // precompute 4 (proto)corners of 2x1 environment OR 1x2 environment
    std::vector<ITensor> pc(4);
    std::array<std::array<int, 3>, 2> tmp_iToE;

    int plI1, plI2, crI;
    plI1 = (pl[0] + 1) % 4;  // direction of the connecting edge
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[0] = ((*iToT.at(plI1)).at(tn[0]) * (*iToC.at(crI)).at(tn[0])) *
            (*iToT.at(plI2)).at(tn[0]);
    tmp_iToE[0][0] = plI1;
    tmp_iToE[0][1] = plI2;

    plI1 = plI2;
    plI2 = (plI1 + 1) % 4;  // opposite edge
    crI = plI1 * 10 + plI2;
    pc[1] = (*iToC.at(crI)).at(tn[0]) * (*iToT.at(plI2)).at(tn[0]);
    tmp_iToE[0][2] = plI2;

    plI1 = (pl[1] + 1) % 4;
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[2] = ((*iToT.at(plI1)).at(tn[1]) * (*iToC.at(crI)).at(tn[1])) *
            (*iToT.at(plI2)).at(tn[1]);
    tmp_iToE[1][0] = plI1;
    tmp_iToE[1][1] = plI2;

    plI1 = plI2;
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[3] = (*iToC.at(crI)).at(tn[1]) * (*iToT.at(plI2)).at(tn[1]);
    tmp_iToE[1][2] = plI2;

    if (dbg) {
      std::cout << "primeLevels (pl) of indices connected to ENV - site: "
                << std::endl;
      std::cout << tn[0] << ": " << tmp_iToE[0][0] << " " << tmp_iToE[0][1]
                << " " << tmp_iToE[0][2] << std::endl;
      std::cout << tn[1] << ": " << tmp_iToE[1][0] << " " << tmp_iToE[1][1]
                << " " << tmp_iToE[1][2] << std::endl;
    }

    if (dbg) {
      for (int i = 0; i < 2; i++) {
        std::cout << "Site: " << tn[i] << " ";
        for (auto const& ind : iToE[i])
          if (ind)
            std::cout << ind << " ";
        std::cout << std::endl;
      }
    }

    if (dbg && (dbgLvl >= 3))
      for (int i = 0; i < 4; i++)
        Print(pc[i]);

    // prepare proto-corners for contraction
    // by which edge are the corners connected ?
    // 1) find corresponding Shift
    Shift shift;
    DIRECTION dir0, dir1;
    if (pl[0] == 0 and pl[1] == 2) {
      shift = Shift(-1, 0);
      dir0 = DIRECTION::UP;
      dir1 = DIRECTION::DOWN;
    } else if (pl[0] == 2 and pl[1] == 0) {
      shift = Shift(1, 0);
      dir0 = DIRECTION::DOWN;
      dir1 = DIRECTION::UP;
    } else if (pl[0] == 1 and pl[1] == 3) {
      shift = Shift(0, -1);
      dir0 = DIRECTION::RIGHT;
      dir1 = DIRECTION::LEFT;
    } else if (pl[0] == 3 and pl[1] == 1) {
      shift = Shift(0, 1);
      dir0 = DIRECTION::LEFT;
      dir1 = DIRECTION::RIGHT;
    } else
      throw std::runtime_error("[fullUpdate_ALS2S_IT] Invalid gate");
    // 2) find the vertex of tn[0]
    auto v0 = cls.idToV.at(tn[0]);
    auto v1 = v0 + shift;

    pc[0] *= delta(ctmEnv.tauxByVertex(dir0, v0, pl[0]),
                   ctmEnv.tauxByVertex(dir0, v1, pl[1]));
    pc[1] *= delta(ctmEnv.tauxByVertex(dir1, v0, pl[0]),
                   ctmEnv.tauxByVertex(dir1, v1, pl[1]));

    if (dbg && (dbgLvl >= 3))
      for (int i = 0; i < 4; i++)
        Print(pc[i]);

    // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE *******************

    // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
    t_begin_int = std::chrono::steady_clock::now();

    // A--B
    // Decompose A tensor on which the gate is applied
    {
      // TODO mask ? cutoff ?
      // Internally ITensor takes sqrt of singular values
      auto maskS = [&machine_eps](Real r) {
        return (r > 10.0 * machine_eps) ? 1.0 : 1.0;
      };
      auto cutoffS = [&machine_eps](Real r) {
        return (r > 10.0 * machine_eps) ? r : r;
      };

      ITensor tempSA;
      svd(cls.sites.at(tn[0]), eA, tempSA, QA, {"Truncate", false});

      // has to be non-zero
      tempSA *= 1.0 / tempSA.real(tempSA.inds()[0](1), tempSA.inds()[1](1));
      tempSA.apply(cutoffS);
      auto maskSA = tempSA;
      maskSA.apply(maskS);
      auto tmpI_SA = commonIndex(tempSA, eA);

      iQA = Index("auxQA", commonIndex(QA, tempSA).m(), AUXLINK, 0);
      eA = (eA * tempSA) * delta(commonIndex(QA, tempSA), iQA);
      QA *= maskSA;
      QA *= delta(tmpI_SA, iQA);
      // QA *= delta(commonIndex(QA,tempSA), iQA);

      // Prepare corner of A
      ITensor tempC = (pc[0] * QA) * prime(conj(QA), AUXLINK, 4);
      if (dbg && (dbgLvl >= 3))
        Print(tempC);

      eRE = tempC * pc[1];
      if (dbg && (dbgLvl >= 3))
        Print(eRE);

      eRE *= pc[3];
      if (dbg && (dbgLvl >= 3))
        Print(eRE);

      // Decompose B tensor on which the gate is applied
      ITensor tempSB;
      svd(cls.sites.at(tn[1]), eB, tempSB, QB, {"Truncate", false});

      tempSB *= 1.0 / tempSB.real(tempSB.inds()[0](1), tempSB.inds()[1](1));
      tempSB.apply(cutoffS);
      auto maskSB = tempSB;
      maskSB.apply(maskS);
      auto tmpI_SB = commonIndex(tempSB, eB);

      iQB = Index("auxQB", commonIndex(QB, tempSB).m(), AUXLINK, 0);
      eB = (eB * tempSB) * delta(commonIndex(QB, tempSB), iQB);
      QB *= maskSB;
      QB *= delta(tmpI_SB, iQB);
      // QB *= delta(commonIndex(QB,tempSB), iQB);

      tempC = (pc[2] * QB) * prime(conj(QB), AUXLINK, 4);
      if (dbg && (dbgLvl >= 3))
        Print(tempC);

      eRE *= tempC;
    }

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Constructed reduced Env - T: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                   t_end_int - t_begin_int)
                     .count() /
                   1000000.0
              << " [sec]" << std::endl;
    if (dbg && (dbgLvl >= 3))
      Print(eRE);
    // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
  }

  double condNum = 1.0;
  std::string diag_protoEnv, diag_protoEnv_descriptor;
  double diag_maxMsymLE, diag_maxMasymLE;
  double diag_maxMsymFN, diag_maxMasymFN;
  if (symmProtoEnv) {
    // ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
    t_begin_int = std::chrono::steady_clock::now();
    auto cmbKet = combiner(iQA, iQB);
    auto cmbBra = prime(cmbKet, 4);

    eRE = (eRE * cmbKet) * cmbBra;

    ITensor eRE_sym = 0.5 * (eRE + swapPrime(eRE, 0, 4));
    ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE, 0, 4));

    m = 0.;
    eRE_sym.visit(max_m);
    diag_maxMsymLE = m;
    std::cout << "eRE_sym max element: " << m << std::endl;
    m = 0.;
    eRE_asym.visit(max_m);
    diag_maxMasymLE = m;
    std::cout << "eRE_asym max element: " << m << std::endl;

    diag_maxMsymFN = norm(eRE_sym);
    diag_maxMasymFN = norm(eRE_asym);

    if (posDefProtoEnv) {
      eRE_sym *= delta(combinedIndex(cmbBra), prime(combinedIndex(cmbKet)));

      // ##### V3 ######################################################
      ITensor U_eRE, D_eRE;
      diagHermitian(eRE_sym, U_eRE, D_eRE);

      double msign = 1.0;
      double mval = 0.;
      double nval = 1.0e+16;
      std::vector<double> dM_elems;
      for (int idm = 1; idm <= D_eRE.inds().front().m(); idm++) {
        dM_elems.push_back(
          D_eRE.real(D_eRE.inds().front()(idm), D_eRE.inds().back()(idm)));
        if (std::abs(dM_elems.back()) > mval) {
          mval = std::abs(dM_elems.back());
          msign = dM_elems.back() / mval;
        }
        // find the lowest eigenvalue in magnitude
        if (std::abs(dM_elems.back()) < nval)
          nval = std::abs(dM_elems.back());
      }
      
      if (msign < 0.0)
        for (auto& elem : dM_elems)
          elem = elem * (-1.0);

      // Drop negative EV's and count negative EVs, EVs lower than cutoff
      double traceDM = 0.0;
      int countCTF = 0;
      int countNEG = 0;
      if (dbg && (dbgLvl >= 1)) {
        std::cout << "REFINED SPECTRUM" << std::endl;
        std::cout << "MAX EV: " << mval << std::endl;
      }
      for (auto& elem : dM_elems) {
        if (elem < 0.0) {
          if (dbg && (dbgLvl >= 2))
            std::cout << elem << " -> " << 0.0 << std::endl;
          elem = 0.0;
          // elem = 1.25e-4;
          countNEG += 1;
          // } else if (elem < svd_cutoff) {
          // 	countCTF += 1;
          // 	if(dbg && (dbgLvl >= 2)) std::cout<< elem << std::endl;
        }
      }

      condNum = mval / nval;

      std::ostringstream oss;
      oss << std::scientific << mval << " " << condNum << " " << countCTF << " "
          << countNEG << " " << dM_elems.size();

      diag_protoEnv_descriptor = "MaxEV condNum EV<CTF EV<0 TotalEV";
      diag_protoEnv = oss.str();

      if (dbg && (dbgLvl >= 1)) {
        std::cout << "REFINED SPECTRUM" << std::endl;
        std::cout << std::scientific << "MAX EV: " << mval
                  << " MIN EV: " << nval << std::endl;
        std::cout << "RATIO svd_cutoff/negative/all " << countCTF << "/"
                  << countNEG << "/" << dM_elems.size() << std::endl;
      }
      // ##### END V3 ##################################################

      D_eRE = diagTensor(dM_elems, D_eRE.inds().front(), D_eRE.inds().back());
      // D_eRE = D_eRE / traceDM;

      eRE_sym = ((conj(U_eRE) * D_eRE) * prime(U_eRE)) *
                delta(combinedIndex(cmbBra), prime(combinedIndex(cmbKet)));

      // eRE_sym *= 1.0/mval;
    }

    eRE = (eRE_sym * cmbKet) * cmbBra;

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Symmetrized reduced env - T: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                   t_end_int - t_begin_int)
                     .count() /
                   1000000.0
              << " [sec]" << std::endl;
    // ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
  }

  // ***** FORM "PROTO" ENVIRONMENTS FOR M and K *****************************
  t_begin_int = std::chrono::steady_clock::now();

  ITensor protoK =
    (eRE * eA) * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1]));
  protoK *= eB;

  {
    auto tempOp = mpo.H1 * mpo.H2;
    tempOp = (tempOp * delta(mpo.Is1, phys[0])) * delta(mpo.Is2, phys[1]);
    tempOp = (tempOp * prime(delta(mpo.Is1, phys[0]))) *
             prime(delta(mpo.Is2, phys[1]));

    protoK *= tempOp;
    protoK.noprime(PHYS);
  }
  if (dbg && (dbgLvl >= 3))
    Print(protoK);

  std::cout << "eRE.scale(): " << eRE.scale()
            << " protoK.scale(): " << protoK.scale() << std::endl;
  t_end_int = std::chrono::steady_clock::now();
  std::cout << "Proto Envs for M and K constructed - T: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                 t_end_int - t_begin_int)
                   .count() /
                 1000000.0
            << " [sec]" << std::endl;
  // ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************

  // ********************************************************************************************
  // 	     OPTIMIZE VIA CG *
  // ********************************************************************************************

  // <psi|U^dag U|psi>
  double normUPsi;
  {
    auto tempOp = conj(mpo.H1) * conj(mpo.H2);
    tempOp = (tempOp * delta(mpo.Is1, phys[0])) * delta(mpo.Is2, phys[1]);
    tempOp = (tempOp * prime(delta(mpo.Is1, phys[0]))) *
             prime(delta(mpo.Is2, phys[1]));

    auto NORMUPSI = protoK * tempOp;
    NORMUPSI.noprime(PHYS);
    NORMUPSI *= prime(conj(eA), AUXLINK, 4);
    NORMUPSI *= prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4);
    NORMUPSI *= prime(conj(eB), AUXLINK, 4);

    if (NORMUPSI.r() > 0)
      std::cout << "ERROR: NORMUPSI rank > 0" << std::endl;
    normUPsi = sumels(NORMUPSI);
  }

  int altlstsquares_iter = 0;
  bool converged = false;
  std::vector<double> fdist, fdistN, vec_normPsi;

  // check bond before update
  {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    // reduced tensors
    auto tmpT = eA * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])) * eB;

    ITensor tmpEA(iQA, phys[0]), S, tmpEB;
    svd(tmpT, tmpEA, S, tmpEB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
    std::cout << "RED_SPEC: ";
    S.visit(printS);
    std::cout << std::endl;

    // full tensors
    tmpT = QA * tmpT * QB;

    std::vector<Index> indsA;
    for (int i = 0; i < 4; i++)
      if (i != pl[0])
        indsA.push_back(cls.AIc(tn[0], i));
    indsA.push_back(phys[0]);

    ITensor tmpA(indsA), S2, tmpB;
    svd(tmpT, tmpA, S2, tmpB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S2 *= 1.0 / S2.real(S2.inds()[0](1), S2.inds()[1](1));
    std::cout << "FULL_SPEC: ";
    S2.visit(printS);
    std::cout << std::endl;
  }

  // trial initialization
  if (fuTrialInit) {
    auto SqrtT = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? std::sqrt(r) : std::sqrt(r);
    };
    // auto SqrtT = [&machine_eps](Real r) { return (r > 10.0*machine_eps) ?
    // sqrt(r) : 0; };
    auto printS = [](Real r) {
      std::cout << std::scientific << r << std::endl;
    };

    auto tmpOp = mpo.H1 * mpo.H2;
    tmpOp = (tmpOp * delta(mpo.Is1, phys[0])) * delta(mpo.Is2, phys[1]);
    tmpOp =
      (tmpOp * prime(delta(mpo.Is1, phys[0]))) * prime(delta(mpo.Is2, phys[1]));

    auto tmpT =
      eA * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])) * eB * tmpOp;
    tmpT.noprime(PHYS);

    ITensor tmpEA(iQA, phys[0]), S, tmpEB;
    svd(tmpT, tmpEA, S, tmpEB, {"Truncate", false});

    S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
    S.apply(SqrtT);
    S.visit(printS);

    eA = tmpEA * S * delta(commonIndex(tmpEB, S), cls.AIc(tn[0], pl[0]));
    eB = tmpEB * S * delta(commonIndex(tmpEA, S), cls.AIc(tn[1], pl[1]));
  }

  std::cout << "ENTERING ALS LOOP" << std::endl;
  t_begin_int = std::chrono::steady_clock::now();
  while (not converged) {
    // Optimizing eA
    // 1) construct matrix M, which is defined as <psi~|psi~> = eA^dag * M * eA
    {
      ITensor M =
        (eRE * eB) * delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0]));
      M *= (prime(conj(eB), AUXLINK, 4) *
            prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4));

      ITensor K = protoK * prime(conj(eB), AUXLINK, 4);
      K *= prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4);

      // <psi'|psi'>
      auto NORMPSI = (prime(conj(eA), AUXLINK, 4) * M) * eA;
      // <psi'|U|psi>
      auto OVERLAP = prime(conj(eA), AUXLINK, 4) * K;

      if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
        std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
      double t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
      double t_fdistN =
        1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) +
        1.0;

      fdist.push_back(t_fdist);
      fdistN.push_back(t_fdistN);
      vec_normPsi.push_back(sumels(NORMPSI));

      // condition for stopping ALS procedure
      if (fdist.back() < 1.0e-08) {
        converged = true;
        break;
      }
      if ((fdist.size() > 1) &&
          std::abs(fdist.back() - fdist[fdist.size() - 2]) / fdist[0] <
            epsdistf) {
        converged = true;
        break;
      }

      auto RES = M * eA - K;
      std::cout << "Norm(RES_A)= " << norm(RES) << std::endl;

      // eA: aux, aux, phys
      // K : aux^offset, aux^offset, phys^offset
      M *= delta(phys[0], prime(phys[0], 4));
      K.prime(PHYS, 4);

      auto cmb0 = combiner(iQA, cls.AIc(tn[0], pl[0]), phys[0]);
      auto cmb1 = combiner(prime(iQA, 4), prime(cls.AIc(tn[0], pl[0]), 4),
                           prime(phys[0], 4));
      M = (cmb0 * M) * cmb1;
      // regularize Hessian
      // std::vector<double> eps_reg(combinedIndex(cmb0, epsregularisation));
      // M += diagTensor(eps_reg, combinedIndex(cmb0), combinedIndex(cmb1));
      K *= cmb1;
      eA *= cmb0;

      linsystem(M, K, eA, ls, args);

      eA *= cmb0;
    }

    // Optimizing eB
    // 1) construct matrix M, which is defined as <psi~|psi~> = eB^dag * M * eB
    {
      ITensor M =
        (eRE * eA) * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1]));
      M *= (prime(conj(eA), AUXLINK, 4) *
            prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4));

      ITensor K = protoK * prime(conj(eA), AUXLINK, 4);
      K *= prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4);

      // <psi'|psi'>
      auto NORMPSI = (prime(conj(eB), AUXLINK, 4) * M) * eB;
      // <psi'|U|psi>
      auto OVERLAP = prime(conj(eB), AUXLINK, 4) * K;

      if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
        std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
      double t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
      double t_fdistN =
        1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) +
        1.0;

      fdist.push_back(t_fdist);
      fdistN.push_back(t_fdistN);
      vec_normPsi.push_back(sumels(NORMPSI));

      // condition for stopping ALS procedure
      if (fdist.back() < 1.0e-08) {
        converged = true;
        break;
      }
      if ((fdist.size() > 1) &&
          std::abs(fdist.back() - fdist[fdist.size() - 2]) / fdist[0] <
            epsdistf) {
        converged = true;
        break;
      }
      auto RES = M * eB - K;
      std::cout << "Norm(RES_B)= " << norm(RES) << std::endl;

      M *= delta(phys[1], prime(phys[1], 4));
      K.prime(PHYS, 4);

      auto cmb0 = combiner(iQB, cls.AIc(tn[1], pl[1]), phys[1]);
      auto cmb1 = combiner(prime(iQB, 4), prime(cls.AIc(tn[1], pl[1]), 4),
                           prime(phys[1], 4));
      M = (cmb0 * M) * cmb1;
      K *= cmb1;
      eB *= cmb0;

      linsystem(M, K, eB, ls, args);

      eB *= cmb0;
    }

    altlstsquares_iter++;
    if (altlstsquares_iter >= maxAltLstSqrIter)
      converged = true;
  }
  t_end_int = std::chrono::steady_clock::now();

  std::cout << "STEP f=||psi'>-|psi>|^2 normalized(f) norm(psi') norm(psi)"
            << std::endl;
  for (int i = 0; i < fdist.size(); i++)
    std::cout << i << " " << fdist[i] << " " << fdistN[i] << " "
              << vec_normPsi[i] << " " << normUPsi << std::endl;

  // spectrum post-analysis after update
  {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    // reduced tensors
    auto tmpT = eA * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])) * eB;

    ITensor tmpEA(iQA, phys[0]), S, tmpEB;
    svd(tmpT, tmpEA, S, tmpEB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
    std::cout << "RED_SPEC: ";
    S.visit(printS);
    std::cout << std::endl;

    // full tensors
    tmpT = QA * tmpT * QB;

    std::vector<Index> indsA;
    for (int i = 0; i < 4; i++)
      if (i != pl[0])
        indsA.push_back(cls.AIc(tn[0], i));
    indsA.push_back(phys[0]);

    ITensor tmpA(indsA), S2, tmpB;
    svd(tmpT, tmpA, S2, tmpB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S2 *= 1.0 / S2.real(S2.inds()[0](1), S2.inds()[1](1));
    std::cout << "FULL_SPEC: ";
    S2.visit(printS);
    std::cout << std::endl;
  }

  // BALANCE tensors
  double iso_tot_mag = 1.0;
  m = 0.;
  eA.visit(max_m);
  eA = eA / m;
  iso_tot_mag = iso_tot_mag * m;
  eB.visit(max_m);
  eB = eB / m;
  iso_tot_mag = iso_tot_mag * m;
  eA = eA * std::pow(iso_tot_mag, (1.0 / 2.0));
  eB = eB * std::pow(iso_tot_mag, (1.0 / 2.0));

  // update on-site tensors of cluster
  cls.sites.at(tn[0]) = QA * eA;
  cls.sites.at(tn[1]) = QB * eB;

  // max element of on-site tensors
  // or norm-distance of new vs original tensors
  std::string diag_maxElem;
  std::ostringstream oss_diag_siteScale;
  for (int i = 0; i < 2; i++) {
    m = 0.;
    m = norm(cls.sites.at(tn[i]) - orig_tensors[i]);
    // site_e.second.visit(max_m);
    diag_maxElem = diag_maxElem + tn[i] + " : " + std::to_string(m) + " ";
    oss_diag_siteScale << tn[i] << " : " << cls.sites.at(tn[i]).scale() << " ";
  }
  std::cout << oss_diag_siteScale.str() << std::endl;
  std::cout << diag_maxElem << std::endl;

  // normalize updated tensors
  if (otNormType == "BLE") {
    for (int i = 0; i < tn.size(); i++) {
      m = 0.;
      cls.sites.at(tn[i]).visit(max_m);
      cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / sqrt(m);
    }
  } else if (otNormType == "BALANCE") {
    // double iso_tot_mag = 1.0;
    //    for ( auto & site_e : cls.sites)  {
    //    	m = 0.;
    // 	site_e.second.visit(max_m);
    //    	site_e.second = site_e.second / m;
    //    	iso_tot_mag = iso_tot_mag * m;
    //    }
    //    for (auto & site_e : cls.sites) {
    //    	site_e.second = site_e.second * std::pow(iso_tot_mag, (1.0/(2.0 *
    //    cls.sites.size())) );
    //    }
    for (auto& site_e : cls.sites) {
      m = 0.;
      site_e.second.visit(max_m);
      site_e.second = site_e.second / m;
    }
    for (auto& site_e : cls.sites) {
      site_e.second =
        site_e.second /
        std::pow(vec_normPsi.back(), (1.0 / (2.0 * cls.sizeN * cls.sizeM)));
    }
  } else if (otNormType == "NONE") {
  } else {
    std::cout << "Unsupported on-site tensor normalisation after full update: "
              << otNormType << std::endl;
    exit(EXIT_FAILURE);
  }

  // max element of on-site tensors after normalization
  for (int i = 0; i < tn.size(); i++) {
    m = 0.;
    cls.sites.at(tn[i]).visit(max_m);
    std::cout << tn[i] << " : " << std::to_string(m) << " ";
  }
  std::cout << std::endl;

  // prepare and return diagnostic data
  diag_data.add("alsSweep", altlstsquares_iter);

  std::string siteMaxElem_descriptor = "site max_elem site max_elem";
  diag_data.add("siteMaxElem_descriptor", siteMaxElem_descriptor);
  diag_data.add("siteMaxElem", diag_maxElem);

  diag_data.add("ratioNonSymLE",
                diag_maxMasymLE / diag_maxMsymLE);  // ratio of largest elements
  diag_data.add("ratioNonSymFN",
                diag_maxMasymFN / diag_maxMsymFN);  // ratio of norms

  std::ostringstream oss;
  oss << std::scientific << fdist[0] << " " << fdist.back() << " " << fdistN[0]
      << " " << fdistN.back() << " " << vec_normPsi[0] << " "
      << vec_normPsi.back() << " "
      << std::chrono::duration_cast<std::chrono::microseconds>(t_end_int -
                                                               t_begin_int)
             .count() /
           1000000.0;

  std::string logMinDiag_descriptor =
    "f_init f_final nf_init nf_final norm(psi')_init norm(psi')_final time[s]";
  diag_data.add("locMinDiag_descriptor", logMinDiag_descriptor);
  diag_data.add("locMinDiag", oss.str());

  if (symmProtoEnv) {
    diag_data.add("diag_protoEnv", diag_protoEnv);
    diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
  }

  return diag_data;
}

Args fullUpdate_2S(MPO_2site const& mpo,
                   Cluster& cls,
                   CtmEnv const& ctmEnv,
                   std::vector<std::string> const& tn,
                   std::vector<int> pl,
                   LinSysSolver const& ls,
                   Args const& args) {
  using DIRECTION = CtmEnv::DIRECTION;

  auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter", 50);
  auto dbg = args.getBool("fuDbg", false);
  auto dbgLvl = args.getInt("fuDbgLevel", 0);
  auto symmProtoEnv = args.getBool("symmetrizeProtoEnv", true);
  auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv", true);
  auto fuTrialInit = args.getBool("fuTrialInit", false);
  auto epsdistf = args.getReal("epsdistf", 1.0e-8);
  auto epsregularisation = args.getReal("epsregularisation", 0.0);
  auto svd_cutoff = args.getReal("pseudoInvCutoff", 1.0e-15);
  auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap", 0.0);
  auto otNormType = args.getString("otNormType");

  double machine_eps = std::numeric_limits<double>::epsilon();
  if (dbg && (dbgLvl >= 1))
    std::cout << "M EPS: " << machine_eps << std::endl;

  std::chrono::steady_clock::time_point t_begin_int, t_end_int;

  // prepare to hold diagnostic data
  Args diag_data = Args::global();
  std::vector<ITensor> orig_tensors = {cls.sites.at(tn[0]),
                                       cls.sites.at(tn[1])};

  if (dbg) {
    std::cout << "GATE: " << mpo.uuid << " | ";
    std::cout << tn[0] << " -> " << pl[0] << " -> " << pl[1] << " -> " << tn[1]
              << std::endl;

    if (dbg && (dbgLvl >= 2)) {
      PrintData(mpo.H1);
      PrintData(mpo.H2);
    }
  }

  // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
  double m = 0.;
  auto max_m = [&m](double d) {
    if (std::abs(d) > m)
      m = std::abs(d);
  };

  // read off auxiliary and physical indices of the cluster sites
  // std::array<Index, 2> aux;
  // for (int i=0; i<2; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

  std::array<Index, 2> phys;
  for (int i = 0; i < 2; i++)
    phys[i] = cls.mphys.at(tn[i]);

  std::array<Index, 2> opPI({mpo.Is1, mpo.Is2});

  if (dbg) {
    std::cout << "On-site indices:" << std::endl;
    for (int i = 0; i < 2; i++) {
      // std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
      std::cout << tn[i] << " : "
                << " " << phys[i] << std::endl;
    }
  }

  // reduced on-site tensors over which the gate is applied
  Index iQA, iQB;
  // ITensor QA, eA(prime(aux[0],pl[0]), phys[0]);
  // ITensor QB, eB(prime(aux[1],pl[1]), phys[1]);
  ITensor QA, eA(cls.AIc(tn[0], pl[0]), phys[0]);
  ITensor QB, eB(cls.AIc(tn[1], pl[1]), phys[1]);

  // tensor holding the reduced environment of 2 sites
  ITensor eRE;
  {
    t_begin_int = std::chrono::steady_clock::now();

    // find integer identifier of on-site tensors within CtmEnv
    std::vector<int> si;
    for (int i = 0; i < 2; i++) {
      si.push_back(std::distance(ctmEnv.siteIds.begin(),
                                 std::find(std::begin(ctmEnv.siteIds),
                                           std::end(ctmEnv.siteIds), tn[i])));
    }
    if (dbg) {
      std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
      for (int i = 0; i < 2; ++i) {
        std::cout << tn[i] << " -> " << si[i] << std::endl;
      }
    }

    // prepare map from on-site tensor aux-indices to half row/column T
    // environment tensors
    std::array<const std::map<std::string, ITensor>* const, 4> iToT(
      {&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R, &ctmEnv.T_D});

    // prepare map from on-site tensor aux-indices pair to half corner T-C-T
    // environment tensors
    const std::map<int, const std::map<std::string, ITensor>* const> iToC(
      {{23, &ctmEnv.C_RD},
       {32, &ctmEnv.C_RD},
       {21, &ctmEnv.C_RU},
       {12, &ctmEnv.C_RU},
       {3, &ctmEnv.C_LD},
       {30, &ctmEnv.C_LD},
       {1, &ctmEnv.C_LU},
       {10, &ctmEnv.C_LU}});

    // for every on-site tensor point from primeLevel(index) to ENV index
    // eg. I_XH or I_XV (with appropriate prime level).
    std::array<std::array<Index, 3>, 2> iToE;  // indexToENVIndex => iToE

    // precompute 4 (proto)corners of 2x1 environment OR 1x2 environment
    std::vector<ITensor> pc(4);
    std::array<std::array<int, 3>, 2> tmp_iToE;

    int plI1, plI2, crI;
    plI1 = (pl[0] + 1) % 4;  // direction of the connecting edge
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[0] = ((*iToT.at(plI1)).at(tn[0]) * (*iToC.at(crI)).at(tn[0])) *
            (*iToT.at(plI2)).at(tn[0]);
    tmp_iToE[0][0] = plI1;
    tmp_iToE[0][1] = plI2;

    plI1 = plI2;
    plI2 = (plI1 + 1) % 4;  // opposite edge
    crI = plI1 * 10 + plI2;
    pc[1] = (*iToC.at(crI)).at(tn[0]) * (*iToT.at(plI2)).at(tn[0]);
    tmp_iToE[0][2] = plI2;

    plI1 = (pl[1] + 1) % 4;
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[2] = ((*iToT.at(plI1)).at(tn[1]) * (*iToC.at(crI)).at(tn[1])) *
            (*iToT.at(plI2)).at(tn[1]);
    tmp_iToE[1][0] = plI1;
    tmp_iToE[1][1] = plI2;

    plI1 = plI2;
    plI2 = (plI1 + 1) % 4;
    crI = plI1 * 10 + plI2;
    pc[3] = (*iToC.at(crI)).at(tn[1]) * (*iToT.at(plI2)).at(tn[1]);
    tmp_iToE[1][2] = plI2;

    if (dbg) {
      std::cout << "primeLevels (pl) of indices connected to ENV - site: "
                << std::endl;
      std::cout << tn[0] << ": " << tmp_iToE[0][0] << " " << tmp_iToE[0][1]
                << " " << tmp_iToE[0][2] << std::endl;
      std::cout << tn[1] << ": " << tmp_iToE[1][0] << " " << tmp_iToE[1][1]
                << " " << tmp_iToE[1][2] << std::endl;
    }

    if (dbg) {
      for (int i = 0; i < 2; i++) {
        std::cout << "Site: " << tn[i] << " ";
        for (auto const& ind : iToE[i])
          if (ind)
            std::cout << ind << " ";
        std::cout << std::endl;
      }
    }

    if (dbg && (dbgLvl >= 3))
      for (int i = 0; i < 4; i++)
        Print(pc[i]);

    // prepare proto-corners for contraction
    // by which edge are the corners connected ?
    // 1) find corresponding Shift
    Shift shift;
    DIRECTION dir0, dir1;
    if (pl[0] == 0 and pl[1] == 2) {
      shift = Shift(-1, 0);
      dir0 = DIRECTION::UP;
      dir1 = DIRECTION::DOWN;
    } else if (pl[0] == 2 and pl[1] == 0) {
      shift = Shift(1, 0);
      dir0 = DIRECTION::DOWN;
      dir1 = DIRECTION::UP;
    } else if (pl[0] == 1 and pl[1] == 3) {
      shift = Shift(0, -1);
      dir0 = DIRECTION::RIGHT;
      dir1 = DIRECTION::LEFT;
    } else if (pl[0] == 3 and pl[1] == 1) {
      shift = Shift(0, 1);
      dir0 = DIRECTION::LEFT;
      dir1 = DIRECTION::RIGHT;
    } else
      throw std::runtime_error("[fullUpdate_ALS2S_IT] Invalid gate");
    // 2) find the vertex of tn[0]
    auto v0 = cls.idToV.at(tn[0]);
    auto v1 = v0 + shift;

    pc[0] *= delta(ctmEnv.tauxByVertex(dir0, v0, pl[0]),
                   ctmEnv.tauxByVertex(dir0, v1, pl[1]));
    pc[1] *= delta(ctmEnv.tauxByVertex(dir1, v0, pl[0]),
                   ctmEnv.tauxByVertex(dir1, v1, pl[1]));

    if (dbg && (dbgLvl >= 3))
      for (int i = 0; i < 4; i++)
        Print(pc[i]);

    // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE *******************

    // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
    t_begin_int = std::chrono::steady_clock::now();

    // A--B
    // Decompose A and B tensor on which the gate is applied
    {
      // TODO mask ? cutoff ?
      // Internally ITensor takes sqrt of singular values
      auto maskS = [&machine_eps](Real r) {
        return (r > 10.0 * machine_eps) ? 1.0 : 1.0;
      };
      auto cutoffS = [&machine_eps](Real r) {
        return (r > 10.0 * machine_eps) ? r : r;
      };

      ITensor tempSA;
      svd(cls.sites.at(tn[0]), eA, tempSA, QA, {"Truncate", false});

      // has to be non-zero
      tempSA *= 1.0 / tempSA.real(tempSA.inds()[0](1), tempSA.inds()[1](1));
      tempSA.apply(cutoffS);
      auto maskSA = tempSA;
      maskSA.apply(maskS);
      auto tmpI_SA = commonIndex(tempSA, eA);

      iQA = Index("auxQA", commonIndex(QA, tempSA).m(), AUXLINK, 0);
      eA = (eA * tempSA) * delta(commonIndex(QA, tempSA), iQA);
      QA *= maskSA;
      QA *= delta(tmpI_SA, iQA);
      // QA *= delta(commonIndex(QA,tempSA), iQA);

      // Prepare corner of A
      ITensor tempC = (pc[0] * QA) * prime(conj(QA), AUXLINK, 4);
      if (dbg && (dbgLvl >= 3))
        Print(tempC);

      eRE = tempC * pc[1];
      if (dbg && (dbgLvl >= 3))
        Print(eRE);

      eRE *= pc[3];
      if (dbg && (dbgLvl >= 3))
        Print(eRE);

      // Decompose B tensor on which the gate is applied
      ITensor tempSB;
      svd(cls.sites.at(tn[1]), eB, tempSB, QB, {"Truncate", false});

      tempSB *= 1.0 / tempSB.real(tempSB.inds()[0](1), tempSB.inds()[1](1));
      tempSB.apply(cutoffS);
      auto maskSB = tempSB;
      maskSB.apply(maskS);
      auto tmpI_SB = commonIndex(tempSB, eB);

      iQB = Index("auxQB", commonIndex(QB, tempSB).m(), AUXLINK, 0);
      eB = (eB * tempSB) * delta(commonIndex(QB, tempSB), iQB);
      QB *= maskSB;
      QB *= delta(tmpI_SB, iQB);
      // QB *= delta(commonIndex(QB,tempSB), iQB);

      tempC = (pc[2] * QB) * prime(conj(QB), AUXLINK, 4);
      if (dbg && (dbgLvl >= 3))
        Print(tempC);

      eRE *= tempC;
    }

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Constructed reduced Env - T: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                   t_end_int - t_begin_int)
                     .count() /
                   1000000.0
              << " [sec]" << std::endl;
    if (dbg && (dbgLvl >= 3))
      Print(eRE);
    // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
  }

  double condNum = 1.0;
  std::string diag_protoEnv, diag_protoEnv_descriptor;
  double diag_maxMsymLE, diag_maxMasymLE;
  double diag_maxMsymFN, diag_maxMasymFN;
  if (symmProtoEnv) {
    // ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
    t_begin_int = std::chrono::steady_clock::now();
    auto cmbKet = combiner(iQA, iQB);
    auto cmbBra = prime(cmbKet, 4);

    eRE = (eRE * cmbKet) * cmbBra;

    ITensor eRE_sym = 0.5 * (eRE + swapPrime(eRE, 0, 4));
    ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE, 0, 4));

    m = 0.;
    eRE_sym.visit(max_m);
    diag_maxMsymLE = m;
    std::cout << "eRE_sym max element: " << m << std::endl;
    m = 0.;
    eRE_asym.visit(max_m);
    diag_maxMasymLE = m;
    std::cout << "eRE_asym max element: " << m << std::endl;

    diag_maxMsymFN = norm(eRE_sym);
    diag_maxMasymFN = norm(eRE_asym);

    if (posDefProtoEnv) {
      eRE_sym *= delta(combinedIndex(cmbBra), prime(combinedIndex(cmbKet)));

      // ##### V3 ######################################################
      ITensor U_eRE, D_eRE;
      diagHermitian(eRE_sym, U_eRE, D_eRE);

      double msign = 1.0;
      double mval = 0.;
      double nval = 1.0e+16;
      std::vector<double> dM_elems;
      for (int idm = 1; idm <= D_eRE.inds().front().m(); idm++) {
        dM_elems.push_back(
          D_eRE.real(D_eRE.inds().front()(idm), D_eRE.inds().back()(idm)));
        if (std::abs(dM_elems.back()) > mval) {
          mval = std::abs(dM_elems.back());
          msign = dM_elems.back() / mval;
        }
        // find the lowest eigenvalue in magnitude
        if (std::abs(dM_elems.back()) < nval)
          nval = std::abs(dM_elems.back());
      }
      if (msign < 0.0)
        for (auto& elem : dM_elems)
          elem = elem * (-1.0);

      // Drop negative EV's and count negative EVs, EVs lower than cutoff
      double traceDM = 0.0;
      int countCTF = 0;
      int countNEG = 0;
      if (dbg && (dbgLvl >= 1)) {
        std::cout << "REFINED SPECTRUM" << std::endl;
        std::cout << "MAX EV: " << mval << std::endl;
      }
      for (auto& elem : dM_elems) {
        if (elem < 0.0) {
          if (dbg && (dbgLvl >= 2))
            std::cout << elem << " -> " << 0.0 << std::endl;
          elem = 0.0;
          // elem = 1.25e-4;
          countNEG += 1;
        } else if (elem < svd_cutoff) {
          countCTF += 1;
          if (dbg && (dbgLvl >= 2))
            std::cout << elem << std::endl;
        }
      }

      condNum = mval / nval;

      std::ostringstream oss;
      oss << std::scientific << mval << " " << condNum << " " << countCTF << " "
          << countNEG << " " << dM_elems.size();

      diag_protoEnv_descriptor = "MaxEV condNum EV<CTF EV<0 TotalEV";
      diag_protoEnv = oss.str();

      if (dbg && (dbgLvl >= 1)) {
        std::cout << "REFINED SPECTRUM" << std::endl;
        std::cout << std::scientific << "MAX EV: " << mval
                  << " MIN EV: " << nval << std::endl;
        std::cout << "RATIO svd_cutoff/negative/all " << countCTF << "/"
                  << countNEG << "/" << dM_elems.size() << std::endl;
      }
      // ##### END V3 ##################################################

      D_eRE = diagTensor(dM_elems, D_eRE.inds().front(), D_eRE.inds().back());
      // D_eRE = D_eRE / traceDM;

      eRE_sym = ((conj(U_eRE) * D_eRE) * prime(U_eRE)) *
                delta(combinedIndex(cmbBra), prime(combinedIndex(cmbKet)));

      // eRE_sym *= 1.0/mval;
    }

    eRE = (eRE_sym * cmbKet) * cmbBra;

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Symmetrized reduced env - T: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                   t_end_int - t_begin_int)
                     .count() /
                   1000000.0
              << " [sec]" << std::endl;
    // ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
  }

  // ***** FORM "PROTO" ENVIRONMENTS FOR M and K *****************************
  t_begin_int = std::chrono::steady_clock::now();

  ITensor protoK =
    (eRE * eA) * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1]));
  protoK *= eB;

  {
    auto tempOp = mpo.H1 * mpo.H2;
    tempOp = (tempOp * delta(mpo.Is1, phys[0])) * delta(mpo.Is2, phys[1]);
    tempOp = (tempOp * prime(delta(mpo.Is1, phys[0]))) *
             prime(delta(mpo.Is2, phys[1]));

    protoK *= tempOp;
    protoK.noprime(PHYS);
  }
  if (dbg && (dbgLvl >= 3))
    Print(protoK);

  std::cout << "eRE.scale(): " << eRE.scale()
            << " protoK.scale(): " << protoK.scale() << std::endl;
  t_end_int = std::chrono::steady_clock::now();
  std::cout << "Proto Envs for M and K constructed - T: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                 t_end_int - t_begin_int)
                   .count() /
                 1000000.0
            << " [sec]" << std::endl;
  // ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************

  // ********************************************************************************************
  // 	     OPTIMIZE VIA CG *
  // ********************************************************************************************
  t_begin_int = std::chrono::steady_clock::now();

  // <psi|U^dag U|psi>
  double normUPsi;
  {
    auto tempOp = conj(mpo.H1) * conj(mpo.H2);
    tempOp = (tempOp * delta(mpo.Is1, phys[0])) * delta(mpo.Is2, phys[1]);
    tempOp = (tempOp * prime(delta(mpo.Is1, phys[0]))) *
             prime(delta(mpo.Is2, phys[1]));

    auto NORMUPSI = protoK * tempOp;
    NORMUPSI.noprime(PHYS);
    NORMUPSI *= prime(conj(eA), AUXLINK, 4);
    NORMUPSI *= prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4);
    NORMUPSI *= prime(conj(eB), AUXLINK, 4);

    if (NORMUPSI.r() > 0)
      std::cout << "ERROR: NORMUPSI rank > 0" << std::endl;
    normUPsi = sumels(NORMUPSI);
  }

  {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    // reduced tensors
    auto tmpT = eA * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])) * eB;

    ITensor tmpEA(iQA, phys[0]), S, tmpEB;
    svd(tmpT, tmpEA, S, tmpEB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
    std::cout << "RED_SPEC: ";
    S.visit(printS);
    std::cout << std::endl;

    // full tensors
    tmpT = QA * tmpT * QB;

    std::vector<Index> indsA;
    for (int i = 0; i < 4; i++)
      if (i != pl[0])
        indsA.push_back(cls.AIc(tn[0], i));
    indsA.push_back(phys[0]);

    ITensor tmpA(indsA), S2, tmpB;
    svd(tmpT, tmpA, S2, tmpB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S2 *= 1.0 / S2.real(S2.inds()[0](1), S2.inds()[1](1));
    std::cout << "FULL_SPEC: ";
    S2.visit(printS);
    std::cout << std::endl;
  }

  std::vector<double> fdist, fdistN, vec_normPsi;

  // compute intitial norm, fdist, normalized fdist
  auto NORMPSI =
    (eRE * eB) * delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0]));
  NORMPSI *= (prime(conj(eB), AUXLINK, 4) *
              prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4));
  NORMPSI = (prime(conj(eA), AUXLINK, 4) * NORMPSI) * eA;

  auto OVERLAP = protoK * prime(conj(eB), AUXLINK, 4);
  OVERLAP *= prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4);
  OVERLAP = prime(conj(eA), AUXLINK, 4) * OVERLAP;

  if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
    std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
  double t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
  double t_fdistN =
    1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) + 1.0;

  fdist.push_back(t_fdist);
  fdistN.push_back(t_fdistN);
  vec_normPsi.push_back(sumels(NORMPSI));

  // Solve for eA*eB
  auto M = (eRE * delta(phys[0], prime(phys[0], 4))) *
           delta(phys[1], prime(phys[1], 4));
  auto K = prime(protoK, PHYS, 4);

  auto cmb0 = combiner(iQA, iQB, phys[0], phys[1]);
  auto cmb1 = combiner(prime(iQA, 4), prime(iQB, 4), prime(phys[0], 4),
                       prime(phys[1], 4));
  M = (cmb0 * M) * cmb1;
  K *= cmb1;
  ITensor eAeB(combinedIndex(cmb0));
  eAeB.fill(0.0);

  Print(M);
  Print(K);
  Print(eAeB);

  linsystem(M, K, eAeB, ls, args);

  eAeB *= cmb0;

  // svd on eAeB
  ITensor tempS, tmp_eA(iQA, phys[0]), tmp_eB;
  factor(eAeB, tmp_eA, tmp_eB, {"Minm", cls.AIc(tn[0], pl[0]).m()});
  auto cmnI = commonIndex(tmp_eA, tmp_eB);
  eA = tmp_eA * delta(cmnI, cls.AIc(tn[0], pl[0]));
  eB = tmp_eB * delta(cmnI, cls.AIc(tn[1], pl[1]));

  std::cout << "ENTERING ALS LOOP" << std::endl;
  int altlstsquares_iter = 0;
  bool converged = false;
  t_begin_int = std::chrono::steady_clock::now();
  while (not converged) {
    // Optimizing eA
    // 1) construct matrix M, which is defined as <psi~|psi~> = eA^dag * M * eA
    {
      ITensor M =
        (eRE * eB) * delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0]));
      M *= (prime(conj(eB), AUXLINK, 4) *
            prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4));

      ITensor K = protoK * prime(conj(eB), AUXLINK, 4);
      K *= prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4);

      // <psi'|psi'>
      auto NORMPSI = (prime(conj(eA), AUXLINK, 4) * M) * eA;
      // <psi'|U|psi>
      auto OVERLAP = prime(conj(eA), AUXLINK, 4) * K;

      if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
        std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
      double t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
      double t_fdistN =
        1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) +
        1.0;

      fdist.push_back(t_fdist);
      fdistN.push_back(t_fdistN);
      vec_normPsi.push_back(sumels(NORMPSI));

      // condition for stopping ALS procedure
      if (fdist.back() < 1.0e-08) {
        converged = true;
        break;
      }
      if ((fdist.size() > 1) &&
          std::abs(fdist.back() - fdist[fdist.size() - 2]) / fdist[0] <
            epsdistf) {
        converged = true;
        break;
      }

      auto RES = M * eA - K;
      std::cout << "Norm(RES_A)= " << norm(RES) << std::endl;

      // eA: aux, aux, phys
      // K : aux^offset, aux^offset, phys^offset
      M *= delta(phys[0], prime(phys[0], 4));
      K.prime(PHYS, 4);

      auto cmb0 = combiner(iQA, cls.AIc(tn[0], pl[0]), phys[0]);
      auto cmb1 = combiner(prime(iQA, 4), prime(cls.AIc(tn[0], pl[0]), 4),
                           prime(phys[0], 4));
      M = (cmb0 * M) * cmb1;
      // regularize Hessian
      // std::vector<double> eps_reg(combinedIndex(cmb0, epsregularisation));
      // M += diagTensor(eps_reg, combinedIndex(cmb0), combinedIndex(cmb1));
      K *= cmb1;
      eA *= cmb0;

      linsystem(M, K, eA, ls, args);

      eA *= cmb0;
    }

    // Optimizing eB
    // 1) construct matrix M, which is defined as <psi~|psi~> = eB^dag * M * eB
    {
      ITensor M =
        (eRE * eA) * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1]));
      M *= (prime(conj(eA), AUXLINK, 4) *
            prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4));

      ITensor K = protoK * prime(conj(eA), AUXLINK, 4);
      K *= prime(delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])), 4);

      // <psi'|psi'>
      auto NORMPSI = (prime(conj(eB), AUXLINK, 4) * M) * eB;
      // <psi'|U|psi>
      auto OVERLAP = prime(conj(eB), AUXLINK, 4) * K;

      if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
        std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
      double t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
      double t_fdistN =
        1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) +
        1.0;

      fdist.push_back(t_fdist);
      fdistN.push_back(t_fdistN);
      vec_normPsi.push_back(sumels(NORMPSI));

      // condition for stopping ALS procedure
      if (fdist.back() < 1.0e-08) {
        converged = true;
        break;
      }
      if ((fdist.size() > 1) &&
          std::abs(fdist.back() - fdist[fdist.size() - 2]) / fdist[0] <
            epsdistf) {
        converged = true;
        break;
      }
      auto RES = M * eB - K;
      std::cout << "Norm(RES_B)= " << norm(RES) << std::endl;

      M *= delta(phys[1], prime(phys[1], 4));
      K.prime(PHYS, 4);

      auto cmb0 = combiner(iQB, cls.AIc(tn[1], pl[1]), phys[1]);
      auto cmb1 = combiner(prime(iQB, 4), prime(cls.AIc(tn[1], pl[1]), 4),
                           prime(phys[1], 4));
      M = (cmb0 * M) * cmb1;
      K *= cmb1;
      eB *= cmb0;

      linsystem(M, K, eB, ls, args);

      eB *= cmb0;
    }

    altlstsquares_iter++;
    if (altlstsquares_iter >= maxAltLstSqrIter)
      converged = true;
  }
  t_end_int = std::chrono::steady_clock::now();

  Print(eA);
  Print(eB);

  // recompute norm, fdist, normalized fdist
  NORMPSI = (eRE * eB) * delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0]));
  NORMPSI *= (prime(conj(eB), AUXLINK, 4) *
              prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4));
  NORMPSI = (prime(conj(eA), AUXLINK, 4) * NORMPSI) * eA;

  OVERLAP = protoK * prime(conj(eB), AUXLINK, 4);
  OVERLAP *= prime(delta(cls.AIc(tn[1], pl[1]), cls.AIc(tn[0], pl[0])), 4);
  OVERLAP = prime(conj(eA), AUXLINK, 4) * OVERLAP;

  if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
    std::cout << "ERROR: NORMPSI or OVERLAP rank > 0" << std::endl;
  t_fdist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
  t_fdistN =
    1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(sumels(NORMPSI) * normUPsi) + 1.0;

  fdist.push_back(t_fdist);
  fdistN.push_back(t_fdistN);
  vec_normPsi.push_back(sumels(NORMPSI));

  std::cout << "STEP f=||psi'>-|psi>|^2 normalized(f) norm(psi') norm(psi)"
            << std::endl;
  for (int i = 0; i < fdist.size(); i++)
    std::cout << i << " " << fdist[i] << " " << fdistN[i] << " "
              << vec_normPsi[i] << " " << normUPsi << std::endl;

  {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    // reduced tensors
    auto tmpT = eA * delta(cls.AIc(tn[0], pl[0]), cls.AIc(tn[1], pl[1])) * eB;

    ITensor tmpEA(iQA, phys[0]), S, tmpEB;
    svd(tmpT, tmpEA, S, tmpEB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
    std::cout << "RED_SPEC: ";
    S.visit(printS);
    std::cout << std::endl;

    // full tensors
    tmpT = QA * tmpT * QB;

    std::vector<Index> indsA;
    for (int i = 0; i < 4; i++)
      if (i != pl[0])
        indsA.push_back(cls.AIc(tn[0], i));
    indsA.push_back(phys[0]);

    ITensor tmpA(indsA), S2, tmpB;
    svd(tmpT, tmpA, S2, tmpB,
        {"Minm", cls.AIc(tn[0], pl[0]).m(), "Maxm", cls.AIc(tn[0], pl[0]).m()});

    S2 *= 1.0 / S2.real(S2.inds()[0](1), S2.inds()[1](1));
    std::cout << "FULL_SPEC: ";
    S2.visit(printS);
    std::cout << std::endl;
  }

  // BALANCE tensors
  double iso_tot_mag = 1.0;
  m = 0.;
  eA.visit(max_m);
  eA = eA / m;
  iso_tot_mag = iso_tot_mag * m;
  eB.visit(max_m);
  eB = eB / m;
  iso_tot_mag = iso_tot_mag * m;
  eA = eA * std::pow(iso_tot_mag, (1.0 / 2.0));
  eB = eB * std::pow(iso_tot_mag, (1.0 / 2.0));

  // update on-site tensors of cluster
  cls.sites.at(tn[0]) = QA * eA;
  cls.sites.at(tn[1]) = QB * eB;

  // max element of on-site tensors
  // or norm-distance of new vs original tensors
  std::string diag_maxElem;
  std::ostringstream oss_diag_siteScale;
  for (int i = 0; i < 2; i++) {
    m = 0.;
    m = norm(cls.sites.at(tn[i]) - orig_tensors[i]);
    // site_e.second.visit(max_m);
    diag_maxElem = diag_maxElem + tn[i] + " : " + std::to_string(m) + " ";
    oss_diag_siteScale << tn[i] << " : " << cls.sites.at(tn[i]).scale() << " ";
  }
  std::cout << oss_diag_siteScale.str() << std::endl;
  std::cout << diag_maxElem << std::endl;

  // normalize updated tensors
  if (otNormType == "BLE") {
    for (int i = 0; i < tn.size(); i++) {
      m = 0.;
      cls.sites.at(tn[i]).visit(max_m);
      cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / sqrt(m);
    }
  } else if (otNormType == "BALANCE") {
    // double iso_tot_mag = 1.0;
    //    for ( auto & site_e : cls.sites)  {
    //    	m = 0.;
    // 	site_e.second.visit(max_m);
    //    	site_e.second = site_e.second / m;
    //    	iso_tot_mag = iso_tot_mag * m;
    //    }
    //    for (auto & site_e : cls.sites) {
    //    	site_e.second = site_e.second * std::pow(iso_tot_mag, (1.0/(2.0 *
    //    cls.sites.size())) );
    //    }
    for (auto& site_e : cls.sites) {
      m = 0.;
      site_e.second.visit(max_m);
      site_e.second = site_e.second / m;
    }
    for (auto& site_e : cls.sites) {
      site_e.second =
        site_e.second /
        std::pow(vec_normPsi.back(), (1.0 / (2.0 * cls.sizeN * cls.sizeM)));
    }
  } else if (otNormType == "NONE") {
  } else {
    std::cout << "Unsupported on-site tensor normalisation after full update: "
              << otNormType << std::endl;
    exit(EXIT_FAILURE);
  }

  // max element of on-site tensors after normalization
  for (int i = 0; i < tn.size(); i++) {
    m = 0.;
    cls.sites.at(tn[i]).visit(max_m);
    std::cout << tn[i] << " : " << std::to_string(m) << " ";
  }
  std::cout << std::endl;

  t_end_int = std::chrono::steady_clock::now();
  // ***** END OPTIMIZATION *****

  // prepare and return diagnostic data
  diag_data.add("alsSweep", 0);

  std::string siteMaxElem_descriptor = "site max_elem site max_elem";
  diag_data.add("siteMaxElem_descriptor", siteMaxElem_descriptor);
  diag_data.add("siteMaxElem", diag_maxElem);

  diag_data.add("ratioNonSymLE",
                diag_maxMasymLE / diag_maxMsymLE);  // ratio of largest elements
  diag_data.add("ratioNonSymFN",
                diag_maxMasymFN / diag_maxMsymFN);  // ratio of norms

  std::ostringstream oss;
  oss << std::scientific << fdist[0] << " " << fdist.back() << " " << fdistN[0]
      << " " << fdistN.back() << " " << vec_normPsi[0] << " "
      << vec_normPsi.back() << " "
      << std::chrono::duration_cast<std::chrono::microseconds>(t_end_int -
                                                               t_begin_int)
             .count() /
           1000000.0;

  std::string logMinDiag_descriptor =
    "f_init f_final nf_init nf_final norm(psi')_init norm(psi')_final time[s]";
  diag_data.add("locMinDiag_descriptor", logMinDiag_descriptor);
  diag_data.add("locMinDiag", oss.str());

  if (symmProtoEnv) {
    diag_data.add("diag_protoEnv", diag_protoEnv);
    diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
  }

  return diag_data;
}
