#include "pi-peps/config.h"
// #include "full-update-TEST.h"
#include "pi-peps/full-update.h"

using namespace itensor;

Args fullUpdate_ALS3S_IT(MPO_3site const& uJ1J2,
                         Cluster& cls,
                         CtmEnv const& ctmEnv,
                         std::vector<std::string> tn,
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
  auto linsolver = args.getString("linsolver", "default");
  auto otNormType = args.getString("otNormType");

  double machine_eps = std::numeric_limits<double>::epsilon();
  if (dbg && (dbgLvl >= 1))
    std::cout << "M EPS: " << machine_eps << std::endl;

  std::chrono::steady_clock::time_point t_begin_int, t_end_int;

  // prepare to hold diagnostic data
  Args diag_data = Args::global();
  std::vector<ITensor> orig_tensors = {cls.sites.at(tn[0]), cls.sites.at(tn[1]),
                                       cls.sites.at(tn[2])};

  if (dbg) {
    std::cout << "GATE: ";
    for (int i = 0; i <= 3; i++) {
      std::cout << ">-" << pl[2 * i] << "-> " << tn[i] << " >-" << pl[2 * i + 1]
                << "->";
    }
    std::cout << std::endl;

    if (dbgLvl >= 2) {
      std::cout << uJ1J2;
      PrintData(uJ1J2.H1);
      PrintData(uJ1J2.H2);
      PrintData(uJ1J2.H3);
    }
  }

  // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
  double m = 0.;
  auto max_m = [&m](double d) {
    if (std::abs(d) > m)
      m = std::abs(d);
  };

  // read off auxiliary and physical indices of the cluster sites
  // std::array<Index, 4> aux;
  // for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

  std::array<Index, 4> phys;
  for (int i = 0; i < 4; i++)
    phys[i] = cls.mphys.at(tn[i]);

  std::array<Index, 3> opPI({noprime(findtype(uJ1J2.H1, PHYS)),
                             noprime(findtype(uJ1J2.H2, PHYS)),
                             noprime(findtype(uJ1J2.H3, PHYS))});

  if (dbg) {
    std::cout << "On-site indices:" << std::endl;
    for (int i = 0; i < 4; i++) {
      // std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
      std::cout << tn[i] << " : " << phys[i] << std::endl;
    }
  }

  Index iQA, iQD, iQB;
  ITensor QA, eA(cls.AIc(tn[0], pl[1]), phys[0]);
  ITensor QD, eD(cls.AIc(tn[2], pl[4]), phys[2]);
  ITensor QB, eB(cls.AIc(tn[1], pl[2]), cls.AIc(tn[1], pl[3]), phys[1]);

  ITensor eRE;
  ITensor deltaBra, deltaKet;

  {
    t_begin_int = std::chrono::steady_clock::now();

    auto getEdgeCombiners = [](ITensor& cmb_v0, ITensor& cmb_v1,
                               CtmEnv const& ctmEnv, DIRECTION direction,
                               Vertex const& v0, int dir0, Vertex const& v1,
                               int dir1, bool full) {
      if (full) {
        cmb_v0 = combiner(ctmEnv.tauxByVertex(direction, v0, dir0),
                          ctmEnv.p_cluster->AIc(v0, dir0),
                          prime(ctmEnv.p_cluster->AIc(v0, dir0),
                                ctmEnv.p_cluster->BRAKET_OFFSET));
        cmb_v1 = combiner(ctmEnv.tauxByVertex(direction, v1, dir1),
                          ctmEnv.p_cluster->AIc(v1, dir1),
                          prime(ctmEnv.p_cluster->AIc(v1, dir1),
                                ctmEnv.p_cluster->BRAKET_OFFSET));
      } else {
        cmb_v0 = combiner(ctmEnv.tauxByVertex(direction, v0, dir0));
        cmb_v1 = combiner(ctmEnv.tauxByVertex(direction, v1, dir1));
      }
    };

    auto getEdgeCombiners_fromTnAndPl = [&getEdgeCombiners, &tn, &pl](
                                          ITensor& cmb_v0, ITensor& cmb_v1,
                                          CtmEnv const& ctmEnv, int s0, int s1,
                                          bool full) {
      Vertex v0 = ctmEnv.p_cluster->idToV.at(tn[s0]);
      Vertex v1 = ctmEnv.p_cluster->idToV.at(tn[s1]);

      DIRECTION dir_outgoing_s0 = toDIRECTION(pl[2 * s0 + 1]);
      DIRECTION dir_ingoing_s1 = toDIRECTION(pl[2 * s1]);

      DIRECTION edge_dir = toDIRECTION((pl[2 * s0] + 2) % 4);

      getEdgeCombiners(cmb_v0, cmb_v1, ctmEnv, edge_dir, v0, dir_outgoing_s0,
                       v1, dir_ingoing_s1, full);
    };

    // prepare map from on-site tensor aux-indices to half row/column T
    // environment tensors
    std::array<const std::map<std::string, ITensor>* const, 4> iToT(
      {&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R, &ctmEnv.T_D});

    // prepare map from on-site tensor aux-indices pair to half corner T-C-T
    // environment tensors
    const std::map<int, const std::map<std::string, ITensor>* const> iToC(
      {{23, &ctmEnv.C_LU},
       {32, &ctmEnv.C_LU},
       {21, &ctmEnv.C_LD},
       {12, &ctmEnv.C_LD},
       {3, &ctmEnv.C_RU},
       {30, &ctmEnv.C_RU},
       {1, &ctmEnv.C_RD},
       {10, &ctmEnv.C_RD}});

    // precompute 4 (proto)corners of 2x2 environment
    std::vector<ITensor> pc(4);
    for (int s = 0; s <= 3; s++) {
      // taking opposite directions gives previous and current edge direction
      //
      //                                   current edge                 previous
      //                                   edge
      // >pl[2s]->-tn[s]->-pl[2s+1]> ==> (pl[2s] + 2) % 4 -- tn[s] -- (pl[2s+1]
      // + 2) % 4
      //
      // Ex.         ((3+2)%4=1) current edge
      //              |
      // ((2+2)%4=0)--tn[s]=A--pl[2s+1]=2
      //  prev. edge  |
      //              pl[2s]=3

      // aux-indices connected to sites
      //                         incoming   outgoing
      std::vector<int> connected({pl[s * 2], pl[s * 2 + 1]});
      //                           current-edge         previous-edge
      std::vector<int> tmp_iToE({(pl[s * 2] + 2) % 4, (pl[s * 2 + 1] + 2) % 4});
      tmp_iToE.push_back(pl[s * 2] * 10 +
                         pl[s * 2 + 1]);  // identifier for C ENV tensor
      if (dbg) {
        std::cout << "primeLevels (pl) of indices connected to ENV - site: "
                  << tn[s] << std::endl;
        std::cout << tmp_iToE[0] << " " << tmp_iToE[1]
                  << " iToC: " << tmp_iToE[2] << std::endl;
      }

      pc[s] = (*iToT.at(tmp_iToE[0])).at(tn[s]) *
              (*iToC.at(tmp_iToE[2])).at(tn[s]) *
              (*iToT.at(tmp_iToE[1])).at(tn[s]);
      if (dbg)
        Print(pc[s]);
    }

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Constructed proto Corners (without on-site tensors): "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                   t_end_int - t_begin_int)
                     .count() /
                   1000000.0
              << " [sec]" << std::endl;
    // ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE *******************

    // ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
    t_begin_int = std::chrono::steady_clock::now();

    // tn[3](C)  tn[2](D)
    //           |
    // tn[0](A)--tn[1](B)
    // ITensor eRE;
    ITensor cmb0, cmb1, cmb2, cmb3;

    // TODO mask ? cutoff ?
    auto maskS = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? 1.0 : 1.0;
    };
    auto cutoffS = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? r : r;
    };

    // Decompose A tensor on which the gate is applied
    // ITensor QA, tempSA, eA(cls.AIc(tn[0],pl[1]), phys[0]);
    ITensor tempSA;
    svd(cls.sites.at(tn[0]), eA, tempSA, QA, {"Truncate", false});

    tempSA *= 1.0 / tempSA.real(tempSA.inds()[0](1), tempSA.inds()[1](1));
    tempSA.apply(cutoffS);
    auto maskSA = tempSA;
    maskSA.apply(maskS);
    auto tmpI_SA = commonIndex(tempSA, eA);

    // Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
    iQA = Index("auxQA", commonIndex(QA, tempSA).m(), AUXLINK, 0);
    eA = (eA * tempSA) * delta(commonIndex(QA, tempSA), iQA);
    QA *= maskSA;
    QA *= delta(tmpI_SA, iQA);
    // QA *= delta(commonIndex(QA,tempSA), iQA);

    // Prepare corner of A
    // ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
    ITensor tempC = (pc[0] * QA);
    tempC *= prime(conj(QA), AUXLINK, 4);

    if (dbg && (dbgLvl >= 3))
      Print(tempC);

    // deltaKet = delta(cls.AIc(tn[0],pl[0]), cls.AIc(tn[3],pl[7]));
    // deltaBra = prime(deltaKet,4);

    // tempC = (tempC * deltaBra) * deltaKet;
    // if(dbg && (dbgLvl >=3)) Print(tempC);

    eRE = tempC;

    // Prepare corner of C
    // tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >=
    // 3)));
    tempC = pc[3] * cls.sites.at(tn[3]);
    tempC *= prime(conj(cls.sites.at(tn[3])), AUXLINK, 4);
    if (dbg && (dbgLvl >= 3))
      Print(tempC);

    // deltaKet = delta(cls.AIc(tn[3],pl[6]), cls.AIc(tn[2],pl[5]));
    // deltaBra = prime(deltaKet,4);
    // tempC = (tempC * deltaBra) * deltaKet;
    // if(dbg && (dbgLvl >=3)) Print(tempC);

    // TODO use delta instead of reindex
    getEdgeCombiners_fromTnAndPl(cmb1, cmb0, ctmEnv, 3, 0, true);
    eRE *= cmb0;
    eRE = reindex(eRE, combinedIndex(cmb0), combinedIndex(cmb1));
    eRE *= (tempC * cmb1);
    if (dbg && (dbgLvl >= 3)) {
      Print(cmb0);
      Print(cmb1);
      Print(eRE);
    }

    // Decompose D tensor on which the gate is applied
    // ITensor QD, tempSD, eD(cls.AIc(tn[2],pl[4]), phys[2]);
    ITensor tempSD;
    svd(cls.sites.at(tn[2]), eD, tempSD, QD, {"Truncate", false});

    tempSD *= 1.0 / tempSD.real(tempSD.inds()[0](1), tempSD.inds()[1](1));
    tempSD.apply(cutoffS);
    auto maskSD = tempSD;
    maskSD.apply(maskS);
    auto tmpI_SD = commonIndex(tempSD, eD);

    // Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
    iQD = Index("auxQD", commonIndex(QD, tempSD).m(), AUXLINK, 0);
    eD = (eD * tempSD) * delta(commonIndex(QD, tempSD), iQD);
    QD *= maskSD;
    QD *= delta(tmpI_SD, iQD);
    // QD *= delta(commonIndex(QD,tempSD), iQD);

    // Prepare corner of D
    // tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
    tempC = pc[2] * QD;
    tempC *= prime(conj(QD), AUXLINK, 4);
    if (dbg && (dbgLvl >= 3))
      Print(tempC);

    getEdgeCombiners_fromTnAndPl(cmb1, cmb0, ctmEnv, 2, 3, true);
    eRE *= cmb0;
    eRE = reindex(eRE, combinedIndex(cmb0), combinedIndex(cmb1));
    eRE *= (tempC * cmb1);
    if (dbg && (dbgLvl >= 3)) {
      Print(cmb0);
      Print(cmb1);
      Print(eRE);
    }

    // Decompose B tensor on which the gate is applied
    // ITensor QB, tempSB, eB(cls.AIc(tn[1],pl[2]), cls.AIc(tn[1],pl[3]),
    // phys[1]);
    ITensor tempSB;
    svd(cls.sites.at(tn[1]), eB, tempSB, QB, {"Truncate", false});

    tempSB *= 1.0 / tempSB.real(tempSB.inds()[0](1), tempSB.inds()[1](1));
    tempSB.apply(cutoffS);
    auto maskSB = tempSB;
    maskSB.apply(maskS);
    auto tmpI_SB = commonIndex(tempSB, eB);

    // Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
    iQB = Index("auxQB", commonIndex(QB, tempSB).m(), AUXLINK, 0);
    eB = (eB * tempSB) * delta(commonIndex(QB, tempSB), iQB);
    QB *= maskSB;
    QB *= delta(tmpI_SB, iQB);
    // QB *= delta(commonIndex(QB,tempSB), iQB);

    // tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
    tempC = pc[1] * QB;
    tempC *= prime(conj(QB), AUXLINK, 4);
    if (dbg && (dbgLvl >= 3))
      Print(tempC);

    getEdgeCombiners_fromTnAndPl(cmb3, cmb2, ctmEnv, 1, 2, false);
    getEdgeCombiners_fromTnAndPl(cmb0, cmb1, ctmEnv, 0, 1, false);
    eRE = (eRE * cmb2) * cmb0;
    eRE = reindex(eRE, combinedIndex(cmb2), combinedIndex(cmb3),
                  combinedIndex(cmb0), combinedIndex(cmb1));
    eRE *= (tempC * cmb3) * cmb1;

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

  double diag_maxMsymLE, diag_maxMasymLE;
  double diag_maxMsymFN, diag_maxMasymFN;
  std::string diag_protoEnv, diag_protoEnv_descriptor;
  double condNum = 1.0;
  if (symmProtoEnv) {
    // ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
    t_begin_int = std::chrono::steady_clock::now();
    auto cmbKet = combiner(iQA, iQB, iQD);
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

      // find largest and smallest eigenvalues
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
        if (std::abs(dM_elems.back()) < nval)
          nval = std::abs(dM_elems.back());
      }
      if (msign < 0.0)
        for (auto& elem : dM_elems)
          elem = elem * (-1.0);

      // Drop negative EV'std
      int countCTF = 0;
      int countNEG = 0;
      for (auto& elem : dM_elems) {
        if (elem < 0.0) {
          if (dbg && (dbgLvl >= 1))
            std::cout << elem << " -> " << 0.0 << std::endl;
          elem = 0.0;
          countNEG += 1;
          //} else if (elem/mval < svd_cutoff) {
          //	countCTF += 1;
          // elem = 0.0;
          //	if(dbg && (dbgLvl >= 2)) std::cout<< elem <<" -> "<< 0.0 <<
          // std::endl;
        }
      }

      // TODO? estimate codition number
      // condNum = ( std::abs(nval/mval) > svd_cutoff ) ? std::abs(mval/nval)
      // : 1.0/svd_cutoff ;
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

      // ##### V4 ######################################################
      // ITensor U_eRE, D_eRE;
      // diagHermitian(eRE_sym, U_eRE, D_eRE);

      // double msign = 1.0;
      // double mval = 0.;
      // std::vector<double> dM_elems;
      // for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {
      // 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
      // 	if (std::abs(dM_elems.back()) > mval) {
      // 		mval = std::abs(dM_elems.back());
      // 		msign = dM_elems.back()/mval;
      // 	}
      // }
      // if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

      // // Set EV's to ABS Values
      // if(dbg && (dbgLvl >= 1)) {
      // 	std::cout<<"REFINED SPECTRUM"<< std::endl;
      // 	std::cout<<"MAX EV: "<< mval << std::endl;
      // }
      // for (auto & elem : dM_elems) elem = std::fabs(elem);
      // ##### END V4 ##################################################

      // ##### V5 ######################################################
      // eRE *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet))); //
      // 0--eRE--1

      // eRE_sym = conj(eRE); // 0--eRE*--1
      // eRE.mapprime(1,2);   // 0--eRE---2
      // eRE_sym = eRE_sym * eRE; // (0--eRE*--1) * (0--eRE--2) =
      // (1--eRE^dag--0) * (0--eRE--2) eRE_sym.prime(-1);

      // ITensor U_eRE, D_eRE;
      // diagHermitian(eRE_sym, U_eRE, D_eRE);

      // std::vector<double> dM_elems;
      // for (int idm=1; idm<=D_eRE.inds().front().m(); idm++)
      // dM_elems.push_back(
      // 		sqrt(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)))
      // ); D_eRE =
      // diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
      // ##### END V5 ##################################################

      // ##### V6 ######################################################
      // ITensor U_eRE, D_eRE;
      // diagHermitian(eRE_sym, U_eRE, D_eRE);

      // double msign = 1.0;
      // double mval = 0.;
      // std::vector<double> dM_elems;
      // for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {
      // 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
      // 	if (std::abs(dM_elems.back()) > mval) {
      // 		mval = std::abs(dM_elems.back());
      // 		msign = dM_elems.back()/mval;
      // 	}
      // }
      // if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

      // // Drop negative EV's
      // if(dbg && (dbgLvl >= 1)) {
      // 	std::cout<<"REFINED SPECTRUM"<< std::endl;
      // 	std::cout<<"MAX EV: "<< mval << std::endl;
      // }
      // for (auto & elem : dM_elems) {
      // 	if (elem < 0.0 && std::fabs(elem/mval) < svd_cutoff) {
      // 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 <<
      // std::endl; 		elem = 0.0; 	} else if (elem < 0.0) { 		if(dbg && (dbgLvl
      // >= 1)) std::cout<< elem <<" -> "<< std::fabs(elem) << std::endl; elem =
      // std::fabs(elem);
      // 	}
      // }
      // ##### END V6 ##################################################

      D_eRE = diagTensor(dM_elems, D_eRE.inds().front(), D_eRE.inds().back());

      eRE_sym = ((conj(U_eRE) * D_eRE) * prime(U_eRE)) *
                delta(combinedIndex(cmbBra), prime(combinedIndex(cmbKet)));
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
    eRE * (eA * delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2])));
  protoK *= (eB * delta(cls.AIc(tn[1], pl[3]), cls.AIc(tn[2], pl[4])));
  protoK *= eD;
  if (dbg && (dbgLvl >= 2))
    Print(protoK);

  protoK = ((protoK * delta(opPI[0], phys[0])) * uJ1J2.H1) *
           prime(delta(opPI[0], phys[0]));
  protoK = ((protoK * delta(opPI[1], phys[1])) * uJ1J2.H2) *
           prime(delta(opPI[1], phys[1]));
  protoK = ((protoK * delta(opPI[2], phys[2])) * uJ1J2.H3) *
           prime(delta(opPI[2], phys[2]));
  protoK.prime(PHYS, -1);
  if (dbg && (dbgLvl >= 2))
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
  auto NORMUPSI = ((protoK * delta(opPI[0], phys[0])) * conj(uJ1J2.H1)) *
                  prime(delta(opPI[0], phys[0]));
  NORMUPSI = ((NORMUPSI * delta(opPI[1], phys[1])) * conj(uJ1J2.H2)) *
             prime(delta(opPI[1], phys[1]));
  NORMUPSI = ((NORMUPSI * delta(opPI[2], phys[2])) * conj(uJ1J2.H3)) *
             prime(delta(opPI[2], phys[2]));
  NORMUPSI.prime(PHYS, -1);
  NORMUPSI *=
    (prime(conj(eA), AUXLINK, 4) *
     delta(prime(cls.AIc(tn[0], pl[1]), 4), prime(cls.AIc(tn[1], pl[2]), 4)));
  NORMUPSI *=
    (prime(conj(eB), AUXLINK, 4) *
     delta(prime(cls.AIc(tn[1], pl[3]), 4), prime(cls.AIc(tn[2], pl[4]), 4)));
  NORMUPSI *= prime(conj(eD), AUXLINK, 4);

  if (NORMUPSI.r() > 0)
    std::cout << "NORMPSI or OVERLAP rank > 0" << std::endl;
  double normUPsi = sumels(NORMUPSI);

  // trial initialization
  // TODO mask ?
  if (fuTrialInit) {
    auto maskS = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? 1.0 : 1.0;
    };
    auto cutoffS = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? r : 0;
    };
    auto SqrtT = [&machine_eps](Real r) {
      return (r > std::sqrt(10.0 * machine_eps)) ? std::sqrt(r) : 0;
    };
    auto printS = [](Real r) {
      std::cout << std::scientific << r << std::endl;
    };
    auto extractDiagReal = [](Diag<Real> const& d) { return d.store; };
    auto invDiagT = [](ITensor const& t) -> ITensor {
      double machine_eps = std::numeric_limits<double>::epsilon();

      // assume t is diagTensor of rank 2
      std::vector<double> tmpD(t.inds()[0].m());
      double const tol = t.real(t.inds()[0](1), t.inds()[1](1)) *
                         std::max(t.inds()[0].m(), t.inds()[1].m()) *
                         machine_eps;

      double elem;
      for (int i = 1; i <= t.inds()[0].m(); i++) {
        tmpD[i - 1] = (std::abs(t.real(t.inds()[0](i), t.inds()[1](i))) > tol)
                        ? 1.0 / t.real(t.inds()[0](i), t.inds()[1](i))
                        : 0.0;
      }

      return diagTensor(tmpD, t.inds()[0], t.inds()[1]);
    };
    auto sumSquares = [&extractDiagReal](ITensor const& t1,
                                         ITensor const& t2) -> Real {
      // assume equal length
      auto els1 = applyFunc(extractDiagReal, t1.store());
      auto els2 = applyFunc(extractDiagReal, t2.store());
      double sum = 0.0;
      for (int i = 0; i < els1.size(); i++)
        sum += (els1[i] - els2[i]) * (els1[i] - els2[i]);
      return sum;
    };
    auto diagCopyAndIndex = [&extractDiagReal](ITensor const& t, Index i1,
                                               Index i2) -> ITensor {
      // copy storage of t
      auto elems = applyFunc(extractDiagReal, t.store());
      return diagTensor(elems, i1, i2);
    };

    // relabel indices on MPO to phys indices of on-site tensors
    auto tEA = (eA * (delta(opPI[0], phys[0]) * uJ1J2.H1)) *
               delta(prime(opPI[0]), phys[0]);
    auto tEB = (eB * (delta(opPI[1], phys[1]) * uJ1J2.H2)) *
               delta(prime(opPI[1]), phys[1]);
    auto tED = (eD * (delta(opPI[2], phys[2]) * uJ1J2.H3)) *
               delta(prime(opPI[2]), phys[2]);

    // combine enlarged indices
    auto cA = combiner(uJ1J2.a12, cls.AIc(tn[0], pl[1]));
    auto cAB = combiner(uJ1J2.a12, cls.AIc(tn[1], pl[2]));
    auto cBD = combiner(uJ1J2.a23, cls.AIc(tn[1], pl[3]));
    auto cD = combiner(uJ1J2.a23, cls.AIc(tn[2], pl[4]));
    auto iA = combinedIndex(cA);
    auto iAB = combinedIndex(cAB);
    auto iBD = combinedIndex(cBD);
    auto iD = combinedIndex(cD);

    tEA *= cA;
    tEB = (tEB * cAB) * cBD;
    tED *= cD;

    auto lAB = diagTensor(std::vector<double>(iA.m(), 1.0), iA, iAB);
    auto lBD = diagTensor(std::vector<double>(iD.m(), 1.0), iBD, iD);

    // quasi-orthoginalization
    ITensor T;
    double tol = 1.0e-8;
    double dist0 = 0.0;
    bool converged = false;
    while (not converged) {
      double dist = 0.0;

      T = tEA * lAB * tEB * lBD;
      tEA = ITensor(iQA, phys[0]);
      ITensor sAB;
      svd(T, tEA, sAB, tEB, {"Truncate", false});
      sAB *= 1.0 / sAB.real(sAB.inds()[0](1), sAB.inds()[1](1));
      dist += sumSquares(sAB, lAB);
      lAB = diagCopyAndIndex(sAB, iA, iAB);
      tEA *= delta(commonIndex(tEA, sAB), iA);
      tEB *= delta(commonIndex(tEB, sAB), iAB);
      tEB *= invDiagT(lBD);

      T = lAB * tEB * lBD * tED;
      tED = ITensor(iQD, phys[2]);
      ITensor sBD;
      svd(T, tED, sBD, tEB, {"Truncate", false});
      Print(T);
      Print(sBD);
      sBD *= 1.0 / sBD.real(sBD.inds()[0](1), sBD.inds()[1](1));
      dist += sumSquares(sBD, lBD);
      lBD = diagCopyAndIndex(sBD, iD, iBD);
      tED *= delta(commonIndex(tED, sBD), iD);
      tEB *= delta(commonIndex(tEB, sBD), iBD);
      tEB *= invDiagT(lAB);

      if (std::sqrt(std::abs(dist - dist0)) < tol)
        converged = true;
      dist0 = dist;
    }

    // truncate
    Print(lAB);
    lAB.visit(printS);
    Print(lBD);
    lBD.visit(printS);

    lAB.apply(SqrtT);
    lBD.apply(SqrtT);

    eA = (tEA * lAB) * delta(commonIndex(lAB, tEB), cls.AIc(tn[0], pl[1]));
    eB = (tEB * lAB) * delta(commonIndex(lAB, tEA), cls.AIc(tn[1], pl[2]));
    eB *= lBD;
    eB *= delta(commonIndex(lBD, tED), cls.AIc(tn[1], pl[3]));
    eD = (tED * lBD) * delta(commonIndex(lBD, tEB), cls.AIc(tn[2], pl[4]));
  }

  auto cmbX1 = combiner(eA.inds()[0], eA.inds()[1], eA.inds()[2]);
  auto cmbX2 = combiner(eB.inds()[0], eB.inds()[1], eB.inds()[2], eB.inds()[3]);
  auto cmbX3 = combiner(eD.inds()[0], eD.inds()[1], eD.inds()[2]);

  double normPsi, finit, finitN;
  double prev_finit = 1.0;
  ITensor M, K, NORMPSI, OVERLAP;
  double ferr;
  int fiter;

  int altlstsquares_iter = 0;
  bool converged = false;
  std::vector<bool> als_tensor_stop(3, false);
  std::vector<double> fdist, fdistN, vec_normPsi;
  std::cout << "ENTERING ALS LOOP tol: " << epsdistf << " solver: " << linsolver
            << std::endl;
  t_begin_int = std::chrono::steady_clock::now();
  while (not converged) {
    // Optimizing eA
    // 1) construct matrix M, which is defined as <psi~|psi~> = eA^dag * M * eA

    // BRA
    M = eRE * prime(conj(eD), AUXLINK, 4);
    M *=
      delta(prime(cls.AIc(tn[2], pl[4]), 4), prime(cls.AIc(tn[1], pl[3]), 4));
    M *= prime(conj(eB), AUXLINK, 4);
    M *=
      delta(prime(cls.AIc(tn[1], pl[2]), 4), prime(cls.AIc(tn[0], pl[1]), 4));
    if (dbg && (dbgLvl >= 3))
      Print(M);

    // KET
    M *= eD;
    M *= delta(cls.AIc(tn[2], pl[4]), cls.AIc(tn[1], pl[3]));
    M *= eB;
    M *= delta(cls.AIc(tn[1], pl[2]), cls.AIc(tn[0], pl[1]));
    if (dbg && (dbgLvl >= 2))
      Print(M);

    // 2) construct vector K, which is defined as <psi~|psi'> = eA^dag * K
    K = protoK * prime(conj(eD), AUXLINK, 4);
    K *=
      delta(prime(cls.AIc(tn[2], pl[4]), 4), prime(cls.AIc(tn[1], pl[3]), 4));
    K *= prime(conj(eB), AUXLINK, 4);
    K *=
      delta(prime(cls.AIc(tn[1], pl[2]), 4), prime(cls.AIc(tn[0], pl[1]), 4));
    if (dbg && (dbgLvl >= 2))
      Print(K);

    // <psi'|psi'>
    NORMPSI = (prime(conj(eA), AUXLINK, 4) * M) * eA;
    // <psi'|U|psi>
    OVERLAP = prime(conj(eA), AUXLINK, 4) * K;

    if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
      std::cout << "NORMPSI or OVERLAP rank > 0" << std::endl;
    normPsi = sumels(NORMPSI);
    finit = prev_finit = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;

    fdist.push_back(finit);
    fdistN.push_back(finitN);
    vec_normPsi.push_back(normPsi);
    // if ( fdist.back() < cg_fdistance_eps ) { converged = true; break; }
    std::cout << "stopCond: "
              << (fdistN.back() - fdistN[fdistN.size() - 2]) / fdistN[0]
              << std::endl;
    if ((fdistN.size() > 1) &&
        std::abs((fdistN.back() - fdistN[fdistN.size() - 2]) / fdistN[0]) <
          epsdistf) {
      converged = true;
      break;
    }

    // ***** SOLVE LINEAR SYSTEM M*eA = K by CG ***************************
    auto temp = eA;
    FUlinSys fulscg(M, K, eA, cmbX1, combiner(iQA, cls.AIc(tn[0], pl[1])),
                    args);
    fulscg.solve(K, eA, fiter, ferr, ls, args);

    NORMPSI = (prime(conj(eA), AUXLINK, 4) * M) * eA;
    OVERLAP = prime(conj(eA), AUXLINK, 4) * K;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;
    if (finitN > fdistN.back()) {  //|| fulscg.nres < epsdistf ) {
      std::cout << "fdistN increased. Reverting to original tensor";
      eA = temp;
      fdist.pop_back();
      fdistN.pop_back();
      vec_normPsi.pop_back();
      als_tensor_stop[0] = true;
    } else {
      als_tensor_stop[0] = false;
    }
    std::cout << "f_err= " << ferr << " f_iter= " << fiter << std::endl;

    // Optimizing eB
    // 1) construct matrix M, which is defined as <psi~|psi~> = eB^dag * M * eB

    // BRA
    M = eRE * prime(conj(eD), AUXLINK, 4);
    M *= prime(delta(cls.AIc(tn[2], pl[4]), cls.AIc(tn[1], pl[3])), 4);
    M *= prime(conj(eA), AUXLINK, 4);
    M *= prime(delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2])), 4);
    if (dbg && (dbgLvl >= 3))
      Print(M);

    // KET
    M *= eD;
    M *= delta(cls.AIc(tn[2], pl[4]), cls.AIc(tn[1], pl[3]));
    M *= eA;
    M *= delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2]));
    if (dbg && (dbgLvl >= 2))
      Print(M);

    // 2) construct vector K, which is defined as <psi~|psi'> = eB^dag * K
    K = protoK * prime(conj(eD), AUXLINK, 4);
    K *= prime(delta(cls.AIc(tn[2], pl[4]), cls.AIc(tn[1], pl[3])), 4);
    K *= prime(conj(eA), AUXLINK, 4);
    K *= prime(delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2])), 4);
    if (dbg && (dbgLvl >= 2))
      Print(K);

    // <psi'|psi'>
    NORMPSI = (prime(conj(eB), AUXLINK, 4) * M) * eB;
    // <psi'|U|psi>
    OVERLAP = prime(conj(eB), AUXLINK, 4) * K;

    if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
      std::cout << "NORMPSI or OVERLAP rank > 0" << std::endl;
    normPsi = sumels(NORMPSI);
    prev_finit = finit;
    finit = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;

    fdist.push_back(finit);
    fdistN.push_back(finitN);
    vec_normPsi.push_back(normPsi);
    std::cout << "stopCond: " << (finit - prev_finit) / fdist[0] << std::endl;
    if ((fdistN.size() > 1) &&
        std::abs((fdistN.back() - fdistN[fdistN.size() - 2]) / fdistN[0]) <
          epsdistf) {
      converged = true;
      break;
    }

    // ***** SOLVE LINEAR SYSTEM M*eB = K ******************************
    temp = eB;
    FUlinSys fulscgEB(
      M, K, eB, cmbX2,
      combiner(iQB, cls.AIc(tn[1], pl[2]), cls.AIc(tn[1], pl[3])), args);
    fulscgEB.solve(K, eB, fiter, ferr, ls, args);

    NORMPSI = (prime(conj(eB), AUXLINK, 4) * M) * eB;
    OVERLAP = prime(conj(eB), AUXLINK, 4) * K;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;
    if (finitN > fdistN.back()) {  //|| fulscgEB.nres < epsdistf ) {
      std::cout << "fdistN increased. Reverting to original tensor";
      eB = temp;
      fdist.pop_back();
      fdistN.pop_back();
      vec_normPsi.pop_back();
      als_tensor_stop[1] = true;
    } else {
      als_tensor_stop[1] = false;
    }
    std::cout << "EB f_err= " << ferr << " f_iter= " << fiter << std::endl;

    // Optimizing eD
    // 1) construct matrix M, which is defined as <psi~|psi~> = eD^dag * M * eD

    // BRA
    M = eRE * prime(conj(eA), AUXLINK, 4);
    M *= prime(delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2])), 4);
    M *= prime(conj(eB), AUXLINK, 4);
    M *= prime(delta(cls.AIc(tn[1], pl[3]), cls.AIc(tn[2], pl[4])), 4);
    if (dbg && (dbgLvl >= 3))
      Print(M);

    // KET
    M *= eA;
    M *= delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2]));
    M *= eB;
    M *= delta(cls.AIc(tn[1], pl[3]), cls.AIc(tn[2], pl[4]));
    if (dbg && (dbgLvl >= 2))
      Print(M);

    // 2) construct vector K, which is defined as <psi~|psi'> = eD^dag * K
    K = protoK * prime(conj(eA), AUXLINK, 4);
    K *= prime(delta(cls.AIc(tn[0], pl[1]), cls.AIc(tn[1], pl[2])), 4);
    K *= prime(conj(eB), AUXLINK, 4);
    K *= prime(delta(cls.AIc(tn[1], pl[3]), cls.AIc(tn[2], pl[4])), 4);
    if (dbg && (dbgLvl >= 2))
      Print(K);

    // <psi'|psi'>
    NORMPSI = (prime(conj(eD), AUXLINK, 4) * M) * eD;
    // <psi'|U|psi>
    OVERLAP = prime(conj(eD), AUXLINK, 4) * K;

    if (NORMPSI.r() > 0 || OVERLAP.r() > 0)
      std::cout << "NORMPSI or OVERLAP rank > 0" << std::endl;
    normPsi = sumels(NORMPSI);
    prev_finit = finit;
    finit = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;

    fdist.push_back(finit);
    fdistN.push_back(finitN);
    vec_normPsi.push_back(normPsi);
    std::cout << "stopCond: " << (finit - prev_finit) / fdist[0] << std::endl;
    if ((fdistN.size() > 1) &&
        std::abs((fdistN.back() - fdistN[fdistN.size() - 2]) / fdistN[0]) <
          epsdistf) {
      converged = true;
      break;
    }

    // ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
    temp = eD;

    FUlinSys fulscgED(M, K, eD, cmbX3, combiner(iQD, cls.AIc(tn[2], pl[4])),
                      args);
    fulscgED.solve(K, eD, fiter, ferr, ls, args);

    NORMPSI = (prime(conj(eD), AUXLINK, 4) * M) * eD;
    OVERLAP = prime(conj(eD), AUXLINK, 4) * K;
    finitN = 1.0 - 2.0 * sumels(OVERLAP) / std::sqrt(normUPsi * normPsi) + 1.0;
    if (finitN > fdistN.back()) {  //|| fulscgED.nres < epsdistf ) {
      std::cout << "fdistN increased. Reverting to original tensor";
      eD = temp;
      fdist.pop_back();
      fdistN.pop_back();
      vec_normPsi.pop_back();
      als_tensor_stop[2] = true;
    } else {
      als_tensor_stop[2] = false;
    }
    std::cout << "f_err= " << ferr << " f_iter= " << fiter << std::endl;

    altlstsquares_iter++;
    if (altlstsquares_iter >= maxAltLstSqrIter ||
        (als_tensor_stop[0] && als_tensor_stop[1] && als_tensor_stop[2]))
      converged = true;
  }
  t_end_int = std::chrono::steady_clock::now();

  std::cout << "STEP f=||psi'>-|psi>|^2 f_normalized <psi'|psi'>" << std::endl;
  for (int i = 0; i < fdist.size(); i++)
    std::cout << i << " " << fdist[i] << " " << fdistN[i] << " "
              << vec_normPsi[i] << std::endl;

  // update on-site tensors of cluster
  cls.sites.at(tn[0]) = QA * eA;
  cls.sites.at(tn[1]) = QB * eB;
  cls.sites.at(tn[2]) = QD * eD;

  // max element of on-site tensors
  // or norm-distance of new versus original tensors
  std::string diag_maxElem;
  for (int i = 0; i < 4; i++) {
    m = 0.;
    // cls.sites.at(tn[i]).visit(max_m);
    if (i < 3)
      m = norm(cls.sites.at(tn[i]) - orig_tensors[i]);
    diag_maxElem = diag_maxElem + tn[i] + " " + std::to_string(m);
    if (i < 3)
      diag_maxElem += " ";
  }
  std::cout << diag_maxElem << std::endl;

  // normalize updated tensors
  if (otNormType == "BLE") {
    for (int i = 0; i < 3; i++) {
      m = 0.;
      cls.sites.at(tn[i]).visit(max_m);
      cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / std::sqrt(m);
    }
  } else if (otNormType == "BALANCE") {
    double iso_tot_mag = 1.0;
    for (auto& site_e : cls.sites) {
      m = 0.;
      site_e.second.visit(max_m);
      site_e.second = site_e.second / m;
      iso_tot_mag = iso_tot_mag * m;
    }
    for (auto& site_e : cls.sites) {
      site_e.second = site_e.second * std::pow(iso_tot_mag, (1.0 / 8.0));
    }
  } else if (otNormType == "NONE") {
  } else {
    std::cout << "Unsupported on-site tensor normalisation after full update: "
              << otNormType << std::endl;
    exit(EXIT_FAILURE);
  }

  // max element of on-site tensors after normalization
  for (int i = 0; i < 4; i++) {
    m = 0.;
    cls.sites.at(tn[i]).visit(max_m);
    if (i < 3)
      std::cout << tn[i] << " " << std::to_string(m) << " ";
    else
      std::cout << tn[i] << " " << std::to_string(m);
  }
  std::cout << std::endl;

  // prepare and return diagnostic data
  diag_data.add("alsSweep", altlstsquares_iter);

  std::string siteMaxElem_descriptor =
    "site max_elem site max_elem site max_elem site max_elem";
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
    "f_init f_final normalizedf_init normalizedf_final norm(psi')_init "
    "norm(psi')_final time[s]";
  diag_data.add("locMinDiag_descriptor", logMinDiag_descriptor);
  diag_data.add("locMinDiag", oss.str());
  if (symmProtoEnv) {
    diag_data.add("diag_protoEnv", diag_protoEnv);
    diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
  }

  // auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5]
  // 	- overlaps[overlaps.size()-4];
  // auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2]
  // 	- overlaps[overlaps.size()-1];
  // diag_data.add("finalDist0",dist0);
  // diag_data.add("finalDist1",dist1);

  // minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum
  // taken diag_data.add("minGapDisc",minGapDisc);
  // diag_data.add("minEvKept",minEvKept);
  // diag_data.add("maxEvDisc",maxEvDisc);

  return diag_data;
}

FUlinSys::FUlinSys(ITensor& MM,
                   ITensor& BB,
                   ITensor& AA,
                   ITensor ccmbA,
                   ITensor ccmbKet,
                   Args const& aargs)
  : M(MM), B(BB), A(AA), cmbA(ccmbA), cmbKet(ccmbKet), args(aargs) {
  dbg = args.getBool("dbg", false);
  auto epsreg = args.getReal("epsregularisation", 0.0);

  std::vector<Index> iket;
  for (auto const& i : M.inds()) {
    if (i.primeLevel() < 4)
      iket.emplace_back(i);
  }

  cmbKet = combiner(iket);
  auto cmbBra = prime(cmbKet, 4);

  auto i0 = combinedIndex(cmbKet);
  auto i1 = combinedIndex(cmbBra);

  M = (M * cmbKet) * cmbBra;

  // Symmetrize
  // M = 0.5*(M + prime( ((M * delta(i0,prime(i1)) ) * delta(i1,prime(i0)) ),
  // -1) ) ;

  // analyse sparsity
  if (dbg) {
    double min_mag = 1.0e-8;
    int count = 0;
    int countE7 = 0;
    int countE6 = 0;
    auto sparseCheck = [&min_mag, &count, &countE7, &countE6](Real r) {
      double absr = std::fabs(r);
      if (absr > min_mag)
        count += 1;
      if (absr > min_mag * 10.0)
        countE7 += 1;
      if (absr > min_mag * 100.0)
        countE6 += 1;
    };
    M.visit(sparseCheck);
    std::cout << "Sparsity e-8: " << count << "/" << i0.m() * i1.m()
              << " %: " << count / ((double)i0.m() * i1.m()) << std::endl;
    std::cout << "Sparsity e-7: " << countE7 << "/" << i0.m() * i1.m()
              << " %: " << countE7 / ((double)i0.m() * i1.m()) << std::endl;
    std::cout << "Sparsity e-6: " << countE6 << "/" << i0.m() * i1.m()
              << " %: " << countE6 / ((double)i0.m() * i1.m()) << std::endl;
  }

  // regularize
  for (int i = 1; i <= i0.m(); i++) {
    M.set(i0(i), i1(i), M.real(i0(i), i1(i)) + epsreg);
  }

  // Diagonal dominance check
  if (dbg) {
    std::vector<double> diag_elems_mag(i0.m());
    std::vector<double> offdiag_mag_sums(i0.m(), 0.0);
    int count_diag_dominant = 0;
    for (int i = 1; i <= i0.m(); i++) {
      diag_elems_mag[i - 1] = std::abs(M.real(i0(i), i1(i)));
      for (int j = 1; j <= i0.m(); j++) {
        offdiag_mag_sums[i - 1] += std::abs(M.real(i0(j), i1(i)));
      }
      offdiag_mag_sums[i - 1] -= diag_elems_mag[i - 1];
      if (diag_elems_mag[i - 1] > offdiag_mag_sums[i - 1])
        count_diag_dominant += 1;
    }
    std::cout << "DiagDominance: " << count_diag_dominant << "/" << i0.m()
              << std::endl;
  }

  M = (M * cmbKet) * cmbBra;

  auto RES = M * A - B;
  res = norm(RES);
  nres = res / norm(B);
  std::cout << "RES: " << res << " N_RES: " << nres << " ";
}

void FUlinSys::solve(itensor::ITensor const& b,
                     itensor::ITensor& x,
                     int& iter,
                     double& err,
                     LinSysSolver const& ls,
                     Args const& args) {
  std::vector<Index> iket;
  for (auto const& i : M.inds()) {
    if (i.primeLevel() < 4)
      iket.emplace_back(i);
  }
  auto pI = findtype(B, PHYS);
  iket.emplace_back(pI);
  cmbKet = combiner(iket);

  M *= delta(pI, prime(pI, 4));
  M = (cmbKet * M) * prime(cmbKet, 4);

  B.prime(PHYS, 4);
  B *= prime(cmbKet, 4);
  B.prime(-4);

  x *= cmbKet;
  x.prime(4);

  auto bi = B.inds()[0];
  auto xi = x.inds()[0];
  B *= delta(bi, xi);
  x *= delta(xi, bi);

  ls.solve(M, B, x, args);

  B *= delta(bi, xi);
  x *= delta(xi, bi);

  x.prime(-4);
  x *= cmbKet;

  B.prime(combinedIndex(cmbKet), 4);
  B *= prime(cmbKet, 4);
  B.prime(PHYS, -4);

  M = (cmbKet * M) * prime(cmbKet, 4);

  ITensor temp(pI, prime(pI, 4));
  temp.fill(0.0);
  temp.set(pI(1), prime(pI, 4)(1), 1.0);
  M *= temp;
}
