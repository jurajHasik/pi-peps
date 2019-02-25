#include "pi-peps/config.h"
#include "pi-peps/ctm-cluster-env_v2.h"

// TODO Implement convergence check as general function. The actual
// implementation may vary - difference between SVD decomp,
// Eigenvalue decomp, etc.

// TODO Implement Isometry as a general function returning the
// isometry tensor Z - the actual used implementation may vary
// SVD decomp, Eigenvalue decomp, etc.

using namespace itensor;

// ############################################################################
// member methods of CtmEnv

// CtmEnv::CtmEnv () { solver = SvdSolver() }

CtmEnv::CtmEnv(std::string t_name,
               int t_x,
               Cluster const& c,
               SvdSolver& ssolver,
               Args const& args)
  : p_cluster(&c),
    solver(ssolver),
    m_name(t_name),
    d(c.auxBondDim * c.auxBondDim),
    x(t_x),
    sizeN(c.sizeN),
    sizeM(c.sizeM) {
  if (args.defined("isoPseudoInvCutoff")) {
    default_pinv_cutoff = false;
    isoPseudoInvCutoff = args.getReal("isoPseudoInvCutoff");
  }
  isoMinElemWarning = args.getReal("isoMinElemWarning", 1.0e-4);
  isoMaxElemWarning = args.getReal("isoMaxElemWarning", 1.0e4);
  SVD_METHOD = args.getString("SVD_METHOD", "itensor");
  rsvd_power = args.getInt("rsvd_power", 2);
  rsvd_reortho = args.getInt("rsvd_reortho", 1);
  rsvd_oversampling = args.getInt("rsvd_oversampling", 10);
  DBG = args.getBool("dbg", false);
  DBG_LVL = args.getInt("dbgLevel", 0);

  /*
   * Intialize tensors and indices compromising the environment
   *   ____          ____          ____
   *  |C_LU|--I_U0--|T_U |--I_U1--|C_RU|
   *    |             |              |
   *   I_L0          I_XV0          I_R0
   *   _|__           |              |
   *  |T_L0|--I_XH0--|X_xy|--I_XH1--|T_R0|
   *    |             |              |
   *   I_L1          I_XV1          I_R1
   *   _|__          _|__          _|__
   *  |C_LD|--I_D0--|T_D |--I_D1--|C_RD|
   *
   *  where both (e.g.) I(Tl) and I(Tl)' have the same underlying
   *  index, but the "the prime" differentiates the two. This will
   *  enable us to formulate "insert, absorb & renormalize" algorithm
   */

  // Define indices between corner matrices "C_*" and row/column
  // tensors "T_*"
  // I_U = Index(TAG_I_U, x, ULINK);
  // I_R = Index(TAG_I_R, x, RLINK);
  // I_D = Index(TAG_I_D, x, DLINK);
  // I_L = Index(TAG_I_L, x, LLINK);

  // Define indices between row/column tensors "T_*" and on-site
  // tensors "X**"
  I_XH = Index(TAG_I_XH, d, HSLINK);
  I_XV = Index(TAG_I_XV, d, VSLINK);

  for (auto const& id : c.siteIds) {
    eaux[id] = std::vector<Index>(8);
    eaux[id] = {Index(id + "-" + TAG_I_U, x, ULINK),
                Index(id + "-" + TAG_I_U, x, ULINK, 1),
                Index(id + "-" + TAG_I_R, x, RLINK),
                Index(id + "-" + TAG_I_R, x, RLINK, 1),
                Index(id + "-" + TAG_I_D, x, DLINK, 1),
                Index(id + "-" + TAG_I_D, x, DLINK),
                Index(id + "-" + TAG_I_L, x, LLINK, 1),
                Index(id + "-" + TAG_I_L, x, LLINK)};
  }

  itaux.resize(4);
  // LEFT T_* tensors
  for (int direction = 0; direction < 4; direction++)
    for (auto const& id : c.siteIds)
      itaux[direction][id].resize(4);

  for (auto const& id : c.siteIds) {
    itaux[0][id][1] = eaux[id][7];
    itaux[0][id][3] = eaux[id][6];
  }
  // UP T_* tensors
  for (auto const& id : c.siteIds) {
    itaux[1][id][0] = eaux[id][0];
    itaux[1][id][2] = eaux[id][1];
  }
  // RIGHT T_* tensors
  for (auto const& id : c.siteIds) {
    itaux[2][id][1] = eaux[id][2];
    itaux[2][id][3] = eaux[id][3];
  }
  // DOWN T_* tensors
  for (auto const& id : c.siteIds) {
    itaux[3][id][0] = eaux[id][5];
    itaux[3][id][2] = eaux[id][4];
  }

  // Combiners from site AUX indices to I_XH, I_XV
  for (auto const& id : c.siteIds) {
    CMB[id] = std::vector<ITensor>(4);

    auto formCMB = [this](std::string id, int dir) -> ITensor {
      auto tmp = p_cluster->AIBraKetPair(id, dir);
      return combiner(tmp);
    };

    for (int dir = 0; dir < 4; dir++)
      CMB[id][dir] = formCMB(id, dir);
  }
  // directions on lattice and their corresponding fused indices
  fusedSiteI = std::vector<Index>({I_XH, I_XV, prime(I_XH), prime(I_XV)});
  for (auto const& id : c.siteIds) {
    faux[id] = std::vector<Index>(4);

    faux[id] = {Index(TAG_I_XH, std::pow(c.AIc(id, 0).m(), 2), HSLINK),
                Index(TAG_I_XV, std::pow(c.AIc(id, 1).m(), 2), VSLINK),
                Index(TAG_I_XH, std::pow(c.AIc(id, 2).m(), 2), HSLINK, 1),
                Index(TAG_I_XV, std::pow(c.AIc(id, 3).m(), 2), VSLINK, 1)};
  }

  // create environment tensors
  for (const auto& id : p_cluster->siteIds) {
    auto AIBraKetPair = [this](std::string id, int dir) {
      return p_cluster->AIBraKetPair(id, dir);
    };

    // Construct tensors "C_*" for every non-eq cluster site
    C_LU[id] = ITensor(envIndPair(id, 7, id, 0));
    C_RU[id] = ITensor(envIndPair(id, 1, id, 2));
    C_RD[id] = ITensor(envIndPair(id, 3, id, 4));
    C_LD[id] = ITensor(envIndPair(id, 5, id, 6));

    // Construct tensors "T_*"
    auto tmp = envIndPair(id, 6, id, 7);
    auto tmp2 = AIBraKetPair(id, 0);
    tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
    T_L[id] = ITensor(tmp);

    tmp = envIndPair(id, 0, id, 1);
    tmp2 = AIBraKetPair(id, 1);
    tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
    T_U[id] = ITensor(tmp);

    tmp = envIndPair(id, 2, id, 3);
    tmp2 = AIBraKetPair(id, 2);
    tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
    T_R[id] = ITensor(tmp);

    tmp = envIndPair(id, 4, id, 5);
    tmp2 = AIBraKetPair(id, 3);
    tmp.insert(tmp.end(), tmp2.begin(), tmp2.end());
    T_D[id] = ITensor(tmp);
  }

  // Iterate over siteIds as given in Cluster c
  // for( const auto& siteIdEntry : c.siteIds ) {
  //     sites.push_back( contractOST(c.sites.at(siteIdEntry)) );
  //     siteIds.push_back( siteIdEntry );

  //     // add entries for map from cluster site to tensor
  //     for ( const auto& cToSEntry : c.cToS ) {
  //         if( cToSEntry.second == siteIdEntry ) {
  //             cToS[ cToSEntry.first ] = sites.size()-1;
  //         }
  //     }
  // }

  // Construct vectors holding SVD spectrum of corner matrices
  spec = {
    std::vector<std::vector<double>>(c.siteIds.size(), std::vector<double>(x)),
    std::vector<std::vector<double>>(c.siteIds.size(), std::vector<double>(x)),
    std::vector<std::vector<double>>(c.siteIds.size(), std::vector<double>(x)),
    std::vector<std::vector<double>>(c.siteIds.size(), std::vector<double>(x))};
}

/*
 * Assume the whole environment is delivered as a set of simple
 * environments for each of non-equivalent sites within cluster
 * Preserves compatibility with LEGACY pi-peps/ctm-cluster-io.h
 *
 */
// CtmEnv::CtmEnv (std::string t_name,  std::vector<CtmData> const& ctmD,
//     Cluster const& c, SvdSolver & ssolver, Args const& args)
//     : m_name(t_name), solver(ssolver), d(ctmD[0].auxDimSite),
//     x(ctmD[0].auxDimEnv), sizeN(c.sizeN), sizeM(c.sizeM),
//     // environment indices
//     I_U(ctmD[0].I_U), I_R(ctmD[0].I_R), I_D(ctmD[0].I_D), I_L(ctmD[0].I_L),
//     // on-site indices
//     I_XH(ctmD[0].I_XH), I_XV(ctmD[0].I_XV) {

//     for (std::size_t i=0; i<ctmD.size(); i++) {
//         // corner matrices
//         C_LU.push_back(ctmD[i].C_LU);
//         C_RU.push_back(ctmD[i].C_RU);
//         C_RD.push_back(ctmD[i].C_RD);
//         C_LD.push_back(ctmD[i].C_LD);
//         // half-row/column tensors
//         T_U.push_back(ctmD[i].T_U[0]);
//         T_R.push_back(ctmD[i].T_R[0]);
//         T_D.push_back(ctmD[i].T_D[0]);
//         T_L.push_back(ctmD[i].T_L[0]);
//     }

//     if ( ctmD[0].auxDimSite != c.auxBondDim*c.auxBondDim ) {
//         std::cout <<"AuxDimSite of environment and auxBondDim**2 "
//             <<"of cluster are not in agreement"<< std::endl;
//         exit(EXIT_FAILURE);
//     }

//     // Iterate over siteIds as given in Cluster c
//     for( const auto& siteIdEntry : c.siteIds ) {
//         sites.push_back( contractOST(c.sites.at(siteIdEntry)) );
//         siteIds.push_back( siteIdEntry );

//         // add entries for map from cluster site to tensor
//         for ( const auto& cToSEntry : c.cToS ) {
//             if( cToSEntry.second == siteIdEntry ) {
//                 cToS[ cToSEntry.first ] = sites.size()-1;
//             }
//         }
//     }

//     // Construct vectors holding SVD spectrum of corner matrices
//     spec = {
//         std::vector< std::vector<double> >(c.siteIds.size(),
//         std::vector<double>(x) ), std::vector< std::vector<double>
//         >(c.siteIds.size(), std::vector<double>(x) ), std::vector<
//         std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
//         std::vector< std::vector<double> >(c.siteIds.size(),
//         std::vector<double>(x) )
//     };
// }

// ITensor CtmEnv::contractOST(ITensor const& T, bool expose) const {

//      * Construct on-site tensor X given by the contraction of bra & ket
//      * on-site tensors T^dag & T through physical index physI
//      *
//      *  auxI5  auxI6  auxI' auxI''          auxI5(x)auxI'
//      *      | /           | /                     ||
//      *     |T*|~~~physI~~|T|  =>   auxI4(x)auxI==|T*T|==auxI6(x)auxI''
//      *     / |           / |                      ||
//      * auxI4 auxI7    auxI auxI3            auxI7(x)auxI3
//      *
//      * where indices auxI[0..3] correspond to ket on-site tensor T and
//      * indices auxI[4..7] to ket. (x) denotes a tensor product of indices.
//      * To obtain final form of X we need to cast tensor product of
//      * indices, say auxI4(x)auxI, into a single index I_XH
//      * of size dim(I_XH) = dim(auxI4)*dim(auxI) = D*D = D^2 = d
//      * in accordance with CTM tensor network as defined above.
//      * Therefore we define a conversion tensor Y as follows
//      *            _
//      *    auxI4--| \
//      *           |Y --I_XH =: Y(h)
//      *     auxI--|_/
//      *
//      * with only non-zero elements being
//      *
//      *   Y(auxI4=i, auxI=j, I_XH)=D*(i-1)+j) = 1.0
//      *
//      * then we obtain X with proper indices as
//      *
//      * X = Y(h)*Y(h')*Y(v)*Y(v')*|T*T|
//      *

//     // Get auxBond index of T
//     auto auxI = noprime(findtype(T, AUXLINK));

//     if(auxI.m()*auxI.m() != d) {
//         std::cout <<"ctmEnv.d does not agree with onSiteT.dimD^2"<<
//         std::endl; exit(EXIT_FAILURE);
//     }

//     // Define combiner tensors Y*
//     auto C04 = combiner(auxI, prime(auxI,4));
//     auto C15 = prime(C04,1);
//     auto C26 = prime(C04,2);
//     auto C37 = prime(C04,3);

//     ITensor X;
//     if (expose)
//         X = (T*( conj(T).prime(AUXLINK,4).prime(PHYS,1) ))*C04*C15*C26*C37;
//     else
//         X = (T*( conj(T).prime(AUXLINK,4) ))*C04*C15*C26*C37;

//     // Define delta tensors D* to relabel combiner indices to I_XH, I_XV
//     auto DH0 = delta(I_XH, commonIndex(X,C04));
//     auto DV0 = delta(I_XV, commonIndex(X,C15));
//     auto DH1 = delta(prime(I_XH,1), commonIndex(X,C26));
//     auto DV1 = delta(prime(I_XV,1), commonIndex(X,C37));

//     return X*DH0*DV0*DH1*DV1;
// }

// ########################################################################
// environment initalization methods

void CtmEnv::initMockEnv() {
  for (const auto& id : p_cluster->siteIds) {
    C_LU[id].fill(1. + 0._i);
    C_RU[id].fill(1. + 0._i);
    C_RD[id].fill(1. + 0._i);
    C_LD[id].fill(1. + 0._i);
    T_L[id].fill(1. + 0._i);
    T_U[id].fill(1. + 0._i);
    T_R[id].fill(1. + 0._i);
    T_D[id].fill(1. + 0._i);
  }

  // normalizePTN();

  // computeSVDspec();

  std::cout << "INIT_ENV_const1 with all elements of C's and T's"
            << " set to constant" << std::endl;
  std::cout << std::string(72, '=') << std::endl;
}

void CtmEnv::initRndEnv(bool isComplex) {
  // Fill with random numbers (real or complex)

  auto shift05 = [](double r) { return r - 0.5; };

  for (const auto& id : p_cluster->siteIds) {
    randomize(C_LU[id], {"Complex", isComplex});
    randomize(C_RU[id], {"Complex", isComplex});
    randomize(C_RD[id], {"Complex", isComplex});
    randomize(C_LD[id], {"Complex", isComplex});
    randomize(T_L[id], {"Complex", isComplex});
    randomize(T_U[id], {"Complex", isComplex});
    randomize(T_R[id], {"Complex", isComplex});
    randomize(T_D[id], {"Complex", isComplex});
  }

  // for ( auto& t : C_LU ) { t.apply(shift05); }
  // for ( auto& t : C_RU ) { t.apply(shift05); }
  // for ( auto& t : C_RD ) { t.apply(shift05); }
  // for ( auto& t : C_LD ) { t.apply(shift05); }
  // for ( auto& t : T_U ) { t.apply(shift05); }
  // for ( auto& t : T_R ) { t.apply(shift05); }
  // for ( auto& t : T_D ) { t.apply(shift05); }
  // for ( auto& t : T_L ) { t.apply(shift05); }

  // normalizePTN();

  // computeSVDspec();

  std::cout << "INIT_ENV_rnd with all C's and T's random (complex ? "
            << isComplex << ")" << std::endl;
  std::cout << std::string(72, '=') << std::endl;
}

void CtmEnv::initCtmrgEnv(bool dbg) {
  std::cout << "===== INIT_ENV_ctmrg called " << std::string(44, '=')
            << std::endl;

  int const BRAKET_OFFSET = 4;

  auto TBraKet = [this, &BRAKET_OFFSET](std::string id) -> ITensor {
    return p_cluster->sites.at(id) *
           dag(prime(p_cluster->sites.at(id), AUXLINK, BRAKET_OFFSET));
  };

  auto contractBraKetInd = [this, &BRAKET_OFFSET](ITensor& t, std::string id0,
                                                  int dir0, std::string id1,
                                                  int dir1) {
    t *= p_cluster->DContract(id0, dir0, id1, dir1);
    t *= prime(p_cluster->DContract(id0, dir0, id1, dir1), BRAKET_OFFSET);
  };

  // TODO? dag for one of the indices
  auto DBraKetInd = [this, &BRAKET_OFFSET](std::string id, int dir) -> ITensor {
    auto a = p_cluster->AIc(id, dir);
    return delta(a, prime(a, BRAKET_OFFSET));
  };
  auto combineSiteBraKetToEnv = [this](ITensor& t, std::string id, int dir,
                                       std::string env_id, int env_dir) {
    t *= CMB.at(id)[dir];
    t *= delta(combinedIndex(CMB.at(id)[dir]), eaux.at(env_id)[env_dir]);
  };

  for (const auto& id : p_cluster->siteIds) {
    Vertex v = p_cluster->idToV.at(id);

    if (dbg)
      std::cout << "----- generating init env for site " << id << " -----"
                << std::endl;

    // left-upper corner v + (-1,-1)
    auto tmp_id = p_cluster->vertexToId(v + Shift(-1, -1));
    auto t = TBraKet(tmp_id);
    // perform appropriate contractions
    t *= DBraKetInd(tmp_id, 0);
    t *= DBraKetInd(tmp_id, 1);
    combineSiteBraKetToEnv(t, tmp_id, 2, id, 0);
    combineSiteBraKetToEnv(t, tmp_id, 3, id, 7);
    C_LU[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_C_LU;
      printfln(" = %s", C_LU[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(1, -1));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 1);
    t *= DBraKetInd(tmp_id, 2);
    combineSiteBraKetToEnv(t, tmp_id, 3, id, 2);
    combineSiteBraKetToEnv(t, tmp_id, 0, id, 1);
    C_RU[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_C_RU;
      printfln(" = %s", C_RU[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(-1, -1));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 2);
    t *= DBraKetInd(tmp_id, 3);
    combineSiteBraKetToEnv(t, tmp_id, 0, id, 4);
    combineSiteBraKetToEnv(t, tmp_id, 1, id, 3);
    C_RD[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_C_RD;
      printfln(" = %s", C_RD[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(-1, 1));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 3);
    t *= DBraKetInd(tmp_id, 0);
    combineSiteBraKetToEnv(t, tmp_id, 1, id, 6);
    combineSiteBraKetToEnv(t, tmp_id, 2, id, 5);
    C_LD[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_C_LD;
      printfln(" = %s", C_LD[id]);
    }

    // Construct half-row/col matrices
    tmp_id = p_cluster->vertexToId(v + Shift(-1, 0));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 0);
    contractBraKetInd(t, tmp_id, 2, id, 0);
    combineSiteBraKetToEnv(t, tmp_id, 1, id, 7);
    combineSiteBraKetToEnv(t, tmp_id, 3, id, 6);
    T_L[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_T_L;
      printfln(" = %s", T_L[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(0, -1));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 1);
    contractBraKetInd(t, tmp_id, 3, id, 1);
    combineSiteBraKetToEnv(t, tmp_id, 0, id, 0);
    combineSiteBraKetToEnv(t, tmp_id, 2, id, 1);
    T_U[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_T_U;
      printfln(" = %s", T_U[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(1, 0));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 2);
    contractBraKetInd(t, tmp_id, 0, id, 2);
    combineSiteBraKetToEnv(t, tmp_id, 1, id, 2);
    combineSiteBraKetToEnv(t, tmp_id, 3, id, 3);
    T_R[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_T_R;
      printfln(" = %s", T_R[id]);
    }

    tmp_id = p_cluster->vertexToId(v + Shift(0, 1));
    t = TBraKet(tmp_id);
    t *= DBraKetInd(tmp_id, 3);
    contractBraKetInd(t, tmp_id, 1, id, 3);
    combineSiteBraKetToEnv(t, tmp_id, 0, id, 5);
    combineSiteBraKetToEnv(t, tmp_id, 2, id, 4);
    T_D[id] = t;
    if (dbg) {
      std::cout << id << " -> " << TAG_T_D;
      printfln(" = %s", T_D[id]);
    }
  }

  // normalizePTN();

  // computeSVDspec();

  std::cout << "===== INIT_ENV_ctmrg done " << std::string(46, '=')
            << std::endl;
}

// void CtmEnv::initOBCEnv(bool dbg) {
//     std::cout <<"===== INIT_ENV_obc called "<< std::string(44,'=')
//         << std::endl;

//     //Define "contractor" tensor
//     int D = round(sqrt(d));
//     auto cI = Index("C",d);
//     auto CT = ITensor(cI);
//     for ( int i=1; i<= D*D; i++ ) {
//         CT.set(cI(i),1.0);
//     }
//     // OBC (?)
//     // for ( int i=1; i<= D*D; i++ ) {
//     //     CT.set(cI(i),1.0);
//     // }

//     if(dbg) PrintData(CT);

//     for ( size_t i=0; i<sites.size(); i++ ) {
//         if(dbg) std::cout <<"----- generating init env for site "<<
//         siteIds[i]
//             <<" -----"<< std::endl;
//         // Locate the first appearance of given site within cluster
//         int row, col;
//         for ( const auto& cToSEntry : cToS ) {
//             if ( cToSEntry.second == i) {
//                 col = cToSEntry.first.first;
//                 row = cToSEntry.first.second;
//                 break;
//             }
//         }
//         if(dbg) std::cout <<"Found "<< siteIds[i] <<" at ["<< col <<","<< row
//             <<"]"<< std::endl;

//         //Construct corner matrices
//         std::pair<int,int> site =
//             std::make_pair((col-1+sizeM)%sizeM,(row-1+sizeN)%sizeN);
//         C_LU[i] = ( ( ( sites[ cToS.at(site) ]
//             *(CT*delta(cI,I_XH)) )*(CT*delta(cI,I_XV)) )
//             *delta( prime(I_XH,1), I_U ) )
//             *delta( prime(I_XV,1), I_L);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LU;
//         printfln(" = %s", C_LU[i]); }

//         site =
//             std::make_pair((col+1)%sizeM,(row-1+sizeN)%sizeN);
//         C_RU[i] = ( ( ( sites[ cToS.at(site)]
//             *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,I_XV)) )
//             *delta( I_XH, prime(I_U,1) ) )
//             *delta( prime(I_XV,1), I_R);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RU;
//         printfln(" = %s", C_RU[i]); }

//         site =
//             std::make_pair((col+1)%sizeM,(row+1)%sizeN);
//         C_RD[i] = ( ( ( sites[ cToS.at(site)]
//             *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,prime(I_XV,1))) )
//             *delta( I_XH, prime(I_D,1) ) )
//             *delta( I_XV, prime(I_R,1));
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RD;
//         printfln(" = %s", C_RD[i]); }

//         site =
//             std::make_pair((col-1+sizeM)%sizeM,(row+1)%sizeN);
//         C_LD[i] = ( ( ( sites[ cToS.at(site)]
//             *(CT*delta(cI,I_XH)) )*(CT*delta(cI,prime(I_XV,1))) )
//             *delta( prime(I_XH,1), I_D ) )
//             *delta( I_XV, prime(I_L,1));
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LD;
//         printfln(" = %s", C_LD[i]); }

//         //Construct half-row/col matrices
//         site =
//             std::make_pair(col,(row-1+sizeN)%sizeN);
//         T_U[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XV)) )
//             *delta(I_XH, I_U) )*delta(prime(I_XH,1), prime(I_U,1));
//         T_U[i].prime(VSLINK,-1);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_U;
//         printfln(" = %s", T_U[i]); }

//         site =
//             std::make_pair((col+1)%sizeM,row);
//         T_R[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XH,1))) )
//             *delta(I_XV, I_R) )*delta(prime(I_XV,1), prime(I_R,1));
//         T_R[i].prime(I_XH,1);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_R;
//         printfln(" = %s", T_R[i]); }

//         site =
//             std::make_pair(col,(row+1)%sizeN);
//         T_D[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XV,1))) )
//             *delta(I_XH, I_D) )*delta(prime(I_XH,1), prime(I_D,1));
//         T_D[i].prime(I_XV,1);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_D;
//         printfln(" = %s", T_D[i]); }

//         site =
//             std::make_pair((col-1+sizeM)%sizeM,row);
//         T_L[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XH)) )
//             *delta(I_XV, I_L) )*delta(prime(I_XV,1), prime(I_L,1));
//         T_L[i].prime(HSLINK,-1);
//         if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_L;
//         printfln(" = %s", T_L[i]); }
//     }

//     //    normalizePTN();

//     computeSVDspec();

//     std::cout <<"===== INIT_ENV_obc done "<< std::string(46,'=')
//         << std::endl;
// }

// void CtmEnv::initPWREnv(bool dbg) {
//     double pwr = 1.0;

//     for ( auto& t : C_LU ) { for (int i=1; i<=x; i++)
//     t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double)
//     i,-pwr)); } for ( auto& t : C_RU ) { for (int i=1; i<=x; i++)
//     t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double)
//     i,-pwr)); } for ( auto& t : C_RD ) { for (int i=1; i<=x; i++)
//     t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double)
//     i,-pwr)); } for ( auto& t : C_LD ) { for (int i=1; i<=x; i++)
//     t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double)
//     i,-pwr)); } for ( auto& t : T_U ) {
//         for (int i=1; i<=x; i++) {
//             for (int j=1; j<=d; j++) { t.set(I_U(i), prime(I_U,1)(i),
//             I_XV(j), 1.0); }
//         }
//     }
//     for ( auto& t : T_R ) {
//         for (int i=1; i<=x; i++) {
//             for (int j=1; j<=d; j++) { t.set(I_R(i), prime(I_R,1)(i),
//             prime(I_XH,1)(j), 1.0); }
//         }
//     }
//     for ( auto& t : T_D ) {
//         for (int i=1; i<=x; i++) {
//             for (int j=1; j<=d; j++) { t.set(I_D(i), prime(I_D,1)(i),
//             prime(I_XV,1)(j), 1.0); }
//         }
//     }
//     for ( auto& t : T_L ) {
//         for (int i=1; i<=x; i++) {
//             for (int j=1; j<=d; j++) { t.set(I_L(i), prime(I_L,1)(i),
//             I_XH(j), 1.0); }
//         }
//     }

//     // normalizePTN();

//     computeSVDspec();

//     std::cout <<"INIT_ENV_pwr called"<< std::endl;
//     std::cout << std::string(72,'=') << std::endl;
// }

void CtmEnv::init(CtmEnv::INIT_ENV initEnvType, bool isComplex, bool dbg) {
  switch (initEnvType) {
    case CtmEnv::INIT_ENV_const1: {
      initMockEnv();
      break;
    }
    case CtmEnv::INIT_ENV_ctmrg: {
      initCtmrgEnv(dbg);
      break;
    }
    // case CtmEnv::INIT_ENV_obc: {
    //     initOBCEnv(dbg);
    //     break;
    // }
    // case CtmEnv::INIT_ENV_pwr: {
    //     initPWREnv(dbg);
    //     break;
    // }
    case CtmEnv::INIT_ENV_rnd: {
      initRndEnv(isComplex);
      break;
    }
    default: {
      std::cout << "Unsupported INIT_ENV" << std::endl;
      exit(EXIT_FAILURE);
    }
      // TODO?
      // case CtmEnv::INIT_ENV_file: {
      //     io_env_fmt_type ioEnvFmt = toIO_ENV_FMT(std::string(argv[5]));
      //     std::string in_files_prefix = std::string(argv[6]);

      //     ctmEnv = CtmEnv("TEST_ENV_2x2",
      //         readEnv_V2(ioEnvFmt, in_files_prefix, cluster),
      //         cluster);
      //     break;
      // }
  }
}

// void CtmEnv::symmetrizeEnv(bool dbg) {

//     //Define "contractor" tensor
//     int D = round(sqrt(d));
//     auto cI  = Index("C",d);
//     auto cIp = prime(cI);
//     auto CT = ITensor(cI,cIp);
//     for ( int i=0; i<D; i++ ) {
//         for ( int j=0; j<D; j++ ) {
//             CT.set(cI(1+i*D),cIp(1+i+j*D),1.0);
//         }
//     }

//     for ( size_t i=0; i<sites.size(); i++ ) {
//         if(dbg) std::cout <<"----- symmetrizing T's for site "<< siteIds[i]
//             <<" -----"<< std::endl;
//         // Locate the first appearance of given site within cluster
//         int row, col;
//         for ( const auto& cToSEntry : cToS ) {
//             if ( cToSEntry.second == i) {
//                 col = cToSEntry.first.first;
//                 row = cToSEntry.first.second;
//                 break;
//             }
//         }
//         if(dbg) std::cout <<"Found "<< siteIds[i] <<" at ["<< col <<","<< row
//             <<"]"<< std::endl;

//         //Construct half-row/col matrices
//         T_U[i] = 0.5*( T_U[i] + (T_U[i]*(delta(cI,I_XV)*CT))*delta(cIp,I_XV)
//         ); if(dbg) { printfln(" = %s", T_U[i]); }

//         T_R[i] = 0.5*( T_R[i] +
//         (T_R[i]*(delta(cI,prime(I_XH))*CT))*delta(cIp,prime(I_XH)) ); if(dbg)
//         { printfln(" = %s", T_R[i]); }

//         T_D[i] = 0.5*( T_D[i] +
//         (T_D[i]*(delta(cI,prime(I_XV))*CT))*delta(cIp,prime(I_XV)) ); if(dbg)
//         { printfln(" = %s", T_D[i]); }

//         T_L[i] = 0.5*( T_L[i] + (T_L[i]*(delta(cI,I_XH)*CT))*delta(cIp,I_XH)
//         ); if(dbg) { printfln(" = %s", T_L[i]); }
//     }
// }

// void CtmEnv::testCtmrgEnv() {

//     double m = 0.;
//     auto max_m = [&m](double d) {
//         if(std::abs(d) > m) m = std::abs(d);
//     };

//     ITensor temp;
//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = C_LU[a] - C_LU[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"C_LU["<<a<<"]-C_LU["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = C_RU[a] - C_RU[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"C_RU["<<a<<"]-C_RU["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(C_LU[0]);
//     Print(C_RU[0]);
//     ITensor ttemp = (C_RU[0] * delta(I_R, I_L))
//         * delta(prime(I_U,1),I_U);
//     temp = C_LU[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = C_RD[a] - C_RD[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"C_RD["<<a<<"]-C_RD["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(C_LU[0]);
//     Print(C_RD[0]);
//     ttemp = (C_RD[0] * delta(prime(I_R,1), I_L))
//         * delta(prime(I_D,1), I_U);
//     temp = C_LU[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = C_LD[a] - C_LD[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"C_LD["<<a<<"]-C_LD["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(C_LU[0]);
//     Print(C_LD[0]);
//     ttemp = (C_LD[0] * delta(prime(I_L,1), I_L))
//         * delta(I_D, I_U);
//     temp = C_LU[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = T_L[a] - T_L[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"T_L["<<a<<"]-T_L["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = T_U[a] - T_U[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"T_U["<<a<<"]-T_U["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(T_L[0]);
//     Print(T_U[0]);
//     ttemp = ((T_U[0] * delta(I_U, I_L))
//         * delta(prime(I_U,1), prime(I_L,1)))
//         * delta(I_XV, I_XH);
//     temp = T_L[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = T_R[a] - T_R[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"T_R["<<a<<"]-T_R["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(T_L[0]);
//     Print(T_R[0]);
//     ttemp = ((T_R[0] * delta(I_R, I_L))
//         * delta(prime(I_R,1), prime(I_L,1)))
//         * delta(prime(I_XH,1), I_XH);
//     temp = T_L[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;

//     for (int a=0; a<sites.size(); a++) {
//         for (int b=a+1; b<sites.size(); b++) {
//             temp = T_D[a] - T_D[b];
//             m = 0.;
//             temp.visit(max_m);
//             std::cout<<"T_D["<<a<<"]-T_D["<<b<<"] Max element: "<< m
//             <<std::endl;
//         }
//     }

//     Print(T_L[0]);
//     Print(T_D[0]);
//     ttemp = ((T_D[0] * delta(I_D, I_L))
//         * delta(prime(I_D,1), prime(I_L,1)))
//         * delta(prime(I_XV,1), I_XH);
//     temp = T_L[0] - ttemp;
//     m = 0.;
//     temp.visit(max_m);
//     std::cout<<"Max element: "<< m <<std::endl;
// }

// ########################################################################
// environment normalization methods

/*
 * TODO implement optimization according to largest dim of cluster
 * TODO more detailed comments about algorithm for contracting TN
 *
 * Compute the norm of canonical cluster surrounded by environment,
 * that is a cluster spanned by sites with coords
 * [0,sizeM-1] x [0, sizeN-1] (cols) x (rows)
 *
 */
// double CtmEnv::getNorm() const {
//     // Upper edge - position [0,0]
//     auto Norm = C_LU.at( cToS.at(std::make_pair(0,0)) );

//     /*
//      * Suppose sizeM >= sizeN, contract say left boundary of environment
//      *
//      *   |C_LU_00|----I_U0
//      *     |
//      *   |T_L_00|----I_XH(n-1)*2
//      *     |
//      *    I_L1>>I_L0 (via mapprime)
//      *    ...
//      *     |
//      * |T_L_0n-1|----I_XH0
//      *     |
//      *   |C_LD_0n-1|----I_D0
//      *
//      */
//     for ( int row=0; row<=sizeN-1; row++ ) {
//         Norm.prime(HSLINK,2);
//         Norm.mapprime(1,0,LLINK);
//         Norm *= T_L.at( cToS.at(std::make_pair(0,row)) );
//     }
//     Norm *= C_LD.at( cToS.at(std::make_pair(0,sizeN-1)) );

//     for ( int col=0; col<sizeM; col++ ) {
//         Norm *= T_D.at( cToS.at(std::make_pair(col,0)) );
//         Norm.mapprime(1,0,DLINK);

//          * Contract the cluster+environment column by column
//          *
//          *   |C_LU_00|----I_U0
//          *     |
//          *   |T_L_00|----I_XH(n-1)*2
//          *     |
//          *    ...              I_XV0>>I_XV1
//          *     |                |
//          * |T_L0n-1|----I_XH0--|X_0n-1  |--I_XH1
//          *     |                |
//          *     |               I_XV1>>I_XV1 (via mapprime)
//          *     |                |
//          * |C_LD_0n-1|--I_D0---|T_D_0n-1|--I_D1>>I_D0
//          *

//         for ( int row=sizeN-1; row>=0; row-- ) {
//             Norm.mapprime(0,1,VSLINK);
//             Norm *= prime(sites.at( cToS.at(std::make_pair(row,col)) ),
//             HSLINK,
//                 2*(sizeN-1-row));
//         }
//         Norm.prime(HSLINK,-1);
//         Norm *= T_U.at( cToS.at(std::make_pair(col,sizeN-1)) );
//         Norm.mapprime(1,0,ULINK);
//     }

//     /*
//      * Contract with right edge from bottom to top
//      *
//      *  |C_LU_00|--...--|T_U_m-10|--I_U1>>I_U0
//      *     |              |
//      *   |T_L_00|--...--|X_m-10  |--I_XH(n-1)*2>>I_XH(n-1)*2+m
//      *     |              |
//      *    ...            ...                         I_R0>>I_R1
//      *     |              |                           |
//      * |T_L_0n-1|--...--|X_m-1n-1|----I_XH0>>I_XHm--|T_R_m-1n-1|
//      *     |               |                          |
//      * |C_LD_0n-1|--...--|T_Dm-1n-1|--I_D1>>I_D0---|C_RD_m-1n-1|
//      *
//      */
//     Norm *= C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) );
//     for ( int row=sizeN-1; row>=0; row-- ) {
//         Norm.mapprime(2*(sizeN-1-row),1,HSLINK);
//         Norm.mapprime(0,1,RLINK);
//         Norm *= T_R.at( cToS.at(std::make_pair(sizeM-1,row)) );
//     }
//     Norm *= C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) );

//     return sumels(Norm);
// }

/*
 * Normalize the tensors By their Largest Element (BLE)
 *
 */
// void CtmEnv::normalizeBLE() {

//     auto normalizeBLE_T = [](ITensor& t)
//     {
//         double m = 0.;
//         auto max_m = [&m](double d)
//         {
//             if(std::abs(d) > m) m = std::abs(d);
//         };

//         t.visit(max_m);
//         t /= m;
//     };

//     for ( auto& t : C_LU ) { normalizeBLE_T(t); }
//     for ( auto& t : C_RU ) { normalizeBLE_T(t); }
//     for ( auto& t : C_RD ) { normalizeBLE_T(t); }
//     for ( auto& t : C_LD ) { normalizeBLE_T(t); }
//     for ( auto& t : T_U ) { normalizeBLE_T(t); }
//     for ( auto& t : T_R ) { normalizeBLE_T(t); }
//     for ( auto& t : T_D ) { normalizeBLE_T(t); }
//     for ( auto& t : T_L ) { normalizeBLE_T(t); }
// }

/*
 * Normalize the tensors By their Largest Element (BLE)
 * taking vector of half-row/column tensors and two corner tensors
 * To be used after any of directional CTM steps labeled U,R,D and L
 *
 * [int,int]=[m,n]=[col,row] of upper left cluster site (A) defining
 * the iterative step in the procedure
 *
 * CTTC
 * TABT
 * TCDT
 * CTTC
 *
 */
// void CtmEnv::normalizeBLE_ctmStep(char ctmMove, int col, int row, bool dbg) {

//     double iMinEW = isoMinElemWarning;
//     double iMaxEW = isoMaxElemWarning;

//     auto normalizeBLE_T = [&dbg,&ctmMove,&col,&row,&iMinEW,&iMaxEW](ITensor&
//     t)
//     {
//         double m = 0.;
//         auto max_m = [&m](double d)
//         {
//             if(std::abs(d) > m) m = std::abs(d);
//         };

//         t.visit(max_m);
//         if( m > iMaxEW || m < iMinEW ) {
//             std::cout << "WARNING: NormalizeBLE " << ctmMove << " [col:row]=
//             ["
//                 << col <<":"<< row <<"] Max elem.: "<< m << std::endl;
//         }

//         if(dbg) std::cout << "MAX elem = "<< m << std::endl;
//         t /= m;
//     };

//     if(dbg) std::cout <<"----- normalizeBLE_ctmStep called for "<< ctmMove
//         <<" ["<< col <<","<< row <<"]-----"<< std::endl;

//     switch(ctmMove) {
//         case 'U': { // col is ignored as we iterate over cols
//             for (int c=0; c<sizeM; c++) {
//                 normalizeBLE_T( C_LU.at( cToS.at( std::make_pair(c, row) ) )
//                 ); normalizeBLE_T( T_U.at( cToS.at( std::make_pair(c, row) )
//                 ) ); normalizeBLE_T( C_RU.at( cToS.at( std::make_pair(c, row)
//                 ) ) );
//             }
//             break;
//         }
//         case 'R': { // row is ignored as we iterate over rows
//             for (int r=0; r<sizeN; r++) {
//                 normalizeBLE_T( C_RU.at( cToS.at( std::make_pair(col, r) ) )
//                 ); normalizeBLE_T( T_R.at( cToS.at( std::make_pair(col, r) )
//                 ) ); normalizeBLE_T( C_RD.at( cToS.at( std::make_pair(col, r)
//                 ) ) );
//             }
//             break;
//         }
//         case 'D': { // col is ignored as we iterate over cols
//             for (int c=0; c<sizeM; c++) {
//                 normalizeBLE_T( C_LD.at( cToS.at( std::make_pair(c, row) ) )
//                 ); normalizeBLE_T( T_D.at( cToS.at( std::make_pair(c, row) )
//                 ) ); normalizeBLE_T( C_RD.at( cToS.at( std::make_pair(c, row)
//                 ) ) );
//             }
//             break;
//         }
//         case 'L': { // row is ignored as we iterate over rows
//             for (int r=0; r<sizeN; r++) {
//                 normalizeBLE_T( C_LU.at(
//                     cToS.at( std::make_pair(col, r) ) ) );
//                 if(dbg) std::cout <<"C_LU ["<< col <<","<< r <<"]"<<
//                 std::endl;

//                 normalizeBLE_T( T_L.at(
//                     cToS.at( std::make_pair(col, r) ) ) );
//                 if(dbg) std::cout <<"T_L ["<< col <<","<< r <<"]"<<
//                 std::endl;

//                 normalizeBLE_T( C_LD.at(
//                 cToS.at( std::make_pair(col, r) ) ) );
//                 if(dbg) std::cout <<"C_LD ["<< col <<","<< r <<"]"<<
//                 std::endl;
//             }
//             break;
//         }
//         default: {
//             std::cout <<"Unsupported ctmMove type - expecting one of "
//                 <<" U,R,D or L"<< std::endl;
//             exit(EXIT_FAILURE);
//             break;
//         }
//     }

//     if(dbg) std::cout <<"----- normalizeBLE_ctmStep for "<< ctmMove <<" END
//     -----"
//         << std::endl;
// }

/*
 * Normalize the tensors Proportional To (Frobenius) Norm (PTN)
 *
 */
// void CtmEnv::normalizePTN() {

//     for ( auto& t : C_LU ) { t /= norm(t); }
//     for ( auto& t : C_RU ) { t /= norm(t); }
//     for ( auto& t : C_RD ) { t /= norm(t); }
//     for ( auto& t : C_LD ) { t /= norm(t); }
//     for ( auto& t : T_U ) { t /= norm(t); }
//     for ( auto& t : T_R ) { t /= norm(t); }
//     for ( auto& t : T_D ) { t /= norm(t); }
//     for ( auto& t : T_L ) { t /= norm(t); }

// }

/*
 * Normalize the tensors Proportional To (Frobenius) Norm (PTN)
 * To be used after any of directional CTM steps labeled U,R,D and L
 *
 * [int,int]=[m,n]=[col,row] of upper left cluster site (A) defining
 * the iterative step in the procedure
 *
 * CTTC
 * TABT
 * TCDT
 * CTTC
 */
// void CtmEnv::normalizePTN_ctmStep(char ctmMove, int col, int row) {

//     auto normalizePTN_T = [](ITensor& t) { t /= norm(t); };

//     std::cout <<"----- normalizePTN_ctmStep called for "<< ctmMove
//         <<" ["<< col <<","<< row <<"]-----"<< std::endl;

//     switch(ctmMove) {
//         case 'U': { // col is ignored as we iterate over cols
//             for (int c=0; c<sizeM; c++) {
//                 normalizePTN_T( C_LU.at( cToS.at( std::make_pair(c, row) ) )
//                 ); normalizePTN_T( T_U.at( cToS.at( std::make_pair(c, row) )
//                 ) ); normalizePTN_T( C_RU.at( cToS.at( std::make_pair(c, row)
//                 ) ) );
//             }
//             break;
//         }
//         case 'R': { // row is ignored as we iterate over rows
//             for (int r=0; r<sizeN; r++) {
//                 normalizePTN_T( C_RU.at( cToS.at( std::make_pair(col, r) ) )
//                 ); normalizePTN_T( T_R.at( cToS.at( std::make_pair(col, r) )
//                 ) ); normalizePTN_T( C_RD.at( cToS.at( std::make_pair(col, r)
//                 ) ) );
//             }
//             break;
//         }
//         case 'D': { // col is ignored as we iterate over cols
//             for (int c=0; c<sizeM; c++) {
//                 normalizePTN_T( C_LD.at( cToS.at( std::make_pair(c, row) ) )
//                 ); normalizePTN_T( T_D.at( cToS.at( std::make_pair(c, row) )
//                 ) ); normalizePTN_T( C_RD.at( cToS.at( std::make_pair(c, row)
//                 ) ) );
//             }
//             break;
//         }
//         case 'L': { // row is ignored as we iterate over rows
//             for (int r=0; r<sizeN; r++) {
//                 normalizePTN_T( C_LU.at( cToS.at( std::make_pair(col, r) ) )
//                 ); std::cout <<"C_LU ["<< col <<","<< r <<"]"<< std::endl;

//                 normalizePTN_T( T_L.at( cToS.at( std::make_pair(col, r) ) )
//                 ); std::cout <<"T_L ["<< col <<","<< r <<"]"<< std::endl;

//                 normalizePTN_T( C_LD.at( cToS.at( std::make_pair(col, r) ) )
//                 ); std::cout <<"C_LD ["<< col <<","<< r <<"]"<< std::endl;
//             }
//             break;
//         }
//         default: {
//             std::cout <<"Unsupported ctmMove type - expecting one of "
//                 <<" U,R,D or L"<< std::endl;
//             exit(EXIT_FAILURE);
//             break;
//         }
//     }

// }

// ########################################################################
// methods handling SVD spectra of corner matrices

// void CtmEnv::computeSVDspec() {

//     ITensor S, V;
//     // indices of S matrix holding singular values
//     IndexSet iS;

//     /*
//      * Decomposing
//      *       ____
//      * I_L--|C_LU|--I_U = I_L--|U|--|S|--|V|--I_U
//      *
//      * where S is diagonal and holds the singular values
//      * of C_LU
//      *
//      */
//     auto U = ITensor(I_L);
//     for(int s=0; s<sites.size(); s++) {
//         svd( C_LU[s], U, S, V);
//         iS = S.inds();
//         for(int i=1; i<=x; i++) {
//             spec.spec_clu[s][i-1] = S.real(iS[0](i), iS[1](i));
//         }
//     }

//     /*
//      * I_U1--|C_RU|--I_R
//      *
//      */
//     U = ITensor(prime(I_U,1));
//     for(int s=0; s<sites.size(); s++) {
//         svd( C_RU[s], U, S, V);
//         iS = S.inds();
//         for(int i=1; i<=x; i++) {
//             spec.spec_cru[s][i-1] = S.real(iS[0](i), iS[1](i));
//         }
//     }

//     /*
//      * I_R1--|C_RD|--I_D1
//      *
//      */
//     U = ITensor(prime(I_R,1));
//     for(int s=0; s<sites.size(); s++) {
//         svd( C_RD[s], U, S, V);
//         iS = S.inds();
//         for(int i=1; i<=x; i++) {
//             spec.spec_crd[s][i-1] = S.real(iS[0](i), iS[1](i));
//         }
//     }

//     /*
//      * I_D--|C_LD|--I_L1
//      *
//      */
//     U = ITensor(I_D);
//     for(int s=0; s<sites.size(); s++) {
//         svd( C_LD[s], U, S, V);
//         iS = S.inds();
//         for(int i=1; i<=x; i++) {
//             spec.spec_cld[s][i-1] = S.real(iS[0](i), iS[1](i));
//         }
//     }

// }

/*
 * TODO? possibly optimize the implementation
 *
 */
// void CtmEnv::printSVDspec() const {

//     std::cout << "spec_clu: [0,0] ( ";
//     for( const auto& val : spec.spec_clu[0] ) {
//         std::cout << val << " ";
//     }
//     std::cout <<")"<< std::endl;

//     std::cout << "spec_cru: [0,0] ( ";
//     for( const auto& val : spec.spec_cru[0] ) {
//         std::cout << val << " ";
//     }
//     std::cout <<")"<< std::endl;

//     std::cout << "spec_crd: [0,0] ( ";
//     for( const auto& val : spec.spec_crd[0] ) {
//         std::cout << val << " ";
//     }
//     std::cout <<")"<< std::endl;

//     std::cout << "spec_cld: [0,0] ( ";
//     for( const auto& val : spec.spec_cld[0] ) {
//         std::cout << val << " ";
//     }
//     std::cout <<")"<< std::endl;
// }

// void CtmEnv::specDist(CtmSpec const& s1, CtmSpec const& s2) const {

//     double d_sClu = 0.0;
//     double d_sCru = 0.0;
//     double d_sCrd = 0.0;
//     double d_sCld = 0.0;

//     for(int i=0; i<x; i++) {
//         d_sClu += std::pow(std::abs(s1.spec_clu[i]
//             -s2.spec_clu[i]), 2.0);
//         d_sCru += std::pow(std::abs(s1.spec_cru[i]
//             -s2.spec_cru[i]), 2.0);
//         d_sCrd += std::pow(std::abs(s1.spec_crd[i]
//             -s2.spec_crd[i]), 2.0);
//         d_sCld += std::pow(std::abs(s1.spec_cld[i]
//             -s2.spec_cld[i]), 2.0);
//     }

//     std::cout <<"Dist(C_lu): "<< std::sqrt(d_sClu) << std::endl;
//     std::cout <<"Dist(C_ru): "<< std::sqrt(d_sCru) << std::endl;
//     std::cout <<"Dist(C_rd): "<< std::sqrt(d_sCrd) << std::endl;
//     std::cout <<"Dist(C_ld): "<< std::sqrt(d_sCld) << std::endl;
// }

// CtmEnv::CtmSpec CtmEnv::getCtmSpec() const {
//     return spec;
// }

// TODO check consistency between input cluster c and one currently
// stored in ENV
// void CtmEnv::updateCluster(Cluster const& c) {

//     for( unsigned int i=0; i < siteIds.size(); i++ ) {
//         sites[i] = contractOST(c.sites.at(siteIds[i]));
//     }
// }
void CtmEnv::updateCluster(Cluster const& c) {
  p_cluster = &c;
}

// CtmData_Full CtmEnv::getCtmData_Full_DBG(bool dbg) const {
//     // Indexing of T_* and C_* arrays wrt environment
//     // of non-equivalent sites

//     if(dbg) { std::cout <<"sites: ["<< std::endl;
//     for( std::size_t i=0; i<sites.size(); i++) {
//         std::cout << WS4 << siteIds[i] <<" -> "<< i << std::endl;
//     }
//     std::cout <<"]"<< std::endl; }

//     CtmData_Full ctmDataFull = {
//         x, d, sizeN, sizeM,
//         sites, cToS,
//         T_U, T_R, T_D, T_L,
//         C_LU, C_RU, C_RD, C_LD,
//         I_U, I_R, I_D, I_L,
//         I_XH, I_XV };
//     return ctmDataFull;
// }

// ############################################################################
// IO toString methods

std::ostream& CtmEnv::print(std::ostream& s) const {
  s << "CtmEnv( " << std::endl
    << m_name << ", d=" << d << ", x=" << x << std::endl
    << "cluster(NxM): " << sizeN << "(row) x " << sizeM << "(column)"
    << std::endl;

  s << "clusterToSite: [" << std::endl;
  for (const auto& cToSEntry : cToS) {
    s << WS4 << "(" << cToSEntry.first.first << ", " << cToSEntry.first.second
      << ") -> " << cToSEntry.second << " -> " << siteIds[cToSEntry.second]
      << std::endl;
  }
  s << "]" << std::endl;

  s << "sites: [" << std::endl;
  for (std::size_t i = 0; i < sites.size(); i++) {
    s << WS4 << siteIds[i] << " -> " << i;
    printfln(" = %s", sites[i]);
  }
  s << "]" << std::endl;

  s << "eaux: [" << std::endl;
  for (auto const& e : eaux) {
    s << e.first << " : ";
    for (auto const& i : e.second)
      s << i << " ";
    s << std::endl;
  }
  s << "]" << std::endl;

  s << "CMB: [" << std::endl;
  for (auto const& e : CMB) {
    s << e.first << " : ";
    for (int dir = 0; dir < 4; dir++)
      s << "dir " << dir << " : " << e.second[dir];
  }
  s << "]" << std::endl;

  s << "faux: [" << std::endl;
  for (auto const& e : faux) {
    s << e.first << " : ";
    for (auto const& i : e.second)
      s << i << " ";
    s << std::endl;
  }
  s << "]" << std::endl;

  // Loop over inequivalent sites and print their environment
  for (const auto& id : p_cluster->siteIds) {
    s << "===="
      << " BEGIN ENV OF SITE " << id << " " << std::string(47, '=')
      << std::endl;
    s << "----" << id << " start CORNER TENSORS----" << std::endl;
    s << TAG_C_LU;
    printfln(" = %s", C_LU.at(id));
    s << TAG_C_RU;
    printfln(" = %s", C_RU.at(id));
    s << TAG_C_RD;
    printfln(" = %s", C_RD.at(id));
    s << TAG_C_LD;
    printfln(" = %s", C_LD.at(id));
    s << "------end CORNER TENSORS----" << std::endl;

    s << "----" << id << " start HALF-ROW TENSORS--" << std::endl;
    s << TAG_T_L << id;
    printfln(" = %s", T_L.at(id));
    s << TAG_T_R << id;
    printfln(" = %s", T_R.at(id));
    s << "------end HALF-ROW TENSORS--" << std::endl;

    s << "----" << id << " start HALF-COL TENSORS--" << std::endl;
    s << TAG_T_U << id;
    printfln(" = %s", T_U.at(id));
    s << TAG_T_D << id;
    printfln(" = %s", T_D.at(id));
    s << "------end HALF-COL TENSORS--" << std::endl;
    s << "===="
      << " END ENV OF SITE " << id << " " << std::string(49, '=') << std::endl;
  }

  return s;
}

std::ostream& operator<<(std::ostream& s, CtmEnv const& c) {
  return c.print(s);
}

// std::ostream& operator<<(std::ostream& s, CtmEnv::CtmSpec const& spec) {
//     s << "C_lu [0,0] ";
//     for (std::complex<double> const& sv : spec.spec_clu[0]) {
//         s << sv.real() << ' ';
//     }
//     s << std::endl;
//     s << "C_ru: [0,0]"; //["<< sizeM-1 <<",0] "<<;
//     for (std::complex<double> const& sv : spec.spec_cru[0]) {
//         s << sv.real() << ' ';
//     }
//     s << std::endl;
//     s << "C_rd: [0,0]"; //["<< sizeM-1 <<","<< sizeN-1 <<"] ";
//     for (std::complex<double> const& sv : spec.spec_crd[0]) {
//         s << sv.real() << ' ';
//     }
//     s << std::endl;
//     s << "C_ld: [0,0]"; //[0,"<< sizeN-1 <<"] ";
//     for (std::complex<double> const& sv : spec.spec_cld[0]) {
//         s << sv.real() << ' ';
//     }
//     s << std::endl;
//     return s;
// }

/*
 * TODO Perhaps(?) move to ctm-cluster-io
 *
 */
CtmEnv::DIRECTION toDIRECTION(int i_dir) {
  switch (i_dir) {
    case 0:
      return CtmEnv::DIRECTION::LEFT;
    case 1:
      return CtmEnv::DIRECTION::UP;
    case 2:
      return CtmEnv::DIRECTION::RIGHT;
    case 3:
      return CtmEnv::DIRECTION::DOWN;
    default:
      throw std::runtime_error("[toDIRECTION] invalid integer for direction");
  }
}

CtmEnv::INIT_ENV toINIT_ENV(std::string const& iE) {
  if (iE == "INIT_ENV_const1")
    return CtmEnv::INIT_ENV_const1;
  if (iE == "INIT_ENV_rnd")
    return CtmEnv::INIT_ENV_rnd;
  if (iE == "INIT_ENV_ctmrg")
    return CtmEnv::INIT_ENV_ctmrg;
  if (iE == "INIT_ENV_obc")
    return CtmEnv::INIT_ENV_obc;
  if (iE == "INIT_ENV_pwr")
    return CtmEnv::INIT_ENV_pwr;
  if (iE == "INIT_ENV_file")
    return CtmEnv::INIT_ENV_file;
  std::cout << "Unsupported INIT_ENV" << std::endl;
  exit(EXIT_FAILURE);
}

CtmEnv::ISOMETRY toISOMETRY(std::string const& isoType) {
  if (isoType == "ISOMETRY_T1")
    return CtmEnv::ISOMETRY_T1;
  if (isoType == "ISOMETRY_T2")
    return CtmEnv::ISOMETRY_T2;
  if (isoType == "ISOMETRY_T3")
    return CtmEnv::ISOMETRY_T3;
  if (isoType == "ISOMETRY_T4")
    return CtmEnv::ISOMETRY_T4;
  std::cout << "Unsupported ISOMETRY" << std::endl;
  exit(EXIT_FAILURE);
}

CtmEnv::NORMALIZATION toNORMALIZATION(std::string const& normType) {
  if (normType == "NORM_BLE")
    return CtmEnv::NORM_BLE;
  if (normType == "NORM_PTN")
    return CtmEnv::NORM_PTN;
  std::cout << "Unsupported NORMALIZATION" << std::endl;
  exit(EXIT_FAILURE);
}
