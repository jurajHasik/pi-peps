#include "ctm-cluster-env_v2.h"

// TODO Implement convergence check as general function. The actual
// implementation may vary - difference between SVD decomp,
// Eigenvalue decomp, etc.

// TODO Implement Isometry as a general function returning the
// isometry tensor Z - the actual used implementation may vary
// SVD decomp, Eigenvalue decomp, etc.

using namespace itensor;

// ############################################################################
// member methods of CtmEnv 

CtmEnv::CtmEnv () {}

CtmEnv::CtmEnv (std::string t_name, int t_x, Cluster const& c) 
    : m_name(t_name), d(c.auxBondDim*c.auxBondDim), x(t_x), sizeN(c.sizeN), 
    sizeM(c.sizeM) {

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
    I_U = Index(TAG_I_U, x, ULINK); 
    I_R = Index(TAG_I_R, x, RLINK);   
    I_D = Index(TAG_I_D, x, DLINK);
    I_L = Index(TAG_I_L, x, LLINK);
    
    // Define indices between row/column tensors "T_*" and on-site 
    // tensors "X**"
    I_XH = Index(TAG_I_XH, d, HSLINK);
    I_XV = Index(TAG_I_XV, d, VSLINK);

    for (std::size_t i=0; i<c.siteIds.size(); i++) {
        // Construct tensors "C_*" for every non-eq cluster site
        C_LU.push_back( ITensor(I_L, I_U) );
        C_RU.push_back( ITensor(prime(I_U,1), I_R) );
        C_RD.push_back( ITensor(prime(I_R,1), prime(I_D,1)) );
        C_LD.push_back( ITensor(I_D, prime(I_L,1)) );    
    
        // Construct tensors "T_*"
        T_L.push_back( ITensor(I_L, prime(I_L,1), I_XH) );
        T_R.push_back( ITensor(I_R, prime(I_R,1), 
            prime(I_XH,1)) );
        T_U.push_back( ITensor(I_U, prime(I_U,1), I_XV) );
        T_D.push_back( ITensor(I_D, prime(I_D,1),
            prime(I_XV,1)) );
    }
    
    // Iterate over siteIds as given in Cluster c
    for( const auto& siteIdEntry : c.siteIds ) {
        sites.push_back( contractOST(c.sites.at(siteIdEntry)) );
        siteIds.push_back( siteIdEntry );

        // add entries for map from cluster site to tensor 
        for ( const auto& cToSEntry : c.cToS ) {
            if( cToSEntry.second == siteIdEntry ) {
                cToS[ cToSEntry.first ] = sites.size()-1;
            }
        }
    }

    // Construct vectors holding SVD spectrum of corner matrices
    spec = {
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x)
    };
}

/*
 * Assume the whole environment is delivered as a set of simple
 * environments for each of non-equivalent sites within cluster
 * Preserves compatibility with LEGACY ctm-cluster-io.h
 */
CtmEnv::CtmEnv (std::string t_name,  std::vector<CtmData> const& ctmD, 
    Cluster const& c)
    : m_name(t_name), d(ctmD[0].auxDimSite), x(ctmD[0].auxDimEnv),
    sizeN(c.sizeN), sizeM(c.sizeM),
    // environment indices
    I_U(ctmD[0].I_U), I_R(ctmD[0].I_R), I_D(ctmD[0].I_D), I_L(ctmD[0].I_L),
    // on-site indices
    I_XH(ctmD[0].I_XH), I_XV(ctmD[0].I_XV) {

    for (std::size_t i=0; i<ctmD.size(); i++) {
        // corner matrices 
        C_LU.push_back(ctmD[i].C_LU);
        C_RU.push_back(ctmD[i].C_RU);
        C_RD.push_back(ctmD[i].C_RD);
        C_LD.push_back(ctmD[i].C_LD);
        // half-row/column tensors
        T_U.push_back(ctmD[i].T_U[0]);
        T_R.push_back(ctmD[i].T_R[0]);
        T_D.push_back(ctmD[i].T_D[0]);
        T_L.push_back(ctmD[i].T_L[0]);
    }

    if ( ctmD[0].auxDimSite != c.auxBondDim*c.auxBondDim ) {
        std::cout <<"AuxDimSite of environment and auxBondDim**2 "
            <<"of cluster are not in agreement"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Iterate over siteIds as given in Cluster c
    for( const auto& siteIdEntry : c.siteIds ) {
        sites.push_back( contractOST(c.sites.at(siteIdEntry)) );
        siteIds.push_back( siteIdEntry );

        // add entries for map from cluster site to tensor 
        for ( const auto& cToSEntry : c.cToS ) {
            if( cToSEntry.second == siteIdEntry ) {
                cToS[ cToSEntry.first ] = sites.size()-1;
            }
        }
    }

    // Construct vectors holding SVD spectrum of corner matrices
    spec = {
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x)
    };
}

ITensor CtmEnv::contractOST(ITensor const& T) const {
    /*
     * Construct on-site tensor X given by the contraction of bra & ket 
     * on-site tensors T^dag & T through physical index physI
     * 
     *  auxI5  auxI6  auxI' auxI''          auxI5(x)auxI' 
     *      | /           | /                     ||
     *     |T*|~~~physI~~|T|  =>   auxI4(x)auxI==|T*T|==auxI6(x)auxI''
     *     / |           / |                      ||
     * auxI4 auxI7    auxI auxI3            auxI7(x)auxI3 
     *
     * where indices auxI[0..3] correspond to ket on-site tensor T and
     * indices auxI[4..7] to ket. (x) denotes a tensor product of indices.
     * To obtain final form of X we need to cast tensor product of 
     * indices, say auxI4(x)auxI, into a single index I_XH
     * of size dim(I_XH) = dim(auxI4)*dim(auxI) = D*D = D^2 = d
     * in accordance with CTM tensor network as defined above. 
     * Therefore we define a conversion tensor Y as follows
     *            _
     *    auxI4--| \
     *           |Y --I_XH =: Y(h)
     *     auxI--|_/
     *
     * with only non-zero elements being
     *
     *   Y(auxI4=i, auxI=j, I_XH)=D*(i-1)+j) = 1.0
     *
     * then we obtain X with proper indices as
     *
     * X = Y(h)*Y(h')*Y(v)*Y(v')*|T*T|
     *
     */ 

    // Get auxBond index of T
    auto auxI = noprime(findtype(T.inds(), AUXLINK));

    if(auxI.m()*auxI.m() != d) {
        std::cout <<"ctmEnv.d does not agree with onSiteT.dimD^2"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Define combiner tensors Y*
    auto C04 = combiner(auxI, prime(auxI,4));
    auto C15 = prime(C04,1);
    auto C26 = prime(C04,2);
    auto C37 = prime(C04,3);

    auto X = (T*( conj(T).prime(AUXLINK,4) ))*C04*C15*C26*C37;

    // Define delta tensors D* to relabel combiner indices to I_XH, I_XV
    auto DH0 = delta(I_XH, commonIndex(X,C04));
    auto DV0 = delta(I_XV, commonIndex(X,C15));
    auto DH1 = delta(prime(I_XH,1), commonIndex(X,C26));
    auto DV1 = delta(prime(I_XV,1), commonIndex(X,C37));

    return X*DH0*DV0*DH1*DV1;
}

// ########################################################################
// environment initalization methods

void CtmEnv::initMockEnv() {
    for ( auto& t : C_LU ) { t.fill(1.+0._i); }
    for ( auto& t : C_RU ) { t.fill(1.+0._i); }
    for ( auto& t : C_RD ) { t.fill(1.+0._i); }
    for ( auto& t : C_LD ) { t.fill(1.+0._i); }
    for ( auto& t : T_U ) { t.fill(1.+0._i); }
    for ( auto& t : T_R ) { t.fill(1.+0._i); }
    for ( auto& t : T_D ) { t.fill(1.+0._i); }
    for ( auto& t : T_L ) { t.fill(1.+0._i); }

    normalizePTN();

    CtmEnv::computeSVDspec();

    std::cout <<"INIT_ENV_const1 with all elements of C's and T's"<<
        " set to constant"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

void CtmEnv::initRndEnv(bool isComplex) {
    //Fill with random numbers (real or complex)
    for ( auto& t : C_LU ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_RU ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_RD ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_LD ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_U ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_R ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_D ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_L ) { randomize(t, {"Complex", isComplex}); }

    normalizePTN();

    computeSVDspec();

    std::cout <<"INIT_ENV_rnd with all C's and T's random (complex ? " 
        << isComplex <<")"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

// ########################################################################
// CTM iterative methods

// void CtmEnv::insURow_DBG(CtmEnv::ISOMETRY iso_type, 
//     CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
// {
//     std::cout <<"##### InsURow called "<< std::string(51,'#') << std::endl;

//     // sequentialy contract upper boundary of environment with 
//     // sizeN rows of cluster + half-row matrices T_L* and T_R*
//     for (int row=0; row<sizeN; row++) {
//         std::chrono::steady_clock::time_point t_iso_begin = 
//             std::chrono::steady_clock::now();
//         /*
//          * Insert a copy of row into network to obtain following expanded TN:
//          *     ____         ____                 ______          ____
//          *  A |C_LU|--I_U--|T_U0|--...--I_Um-1--|T_Um-1|--I_Um--|C_RU|  A
//          *  A   |            |                    |               |     A
//          *  A  I_L0         I_XV                 I_XV            I_R0   A
//          *  A  _|__                                              _|__   A 
//          *  A |T_L0|--I_XH                               I_XHm--|T_R0|  A    
//          *  A   |                                                 |     A
//          *  A  I_L1                                              I_R1   A
//          *
//          * Focus on a part of the TN highlighted by A's. Absorb the 
//          * inserted tensors to obtain TN of the original form
//          *
//          * 1) Contract C_LU with T_L(row)
//          *   ____                 ____
//          *  |C_LU|--I_U          |    |--I_U
//          *    |               => |C_LU| 
//          *   I_L(row)         => |____|--I_XH
//          *   _|______         =>    |
//          *  |T_L(row)|--I_XH       I_L(row+1)
//          *    |
//          *   I_L(row+1)
//          * 
//          * 2) Contract C_RU with T_R(row)
//          *         ____                  ____
//          *  I_Um--|C_RU|          I_Um--|    |
//          *          |         =>        |C_RU|  
//          *         I_R(row)   => I_XHm--|____|
//          *         _|______   =>           |
//          * I_XHm--|T_R(row)|              I_R(row+1) 
//          *          |
//          *         I_R(row+1)
//          *
//          */
//         std::cout <<"(1) ----- C_LU & T_L"<< row <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_LU);
//         Print(T_L[row]);

//         C_LU *= T_L[row];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_LU);

//         std::cout <<"(2) ----- C_RU & T_R"<< row <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_RU);
//         Print(T_R[row]);

//         C_RU *= T_R[row];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_RU);

//         /* 
//          * Contract T_U[0..sizeM-1] with X[row,col] to obtain new T_U
//          *            ___                           ___
//          * I_U(col)--|T_U|--I_U(col+1)   I_U(col)--|   |--I_U(col+1)
//          *             |                           |T_U|  
//          *            I_XV         =>        I_XH--|___|--I_XH1
//          *            _|_          =>                |
//          *     I_XH--| X |--I_XH1                   I_XV1 
//          *             |
//          *            I_XV1
//          *
//          */
//         std::cout <<"(3) ----- T_U & X "<< std::string(54,'-') << std::endl;
//         for (int col=0; col<sizeM; col++) {
//             std::cout <<"--- Before contraction T_U["<< col <<"] & X["<< row
//                 <<","<< col <<"] ---"<< std::endl;
//             std::cout << TAG_T_U <<"["<< col <<"]";
//             printfln("= %s",T_U[col]);
//             std::cout <<"("<< row <<","<< col <<") -> sites["<< 
//                 cToS[std::make_pair(row,col)] <<"]"<< std::endl;

//             T_U[col] *= sites[cToS[std::make_pair(row,col)]];
//             T_U[col].prime(HSLINK,col);
//             T_U[col].noprime(VSLINK);

//             std::cout <<"After contraction Col="<< col << std::endl;
//             std::cout << TAG_T_U <<"["<< col <<"]";
//             printfln("= %s",T_U[col]);
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('U');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('U');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         std::chrono::steady_clock::time_point t_iso_end =
//             std::chrono::steady_clock::now();
//         accT[0] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * compute one of the "trivial" isometries
//          *
//          * TODO? how to include more elaborate isometries
//          * TODO? call isometry creation as a function
//          *
//          */
//         std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;

//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_U, I_XH), 
//                     std::make_pair(prime(I_L,row+1),prime(I_R,row+1)),
//                     C_LU, C_RU);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('U');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('U');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('U');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[1] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * Obtain new C_LU,C_RU by contraction of C_LU,C_RU 
//          * with an isometry U,V and reduction matrix R
//          *   ____          _          
//          *  |    |--I_U---| \
//          *  |C_LU|        |tU--I_SVD--|R|--I_U = |C_LU|--I_U  
//          *  |____|--I_XH--|_/                      |
//          *    |                                   I_L(row+1)
//          *   I_L(row+1)
//          *                                     _           ____ 
//          *                                    / |---I_Um--|    |
//          *   I_Um--|C_RU|= I_Um--|R|--I_SVD1--tV|         |C_RU|
//          *           |                        \_|--I_XHm--|____|
//          *          I_R(row+1)                              |
//          *                                                 I_R(row+1)
//          * 
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(5) ----- Construct reduced C_LU,C_RU -----"<< std::endl;
//         /*auto I_SVD_U = findtype(tU_tV.first.inds(),SVD_LINK);
//         auto I_SVD_V = findtype(tU_tV.second.inds(),SVD_LINK);

//         C_LU = (C_LU*tU_tV.first )*delta(I_SVD_U,I_U);
//         C_RU = (C_RU*tU_tV.second )*delta(I_SVD_V,prime(I_U,sizeM));

//         Print(C_LU);
//         Print(C_RU);*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[2] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /* 
//          * Contract T_U[0..sizeM-1] with X[row,col] to obtain new T_U
//          *            ___                           ___
//          * I_U(col)--|T_U|--I_U(col+1)   I_U(col)--|   |--I_U(col+1)
//          *             |                           |T_U|  
//          *            I_XV         =>        I_XH--|___|--I_XH1
//          *            _|_          =>                |
//          *     I_XH--| X |--I_XH1                   I_XV1 
//          *             |
//          *            I_XV1
//          *
//          * Obtain new T_U by contraction of T_U with an isometry tU, tV and
//          * reduction matrix R as follows (since C_LU was reduced by tU--R--
//          * from the right)
//          *                          _              ___ 
//          *                         / |--I_U(col)--|   |--I_U(col+1)
//          *  I_U(col)--|R|--I_SVD1--tV|            |T_U|
//          *                         \_|------I_XH--|___|--I_XH1
//          *                                          |                     
//          *                                        I_XV1
//          * and from the right
//          *            ___                _
//          * I_U(col)--|   |--I_U(col+1)--| \
//          *           |T_U|              |tU--I_SVD--|R|--I_U(col+1)  
//          *           |___|--I_XH1-------|_/
//          *             |
//          *            I_XV1 
//          *
//          *
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(6) ----- REDUCE T_U & X "<< std::string(47,'-') << 
//             std::endl;
//         /*tU_tV.second.noprime();
//         tU_tV.first.prime(ULINK,HSLINK);
//         for (int col=0; col<sizeM; col++) {
//             std::cout <<" --- Construct reduced T_U["<< col <<"] ---"
//                 << std::endl;

//             T_U[col] = ( ( ( T_U[col]*tU_tV.second.mapprime(ULINK,col-1,col) ) 
//                 *delta(I_SVD_V,prime(I_U,col)) )
//                 *tU_tV.first.mapprime(ULINK,col,col+1) )
//                 *delta(I_SVD_U,prime(I_U,col+1));

//             T_U[col].noprime(VSLINK);
//             std::cout << TAG_T_U <<"["<< col <<"]";
//             printfln("= %s",T_U[col]);
//         }*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[3] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         std::cout <<"Row "<< row <<" done"<< std::endl;
//     }

//     // End of cluster absorption
//     C_LU.noprime(LLINK);
//     C_RU.noprime(RLINK);
 
//     std::cout <<"##### InsURow Done "<< std::string(53,'#') << std::endl;
// }

// void CtmEnv::insURow(CtmEnv::ISOMETRY iso_type, 
//     CtmEnv::NORMALIZATION norm_type) {
//     for (int row=0; row<sizeN; row++) {

//         C_LU *= T_L[row];
//         C_RU *= T_R[row];

//         for (int col=0; col<sizeM; col++) {
//             T_U[col] *= sites[cToS[std::make_pair(row,col)]];
//             T_U[col].prime(HSLINK,col);
//             T_U[col].noprime(VSLINK);
//         }

//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_U, I_XH), 
//                     std::make_pair(prime(I_L,row+1),prime(I_R,row+1)),
//                     C_LU, C_RU);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('U');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('U');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('U');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('U');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('U');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }
//     }

//     // End of cluster absorption
//     C_LU.noprime(LLINK);
//     C_RU.noprime(RLINK);
// }

// void CtmEnv::insDRow_DBG(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
// {
//     std::cout <<"##### InsDRow called "<< std::string(51,'#') << std::endl;
//     // sequentialy contract lower boundary of environment with 
//     // sizeN rows of cluster + half-row matrices T_L* and T_R*
//     for (int row=sizeN-1; row>=0; row--) {
//         std::chrono::steady_clock::time_point t_iso_begin = 
//             std::chrono::steady_clock::now();
//         /*
//          * Insert a copy of row into network to obtain following expanded TN:
//          *
//          *  A  I_Ln-1                                            I_Rn-1  A
//          *  A  _|____                                            _|____  A 
//          *  A |T_Ln-1|--I_XH                             I_XHm--|T_Rn-1| A    
//          *  A   |                                                 |      A
//          *  A  I_Ln         I_XVn                I_XVn           I_Rn    A
//          *  A  _|__         _|__                 _|____          _|__    A
//          *  A |C_LD|--I_D--|T_D0|--...--I_Dm-1--|T_Dm-1|--I_Dm--|C_RD|   A
//          *
//          * Focus on a part of the TN highlighted by A's. Absorb the 
//          * inserted tensors to obtain TN of the original form
//          *
//          * 1) Contract C_LU with T_L(row)
//          *
//          *   I_L(row)            I_(row)
//          *   _|______            _|__
//          *  |T_L(row)|--I_XH    |    |--I_XH
//          *    |              => |C_LD| 
//          *   I_L(row+1)      => |____|--I_D
//          *   _|__            => 
//          *  |C_LD|------I_D
//          *
//          * 2) Contract C_RU with T_R(row)
//          *         
//          *         I_R(row)               I_R(row)
//          *         _|______               _|__
//          * I_XHm--|T_R(row)|      I_XHm--|    |
//          *          |         =>         |C_RU|  
//          *         I_R(row+1) =>   I_Dm--|____|
//          *         _|__       =>
//          *  I_Dm--|C_RD|
//          *
//          */
//         std::cout <<"(1) ----- C_LD & T_L"<< row <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_LD);
//         Print(T_L[row]);

//         C_LD *= T_L[row];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_LD);

//         std::cout <<"(2) ----- C_RD & T_R"<< row <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_RD);
//         Print(T_R[row]);

//         C_RD *= T_R[row];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_RD);

//         std::cout <<"(3) ----- T_D & X "<< std::string(54,'-') << std::endl;
//         for (int col=0; col<sizeM; col++) {
//             std::cout <<"--- Before contraction T_D["<< col <<"] & X["<< row
//                 <<","<< col <<"] ---"<< std::endl;
//             std::cout << TAG_T_D <<"["<< col <<"]";
//             printfln("= %s",T_D[col]);
//             std::cout <<"("<< row <<","<< col <<") -> sites["<< 
//                 cToS[std::make_pair(row,col)] <<"]"<< std::endl;

//             T_D[col].mapprime(sizeN,1,VSLINK);
//             T_D[col] *= sites[cToS[std::make_pair(row,col)]];
//             T_D[col].prime(HSLINK,col);
//             T_D[col].prime(VSLINK,sizeN);

//             std::cout <<"After contraction Col="<< col << std::endl;
//             std::cout << TAG_T_D <<"["<< col <<"]";
//             printfln("= %s",T_D[col]);
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('D');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('D');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         std::chrono::steady_clock::time_point t_iso_end =
//             std::chrono::steady_clock::now();
//         accT[0] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * compute one of the "trivial" isometries
//          *
//          */
//         std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_D, I_XH), 
//                     std::make_pair( prime(I_L,row),prime(I_R,row) ),
//                     C_LD, C_RD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('D');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('D');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('D');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[1] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * Obtain new C_LD,C_RD by contraction of C_LD,C_RD 
//          * with an isometry U,V and reduction matrix R
//          *
//          *   I_L(row)
//          *   _|__          _                      I_L(row)
//          *  |    |--I_D---| \                      |
//          *  |C_LD|        |tU--I_SVD--|R|--I_D = |C_LD|--I_D  
//          *  |____|--I_XH--|_/  
//          *
//          *                                                 I_R(row)
//          *          I_R(row)                   _           _|__ 
//          *           |                        / |---I_Dm--|    |
//          *   I_Dm--|C_RD|= I_Dm--|R|--I_SVD1--tV|         |C_RD|
//          *                                    \_|--I_XH1--|____|
//          * 
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(5) ----- Construct reduced C_LD,C_RD -----"<< std::endl;
//         /*auto I_SVD_U = findtype(tU_tV.first.inds(),SVD_LINK);
//         auto I_SVD_V = findtype(tU_tV.second.inds(),SVD_LINK);

//         C_LD = (C_LD*tU_tV.first)*delta(I_SVD_U,I_D);
//         C_RD = (C_RD*tU_tV.second)*delta(I_SVD_V,prime(I_D,sizeM));

//         Print(C_LD);
//         Print(C_RD);*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[2] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
        
//         /* 
//          * Contract T_D[0..sizeM-1] with X[row,col] to obtain new T_D
//          *
//          *            I_XV                           I_XV  
//          *            _|_                            _|_
//          *     I_XH--| X |--I_XH1             I_XH--|   |--I_XH1
//          *             |                            |T_D|  
//          *            I_XV1            => I_D(col)--|___|--I_D(col+1)
//          *            _|_              =>
//          * I_D(col)--|T_D |--I_D(col+1)
//          *
//          * Obtain new T_D by contraction of T_D with an isometry tU, tV and
//          * reduction matrix R as follows (since C_LD was reduced by tU--R--
//          * from the right)
//          *
//          *                                         I_XV 
//          *                          _              _|_ 
//          *                         / |--I_XH------|   |--I_XH1
//          *  I_D(col)--|R|--I_SVD1--tV|            |T_D|
//          *                         \_|--I_D(col)--|___|--I_D(col+1)
//          * and from the right
//          *
//          *            I_XV
//          *            _|_                _
//          * I_D(col)--|   |--I_XH1-------| \
//          *           |T_D|              |tU--I_SVD--|R|--I_D(col+1)  
//          *           |___|--I_D(col+1)--|_/
//          *
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(6) ----- REDUCE T_D & X "<< std::string(47,'-') << 
//             std::endl;
//         /*tU_tV.second.noprime();
//         tU_tV.first.prime(DLINK,HSLINK);
//         for (int col=0; col<sizeM; col++) {
//             std::cout <<" --- Construct reduced T_D["<< col <<"] ---"
//                 << std::endl;
            
//             T_D[col] = ( ( ( T_D[col]*tU_tV.second.mapprime(DLINK,col-1,col) )
//                 *delta(I_SVD_V,prime(I_D,col)) )
//                 *tU_tV.first.mapprime(DLINK,col,col+1) )
//                 *delta(I_SVD_U,prime(I_D,col+1));

//             T_D[col].prime(VSLINK);
//             std::cout << TAG_T_D <<"["<< col <<"]";
//             printfln("= %s",T_D[col]);
//         }*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[3] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         std::cout <<"Row "<< row <<" done"<< std::endl;
//     }

//     // End of cluster absorption
//     C_LD.prime(LLINK,sizeN);
//     C_RD.prime(RLINK,sizeN);

//     std::cout <<"##### InsDRow Done "<< std::string(53,'#') << std::endl;
// }

// void CtmEnv::insDRow(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type) {
//     for (int row=sizeN-1; row>=0; row--) {

//         C_LD *= T_L[row];
//         C_RD *= T_R[row];

//         for (int col=0; col<sizeM; col++) {
//             T_D[col].mapprime(sizeN,1,VSLINK);
//             T_D[col] *= sites[cToS[std::make_pair(row,col)]];
//             T_D[col].prime(HSLINK,col);
//             T_D[col].prime(VSLINK,sizeN);
//         }

//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;

//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_D, I_XH), 
//                     std::make_pair( prime(I_L,row),prime(I_R,row) ),
//                     C_LD, C_RD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('D');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('D');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('D');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('D');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('D');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }
//     }

//     // End of cluster absorption
//     C_LD.prime(LLINK,sizeN);
//     C_RD.prime(RLINK,sizeN);
// }

// void CtmEnv::insLCol_DBG(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT)
// {
//     std::cout <<"##### InsLCol called "<< std::string(51,'#') << std::endl;
//     // sequentialy contract left boundary of environment with 
//     // sizeM rows of cluster + half-row matrices T_U* and T_D*
//     for (int col=0; col<sizeM; col++) {
//         std::chrono::steady_clock::time_point t_iso_begin = 
//             std::chrono::steady_clock::now();
//         /*
//          * Insert a copy of column into network to obtain following expanded TN:
//          *
//          *  AAAAAAAAAAAAAAAAAAAAAAAAAAAA
//          *   ____              ________      
//          *  |C_LU|--I_U(col)--|T_U(col)|--I_U(col+1)
//          *    |                 |
//          *   I_L               I_XV
//          *   _|__
//          *  |T_L0|--I_XH   
//          *    |
//          *   I_L1
//          *    |
//          *   ...
//          *    |
//          *   I_Ln-1
//          *   _|
//          *  |T_Ln-1|--I_XH
//          *    |
//          *   I_Ln              I_XVn  
//          *   _|__              _|______
//          *  |C_LD|--I_D(col)--|T_D(col)|--I_D(col+1)
//          *
//          *  AAAAAAAAAAAAAAAAAAAAAAAAAAAA   
//          *
//          * Focus on a part of the TN highlighted by A's. Absorb the 
//          * inserted tensors to obtain TN of the original form
//          *
//          * 1) Contract C_LU with T_U(col)
//          *   ____            ___                 ______
//          *  |C_LU|-I_U(col)-|T_U|-I_U(col+1) => |C_LU__|--I_U(col+1)  
//          *    |               |                   |   |    
//          *   I_L             I_XV                I_L I_XV    
//          *
//          * 2) Contract C_LD with T_D(col)
//          *
//          *   I_Ln              I_XVn                I_Ln  I_XVn
//          *   _|__              _|_                  _|_____|_
//          *  |C_LD|--I_D(col)--|T_D|--I_U(col+1) => |___C_LD__|--I_U(col+1)
//          *
//          */
//         std::cout <<"(1) ----- C_LU & T_U"<< col <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_LU);
//         Print(T_U[col]);

//         C_LU *= T_U[col];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_LU);

//         std::cout <<"(2) ----- C_LD & T_D"<< col <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_LD);
//         Print(T_D[col]);

//         C_LD *= T_D[col];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_LD);

//         std::cout <<"(3) ----- T_L & X "<< std::string(54,'-') << std::endl;
//         for (int row=0; row<sizeN; row++) {
//             std::cout <<"--- Before contraction T_L["<< row <<"] & X["<< row
//                 <<","<< col <<"] ---"<< std::endl;
//             std::cout << TAG_T_L <<"["<< row <<"]";
//             printfln("= %s",T_L[row]);
//             std::cout <<"("<< row <<","<< col <<") -> sites["<< 
//                 cToS[std::make_pair(row,col)] <<"]"<< std::endl;

//             T_L[row] *= sites[cToS[std::make_pair(row,col)]];
//             T_L[row].prime(VSLINK,row);
//             T_L[row].noprime(HSLINK);

//             std::cout <<"After contraction Row="<< row << std::endl;
//             std::cout << TAG_T_L <<"["<< row <<"]";
//             printfln("= %s",T_L[row]);
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('L');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('L');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         std::chrono::steady_clock::time_point t_iso_end = 
//             std::chrono::steady_clock::now();
//         accT[0] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * compute one of the "trivial" isometries
//          *
//          */
//         std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_L, I_XV), 
//                     std::make_pair( prime(I_U,col+1),prime(I_D,col+1) ),
//                     C_LU, C_LD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('L');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('L');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('L');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[1] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * Obtain new C_LU,C_LD by contraction of C_LU,C_LD 
//          * with an isometry U,V and reduction matrix R
//          *
//          *   I_U(col+1)
//          *   _|__          _                      I_U(col+1)
//          *  |    |--I_L---| \                      |
//          *  |C_LU|        |tU--I_SVD--|R|--I_L = |C_LU|--I_L  
//          *  |____|--I_XV--|_/  
//          *
//          *                                                 I_D(col+1)
//          *          I_D(col+1)                 _           _|__ 
//          *           |                        / |---I_Ln--|    |
//          *   I_Ln--|C_LD|= I_Ln--|R|--I_SVD1--tV|         |C_LD|
//          *                                    \_|--I_XV1--|____|
//          * 
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(5) ----- Construct reduced C_LU,C_LD -----"<< std::endl;
//         /*auto I_SVD_U = findtype(tU_tV.first.inds(),SVD_LINK);
//         auto I_SVD_V = findtype(tU_tV.second.inds(),SVD_LINK);

//         C_LU = ( C_LU*tU_tV.first )*delta(I_SVD_U,I_L);
//         C_LD = ( C_LD*tU_tV.second )*delta(I_SVD_V,prime(I_L,sizeN));

//         Print(C_LU);
//         Print(C_LD);*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[2] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /* 
//          * Contract T_L[0..sizeN-1] with X[row,col] to obtain new T_L
//          *
//          *            I_XH1                          I_XH1  
//          *            _|_                            _|_
//          *     I_XV--| X |--I_XV1             I_XV--|   |--I_XV1
//          *             |                            |T_L|  
//          *            I_XH             => I_L(row)--|___|--I_L(row+1)
//          *            _|_              =>
//          * I_L(row)--|T_L|--I_L(row+1)
//          *
//          * Obtain new T_L by contraction of T_L with an isometry tU, tV and
//          * reduction matrix R as follows (since C_LU was reduced by tU--R--
//          * from the bottom)
//          *
//          *                                         I_XH1 
//          *                          _              _|_ 
//          *                         / |--I_XV------|   |--I_XV1
//          *  I_L(row)--|R|--I_SVD1--tV|            |T_L|
//          *                         \_|--I_L(row)--|___|--I_L(row+1)
//          * and from the bottom
//          *
//          *            I_XH1
//          *            _|_                _
//          * I_L(row)--|   |--I_XV1-------| \
//          *           |T_L|              |tU--I_SVD--|R|--I_L(row+1)  
//          *           |___|--I_L(row+1)--|_/
//          *
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(6) ----- REDUCE T_L & X "<< std::string(47,'-') <<
//             std::endl;
//         /*tU_tV.second.noprime();
//         tU_tV.first.prime(LLINK,VSLINK);
//         for (int row=0; row<sizeN; row++) {
//             std::cout <<" --- Construct reduced T_L["<< row <<"] ---"
//                 << std::endl;
        
//             T_L[row] = ( ( ( T_L[row]*tU_tV.second.mapprime(LLINK,row-1,row) ) 
//                 *delta(I_SVD_V,prime(I_L,row)) )
//                 *tU_tV.first.mapprime(LLINK,row,row+1) ) 
//                 *delta(I_SVD_U,prime(I_L,row+1));

//             T_L[row].noprime(HSLINK);
//             std::cout << TAG_T_L <<"["<< row <<"]";
//             printfln("= %s",T_L[row]);
//         }*/
        
//         t_iso_end = std::chrono::steady_clock::now();
//         accT[3] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         std::cout <<"Column "<< col <<" done"<< std::endl;
//     }

//     // End of cluster absorption
//     C_LU.noprime(ULINK);
//     C_LD.noprime(DLINK);

//     std::cout <<"##### InsLCol Done "<< std::string(53,'#') << std::endl;
// }

// void CtmEnv::insLCol(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type) {
//     for (int col=0; col<sizeM; col++) {
//         C_LU *= T_U[col];
//         C_LD *= T_D[col];

//         for (int row=0; row<sizeN; row++) {
//             T_L[row] *= sites[cToS[std::make_pair(row,col)]];
//             T_L[row].prime(VSLINK,row);
//             T_L[row].noprime(HSLINK);
//         }

//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_L, I_XV), 
//                     std::make_pair( prime(I_U,col+1),prime(I_D,col+1) ),
//                     C_LU, C_LD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('L');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('L');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('L');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('L');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('L');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }
//     }

//     // End of cluster absorption
//     C_LU.noprime(ULINK);
//     C_LD.noprime(DLINK);
// }

// void CtmEnv::insRCol_DBG(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
// {
//     std::cout <<"##### InsRCol called "<< std::string(51,'#') << std::endl;
//     // sequentialy contract left boundary of environment with 
//     // sizeM rows of cluster + half-row matrices T_U* and T_D*
//     for (int col=sizeM-1; col>=0; col--) {
//         std::chrono::steady_clock::time_point t_iso_begin = 
//             std::chrono::steady_clock::now();
//         /*
//          * Insert a copy of column into network to obtain following expanded TN:
//          *
//          *             AAAAAAAAAAAAAAAAAAAAAAAAAAAA
//          *             ________                ____      
//          *  I_U(col)--|T_U(col)|--I_U(col+1)--|C_RU|
//          *              |                       | 
//          *             I_XV                    I_R
//          *                                     _|__
//          *                             I_XHm--|T_R0|
//          *                                      |
//          *                                     I_R1
//          *                                      |
//          *                                     ...
//          *                                      |
//          *                                     I_Rn-1
//          *                                     _|____
//          *                             I_XHm--|T_Rn-1|
//          *                                      |
//          *             I_XVn                   I_Rn
//          *             _|______                _|__
//          *  I_D(col)--|T_D(col)|--I_D(col+1)--|C_RD|
//          *
//          *              AAAAAAAAAAAAAAAAAAAAAAAAAAAA   
//          *
//          * Focus on a part of the TN highlighted by A's. Absorb the 
//          * inserted tensors to obtain TN of the original form
//          *
//          * 1) Contract C_RU with T_U(col)
//          *             ________                ____                 ______
//          *  I_U(col)--|T_U(col)|--I_U(col+1)--|C_RU| =>  I_U(col)--| C_RU |
//          *              |                       |    =>             |    |
//          *             I_XV                    I_R                 I_XV I_R
//          *
//          * 2) Contract C_LD with T_D(col)
//          *
//          *             I_XV1                   I_Rn                I_XV1 I_Rn
//          *             _|______                _|__                 |_____|_
//          *  I_D(col)--|T_D(col)|--I_D(col+1)--|C_RU| =>  I_U(col)--|_C_RU___|
//          *
//          */
//         std::cout <<"(1) ----- C_RU & T_U"<< col <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_RU);
//         Print(T_U[col]);

//         C_RU *= T_U[col];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_RU);

//         std::cout <<"(2) ----- C_RD & T_D"<< col <<"-----"<< std::endl;
//         std::cout <<"Before contraction"<< std::endl;
//         Print(C_RD);
//         Print(T_D[col]);

//         C_RD *= T_D[col];

//         std::cout <<"After contraction"<< std::endl;
//         Print(C_RD);

//         std::cout <<"(3) ----- T_R & X "<< std::string(54,'-') << std::endl;
//         for (int row=0; row<sizeN; row++) {
//             std::cout <<"--- Before contraction T_R["<< row <<"] & X["<< row
//                 <<","<< col <<"] ---"<< std::endl;
//             std::cout << TAG_T_R <<"["<< row <<"]";
//             printfln("= %s",T_R[row]);
//             std::cout <<"("<< row <<","<< col <<") -> sites["<< 
//                 cToS[std::make_pair(row,col)] <<"]"<< std::endl;

//             T_R[row].mapprime(sizeM,1,HSLINK);
//             T_R[row] *= sites[cToS[std::make_pair(row,col)]];
//             T_R[row].prime(VSLINK,row);
//             T_R[row].prime(HSLINK,sizeM);

//             std::cout <<"After contraction Row="<< row << std::endl;
//             std::cout << TAG_T_R <<"["<< row <<"]";
//             printfln("= %s",T_R[row]);
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('R');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('R');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         std::chrono::steady_clock::time_point t_iso_end = 
//             std::chrono::steady_clock::now();
//         accT[0] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * compute one of the "trivial" isometries
//          *
//          */
//         std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_R, I_XV), 
//                     std::make_pair( prime(I_U,col),prime(I_D,col) ),
//                     C_RU, C_RD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('R');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('R');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('R');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[1] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /*
//          * Obtain new C_LU,C_LD by contraction of C_LU,C_LD 
//          * with an isometry U,V and reduction matrix R
//          *
//          *   I_U(col)
//          *   _|__          _                      I_U(col)
//          *  |    |--I_R---| \                      |
//          *  |C_RU|        |tU--I_SVD--|R|--I_R = |C_RU|--I_R  
//          *  |____|--I_XV--|_/  
//          *
//          *                                                 I_D(col)
//          *          I_D(col)                   _           _|__ 
//          *           |                        / |---I_Rn--|    |
//          *   I_Rn--|C_RD|= I_Rn--|R|--I_SVD1--tV|         |C_RD|
//          *                                    \_|--I_XV1--|____|
//          * 
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(5) ----- Construct reduced C_RU,C_RD -----"<< std::endl;
//         /*auto I_SVD_U = findtype(tU_tV.first.inds(),SVD_LINK);
//         auto I_SVD_V = findtype(tU_tV.second.inds(),SVD_LINK);

//         C_RU = (C_RU*tU_tV.first )*delta(I_SVD_U,I_R);
//         C_RD = (C_RD*tU_tV.second)*delta(I_SVD_V,prime(I_R,sizeN));

//         Print(C_RU);
//         Print(C_RD);*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[2] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         /* 
//          * Contract T_R[0..sizeN-1] with X[row,col] to obtain new T_R
//          *
//          *              I_XH                            I_XH
//          *              _|_                             _|_
//          *      I_XV1--| X |--I_XV              I_XV1--|   |--I_XV
//          *               |                             |T_R|  
//          *              I_XH1           => I_R(row+1)--|___|--I_R(row)
//          *              _|_             =>
//          * I_R(row+1)--|T_R|--I_R(row)
//          *
//          * Obtain new T_R by contraction of T_R with an isometry tU, tV and
//          * reduction matrix R as follows (since C_RU was reduced by tU--R--
//          * from the bottom)
//          *
//          *                                         I_XH 
//          *                          _              _|_ 
//          *                         / |--I_XV------|   |--I_XV1
//          *  I_R(row)--|R|--I_SVD1--tV|            |T_R|
//          *                         \_|--I_R(row)--|___|--I_R(row+1)
//          * and from the bottom
//          *
//          *            I_XH
//          *            _|_                _
//          * I_R(row)--|   |--I_XV1-------| \
//          *           |T_R|              |tU--I_SVD--|R|--I_R(row+1)  
//          *           |___|--I_R(row+1)--|_/
//          *
//          * and using appropriate delta-matrices to relabel auxiliary reduction 
//          * indices to the original ones of the environment
//          *
//          */
//         std::cout <<"(6) ----- REDUCE T_R & X "<< std::string(47,'-') << 
//             std::endl;
//         /*tU_tV.second.noprime();
//         tU_tV.first.prime(RLINK,VSLINK);
//         for (int row=0; row<sizeN; row++) {
//             std::cout <<" --- Construct reduced T_R["<< row <<"] ---"
//                 << std::endl;
            
//             // T_R[row] = ( T_R[row]*tU_tV.second.mapprime(RLINK,row-1,row) ) 
//             //     *delta(I_SVD_V,prime(I_R,row));

//             // T_R[row] = ( T_R[row]*tU_tV.first.mapprime(RLINK,row,row+1) ) 
//             //     *delta(I_SVD_U,prime(I_R,row+1));

//             T_R[row] = ( ( ( T_R[row]*tU_tV.second.mapprime(RLINK,row-1,row) ) 
//                 *delta(I_SVD_V,prime(I_R,row)) )
//                 *tU_tV.first.mapprime(RLINK,row,row+1) ) 
//                 *delta(I_SVD_U,prime(I_R,row+1));

//             T_R[row].prime(HSLINK);
//             std::cout << TAG_T_R <<"["<< row <<"]";
//             printfln("= %s",T_R[row]);
//         }*/

//         t_iso_end = std::chrono::steady_clock::now();
//         accT[3] += std::chrono::duration_cast
//             <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

//         std::cout <<"Column "<< col <<" done"<< std::endl;
//     }

//     // End of cluster absorption
//     C_RU.prime(ULINK,sizeM);
//     C_RD.prime(DLINK,sizeM);

//     std::cout <<"##### InsRCol Done "<< std::string(53,'#') << std::endl;
// }

// void CtmEnv::insRCol(CtmEnv::ISOMETRY iso_type,
//     CtmEnv::NORMALIZATION norm_type) {
//     for (int col=sizeM-1; col>=0; col--) {
//         C_RU *= T_U[col];
//         C_RD *= T_D[col];

//         for (int row=0; row<sizeN; row++) {
//             T_R[row].mapprime(sizeM,1,HSLINK);
//             T_R[row] *= sites[cToS[std::make_pair(row,col)]];
//             T_R[row].prime(VSLINK,row);
//             T_R[row].prime(HSLINK,sizeM);
//         }

//         // .first -> tU, .second -> tV from SVD/Diag = tU*S*tV
//         std::pair< ITensor, ITensor > tU_tV;
        
//         switch(iso_type) {
//             case ISOMETRY_T1: {
//                 tU_tV = isoT1( IndexSet(I_R, I_XV), 
//                     std::make_pair( prime(I_U,col),prime(I_D,col) ),
//                     C_RU, C_RD);
//                 break;
//             }
//             case ISOMETRY_T2: {
//                 isoT2('R');
//                 break;
//             }
//             case ISOMETRY_T3: {
//                 isoT3('R');
//                 break;
//             }
//             case ISOMETRY_T4: {
//                 isoT4('R');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Isometry type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }

//         switch(norm_type) {
//             case NORM_BLE: {
//                 normalizeBLE_ctmStep('R');
//                 break;
//             }
//             case NORM_PTN: {
//                 normalizePTN_ctmStep('R');
//                 break;
//             }
//             default: {
//                 std::cout <<"Unsupported Normalization type"<< std::endl;
//                 exit(EXIT_FAILURE);
//                 break;
//             }
//         }
//     }

//     // End of cluster absorption
//     C_RU.prime(ULINK,sizeM);
//     C_RD.prime(DLINK,sizeM);
// }

// ########################################################################
// isometries

void CtmEnv::isoT1(IndexSet const& iS_tU, 
    std::pair< Index, Index > const& iS_delta, ITensor const& t1, 
    ITensor const& t2) {
    
}

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
double CtmEnv::getNorm() const {
    // Upper edge - position [0,0]
    auto Norm = C_LU.at( cToS.at(std::make_pair(0,0)) );

    /*
     * Suppose sizeM >= sizeN, contract say left boundary of environment 
     *
     *   |C_LU_00|----I_U0
     *     |
     *   |T_L_00|----I_XH(n-1)*2
     *     |
     *    I_L1>>I_L0 (via mapprime)
     *    ...
     *     |
     * |T_L_0n-1|----I_XH0
     *     |
     *   |C_LD_0n-1|----I_D0 
     *
     */
    for ( int row=0; row<=sizeN-1; row++ ) {
        Norm.prime(HSLINK,2);
        Norm.mapprime(1,0,LLINK);
        Norm *= T_L.at( cToS.at(std::make_pair(0,row)) );
    }
    Norm *= C_LD.at( cToS.at(std::make_pair(0,sizeN-1)) );

    for ( int col=0; col<sizeM; col++ ) {
        Norm *= T_D.at( cToS.at(std::make_pair(col,0)) );
        Norm.mapprime(1,0,DLINK);

        /*
         * Contract the cluster+environment column by column
         *
         *   |C_LU_00|----I_U0
         *     |
         *   |T_L_00|----I_XH(n-1)*2
         *     |
         *    ...              I_XV0>>I_XV1
         *     |                | 
         * |T_L0n-1|----I_XH0--|X_0n-1  |--I_XH1
         *     |                |
         *     |               I_XV1>>I_XV1 (via mapprime)
         *     |                |
         * |C_LD_0n-1|--I_D0---|T_D_0n-1|--I_D1>>I_D0 
         *
         */
        for ( int row=sizeN-1; row>=0; row-- ) {
            Norm.mapprime(0,1,VSLINK);
            Norm *= prime(sites.at( cToS.at(std::make_pair(row,col)) ), HSLINK,
                2*(sizeN-1-row));
        }
        Norm.prime(HSLINK,-1);
        Norm *= T_U.at( cToS.at(std::make_pair(col,sizeN-1)) );
        Norm.mapprime(1,0,ULINK);
    }

    /*
     * Contract with right edge from bottom to top
     *
     *  |C_LU_00|--...--|T_U_m-10|--I_U1>>I_U0
     *     |              |         
     *   |T_L_00|--...--|X_m-10  |--I_XH(n-1)*2>>I_XH(n-1)*2+m
     *     |              |
     *    ...            ...                         I_R0>>I_R1
     *     |              |                           |
     * |T_L_0n-1|--...--|X_m-1n-1|----I_XH0>>I_XHm--|T_R_m-1n-1|
     *     |               |                          |
     * |C_LD_0n-1|--...--|T_Dm-1n-1|--I_D1>>I_D0---|C_RD_m-1n-1| 
     *
     */
    Norm *= C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) );
    for ( int row=sizeN-1; row>=0; row-- ) {
        Norm.mapprime(2*(sizeN-1-row),1,HSLINK);
        Norm.mapprime(0,1,RLINK);
        Norm *= T_R.at( cToS.at(std::make_pair(sizeM-1,row)) );
    }
    Norm *= C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) );

    return sumels(Norm);
}

/*
 * Normalize the tensors By their Largest Element (BLE)
 *
 */
void CtmEnv::normalizeBLE() {

    auto normalizeBLE_T = [](ITensor& t)
    {
        double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        t /= m;
    };

    for ( auto& t : C_LU ) { normalizeBLE_T(t); }
    for ( auto& t : C_RU ) { normalizeBLE_T(t); }
    for ( auto& t : C_RD ) { normalizeBLE_T(t); }
    for ( auto& t : C_LD ) { normalizeBLE_T(t); }
    for ( auto& t : T_U ) { normalizeBLE_T(t); }
    for ( auto& t : T_R ) { normalizeBLE_T(t); }
    for ( auto& t : T_D ) { normalizeBLE_T(t); }
    for ( auto& t : T_L ) { normalizeBLE_T(t); }
}

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
void CtmEnv::normalizeBLE_ctmStep(char ctmMove, int m, int n) {

    auto normalizeBLE_T = [](ITensor& t)
    {
        double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        t /= m;
    };

    switch(ctmMove) {
        case 'U': {
            normalizeBLE_T( C_LU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( C_RU.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            
            normalizeBLE_T( T_U.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( T_U.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            break;
        }
        case 'R': {
            normalizeBLE_T( C_RU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( C_RD.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            
            normalizeBLE_T( T_R.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( T_R.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            break;
        }
        case 'D': {
            normalizeBLE_T( C_LD.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( C_RD.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            
            normalizeBLE_T( T_D.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( T_D.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
        }
        case 'L': {
            normalizeBLE_T( C_LU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( C_LD.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            
            normalizeBLE_T( T_L.at( cToS.at(std::make_pair(m,n)) ) );
            normalizeBLE_T( T_L.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
        }
        default: {
            std::cout <<"Unsupported ctmMove type - expecting one of "
                <<" U,R,D or L"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }
}

/*
 * Normalize the tensors Proportional To (Frobenius) Norm (PTN)
 *
 */
void CtmEnv::normalizePTN() {

    for ( auto& t : C_LU ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : C_RU ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : C_RD ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : C_LD ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : T_U ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : T_R ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : T_D ) { t *= 1.0/sqrt(norm(t)); }
    for ( auto& t : T_L ) { t *= 1.0/sqrt(norm(t)); }

}

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
void CtmEnv::normalizePTN_ctmStep(char ctmMove, int m, int n) {


    auto normalizePTN_T = [](ITensor& t) { t /= sqrt(norm(t)); };

    switch(ctmMove) {
        case 'U': {
            normalizePTN_T( C_LU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( C_RU.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            
            normalizePTN_T( T_U.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( T_U.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            break;
        }
        case 'R': {
            normalizePTN_T( C_RU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( C_RD.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            
            normalizePTN_T( T_R.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( T_R.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            break;
        }
        case 'D': {
            normalizePTN_T( C_LD.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( C_RD.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
            
            normalizePTN_T( T_D.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( T_D.at( cToS.at(
                std::make_pair((m+1) % sizeM,n)) ) );
        }
        case 'L': {
            normalizePTN_T( C_LU.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( C_LD.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
            
            normalizePTN_T( T_L.at( cToS.at(std::make_pair(m,n)) ) );
            normalizePTN_T( T_L.at( 
                cToS.at(std::make_pair(m,(n+1) % sizeN)) ) );
        }
        default: {
            std::cout <<"Unsupported ctmMove type - expecting one of "
                <<" U,R,D or L"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }

}

// ########################################################################
// methods handling SVD spectra of corner matrices

void CtmEnv::computeSVDspec() {

    ITensor S, V;
    // indices of S matrix holding singular values
    IndexSet iS;

    /*
     * Decomposing 
     *       ____ 
     * I_L--|C_LU|--I_U = I_L--|U|--|S|--|V|--I_U
     *
     * where S is diagonal and holds the singular values 
     * of C_LU
     *
     */
    auto U = ITensor(I_L);
    svd( C_LU.at( cToS.at(std::make_pair(0,0)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_clu[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_Um--|C_RU|--I_R
     *
     */
    U = ITensor(prime(I_U,sizeM));
    svd( C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_cru[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_Rn--|C_RD|--I_Dm
     *
     */
    U = ITensor(prime(I_R,sizeN));
    svd( C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_crd[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_D--|C_LD|--I_Ln
     *
     */
    U = ITensor(I_D);
    svd( C_LD.at( cToS.at(std::make_pair(0,sizeN-1)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_cld[i-1] = S.real(iS[0](i), iS[1](i));
    }
}

/*
 * TODO? possibly optimize the implementation
 *
 */
void CtmEnv::printSVDspec() const {

    std::cout << "spec_clu: ( ";
    for( const auto& val : spec.spec_clu ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_cru: ( ";
    for( const auto& val : spec.spec_cru ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_crd: ( ";
    for( const auto& val : spec.spec_crd ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_cld: ( ";
    for( const auto& val : spec.spec_cld ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;
}

void CtmEnv::specDist(CtmSpec const& s1, CtmSpec const& s2) const {
    
    double d_sClu = 0.0; 
    double d_sCru = 0.0;
    double d_sCrd = 0.0;
    double d_sCld = 0.0; 

    for(int i=0; i<x; i++) {
        d_sClu += std::pow(std::abs(s1.spec_clu[i]
            -s2.spec_clu[i]), 2.0);
        d_sCru += std::pow(std::abs(s1.spec_cru[i]
            -s2.spec_cru[i]), 2.0);
        d_sCrd += std::pow(std::abs(s1.spec_crd[i]
            -s2.spec_crd[i]), 2.0);
        d_sCld += std::pow(std::abs(s1.spec_cld[i]
            -s2.spec_cld[i]), 2.0);
    }

    std::cout <<"Dist(C_lu): "<< std::sqrt(d_sClu) << std::endl;
    std::cout <<"Dist(C_ru): "<< std::sqrt(d_sCru) << std::endl;
    std::cout <<"Dist(C_rd): "<< std::sqrt(d_sCrd) << std::endl;
    std::cout <<"Dist(C_ld): "<< std::sqrt(d_sCld) << std::endl;
}

CtmEnv::CtmSpec CtmEnv::getCtmSpec() const {
    return spec;
}

/* Return environment of original cluster, hence just
 * a selected C_*s and T_*s
 *
 */
CtmData CtmEnv::getCtmData() const {
    std::vector< itensor::ITensor > tT_U;
    std::vector< itensor::ITensor > tT_R;
    std::vector< itensor::ITensor > tT_D;
    std::vector< itensor::ITensor > tT_L;

    for( int col=0; col<sizeM; col++) {
        tT_U.push_back( T_U.at( cToS.at(std::make_pair(col,0)) ) );
        tT_D.push_back( T_D.at( cToS.at(std::make_pair(col,sizeN-1)) ) );
    }
    for( int row=0; row<sizeN; row++) {
        tT_L.push_back( T_L.at( cToS.at(std::make_pair(0,row)) ) );
        tT_R.push_back( T_R.at( cToS.at(std::make_pair(sizeM-1,row)) ) );
    }

    CtmData ctmData = {
        x, d, sizeN, sizeM,
        sites, cToS, 
        tT_U, tT_R, tT_D, tT_L,
        C_LU.at( cToS.at(std::make_pair(0,0)) ),
        C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) ),
        C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) ),
        C_LD.at( cToS.at(std::make_pair(0,sizeN-1)) ),
        I_U, I_R, I_D, I_L,
        I_XH, I_XV };
    return ctmData;
}

// ############################################################################
// IO toString methods

std::ostream& CtmEnv::print(std::ostream& s) const {
    s <<"CtmEnv( "<< std::endl 
        << m_name <<", d="<< d <<", x="<< x << std::endl 
        <<"cluster(NxM): "<< sizeN <<"(row) x "<< sizeM <<"(column)"
        << std::endl;
    
    s <<"clusterToSite: ["<< std::endl;
    for( const auto& cToSEntry : cToS ) {
        s << WS4 <<"("<< cToSEntry.first.first <<", "
            << cToSEntry.first.second <<") -> "<< cToSEntry.second 
            << std::endl;
    }
    s <<"]"<< std::endl;
    
    s <<"sites: ["<< std::endl;
    for( std::size_t i=0; i<sites.size(); i++) {
        s << WS4 << i;
        printfln(" = %s", sites[i]);
    }
    s <<"]"<< std::endl;

    // Loop over inequivalent sites and print their environment
    for (std::size_t i=0; i<sites.size(); i++) {
        s <<"----"<< siteIds[i] <<" start CORNER TENSORS----"<< std::endl;
        s << TAG_C_LU;
        printfln(" = %s", C_LU[i]);
        s << TAG_C_RU;
        printfln(" = %s", C_RU[i]);
        s << TAG_C_RD;
        printfln(" = %s", C_RD[i]);
        s << TAG_C_LD;
        printfln(" = %s", C_LD[i]);
        s <<"------end CORNER TENSORS----"<< std::endl;

        s <<"----"<< siteIds[i] <<"start HALF-ROW TENSORS--"<< std::endl;
        s << TAG_T_L << i;
        printfln(" = %s", T_L[i]);
        s << TAG_T_R << i;
        printfln(" = %s", T_R[i]);
        s <<"------end HALF-ROW TENSORS--"<< std::endl;

        s <<"----"<< siteIds[i] <<"start HALF-COL TENSORS--"<< std::endl;
        s << TAG_T_U << i;
        printfln(" = %s", T_U[i]);
        s << TAG_T_D << i;
        printfln(" = %s", T_D[i]);
        s <<"------end HALF-COL TENSORS--"<< std::endl;
    }

    return s;
}

std::ostream& operator<<(std::ostream& s, CtmEnv const& c) {
    return c.print(s);
}

std::ostream& operator<<(std::ostream& s, CtmEnv::CtmSpec const& spec) {
    s << "C_lu [0,0] ";
    for (std::complex<double> const& sv : spec.spec_clu) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ru: "; //["<< sizeM-1 <<",0] "<<;
    for (std::complex<double> const& sv : spec.spec_cru) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_rd: "; //["<< sizeM-1 <<","<< sizeN-1 <<"] ";
    for (std::complex<double> const& sv : spec.spec_crd) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ld: "; //[0,"<< sizeN-1 <<"] ";
    for (std::complex<double> const& sv : spec.spec_cld) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    return s;
}

/*
 * TODO Perhaps(?) move to ctm-cluster-io
 *
 */
CtmEnv::INIT_ENV toINIT_ENV(std::string const& iE) {
    if( iE=="INIT_ENV_const1") return CtmEnv::INIT_ENV_const1;
    if( iE=="INIT_ENV_rnd"   ) return CtmEnv::INIT_ENV_rnd;
    if( iE=="INIT_ENV_file"  ) return CtmEnv::INIT_ENV_file;
    std::cout << "Unsupported INIT_ENV" << std::endl;
    exit(EXIT_FAILURE);
}

CtmEnv::ISOMETRY toISOMETRY(std::string const& isoType) {
    if(isoType == "ISOMETRY_T1") return CtmEnv::ISOMETRY_T1;
    std::cout << "Unsupported ISOMETRY" << std::endl;
    exit(EXIT_FAILURE);
}

CtmEnv::NORMALIZATION toNORMALIZATION(std::string const& normType) {
    if(normType == "NORM_BLE") return CtmEnv::NORM_BLE;
    if(normType == "NORM_PTN") return CtmEnv::NORM_PTN;
    std::cout << "Unsupported NORMALIZATION" << std::endl;
    exit(EXIT_FAILURE);
}