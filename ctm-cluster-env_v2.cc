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

    computeSVDspec();

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

void CtmEnv::insURow_DBG(CtmEnv::ISOMETRY iso_type, 
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
{
    std::cout <<"##### InsURow called "<< std::string(51,'#') << std::endl;

    // sequentialy contract upper boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*
    for (int row=0; row<sizeN; row++) {
        std::chrono::steady_clock::time_point t_iso_begin = 
            std::chrono::steady_clock::now();
        /*
         * Absorb one row of network up, leading to new (unreduced) environment
         * tensors for sites one row down (+1 in Y coord)
         *   _______                   ______                    _________
         *  |C_LU_00|--I_U  ...  I_U--|T_U_x0|--I_U1 ...  I_U1--|C_RU_m-10|
         *   _|____                     |                         _|______
         *  |T_L_00|--I_XH  ... I_XH--|X_x0|--I_XH1  ...  I_XH1--|T_R_m-10|   
         *    |                         |                          | 
         *   I_L1>>I_L                 I_XV1>>I_XV               I_R1>>I_R
         *
         *    V V V                  V V V V V                    V V V
         *   _______                  ______                    _________
         *  |       |--I_U      I_U--|      |--I_U1      I_U1--|         |
         *  |C_LU_01|                |T_U_x1|                  |C_RU_m-11|
         *  |_______|--I_XH    I_XH--|______|--I_XH1    I_XH1--|_________|
         *     |                        |                          |
         *    I_L                      I_XV                       I_R
         *
         */
        std::cout <<"(1) ----- C_LU & T_L ["<< 0 <<","<< row <<"] -----"
            << std::endl;
        auto tC1 = C_LU.at( cToS.at(std::make_pair(0,row)) ) * 
            T_L.at( cToS.at(std::make_pair(0,row)) );
        tC1.prime(LLINK, -1);
        Print(tC1);
        
        std::cout <<"(2) ----- C_RU & T_R ["<< sizeM-1 <<","<< row <<"] -----"
            << std::endl;
        auto tC2 = C_RU.at( cToS.at(std::make_pair(sizeM-1,row)) ) * 
            T_R.at( cToS.at(std::make_pair(sizeM-1,row)) );
        tC2.prime(RLINK, -1);
        Print(tC2);

        std::vector<ITensor> tT1; 
        for (int col=0; col<sizeM; col++) {
            std::cout <<"(3."<< col <<") ----- T_U & X ["<< col <<","<< row <<
                "] -----"<< std::endl;

            tT1.push_back( T_U.at( cToS.at(std::make_pair(col,row)) ) * 
                sites[cToS[std::make_pair(col,row)]] );
            tT1.at(tT1.size()-1).prime(VSLINK, -1);

            printfln("= %s", tT1[col]);
        }

        std::chrono::steady_clock::time_point t_iso_end =
            std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_U0--|             \
         *          |Z[0..sizeM-1] |--I_UsizeM+10
         *   I_XH0--|_____________/
         *
         */

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('U', -1, row);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[1] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(5) ----- Construct reduced C_LU,C_RU -----"<< std::endl;
        /*   _______                            ________
         *  |       |--I_U               I_U0--|        \
         *  |C_LU_01|        *contract*        |ZsizeM-1 --I_UsizeM+10>>I_U0
         *  |_______|--I_XH             I_XH0--|________/
         *     |
         *    I_L
         *
         */
        C_LU.at( cToS.at(std::make_pair(0,(row+1)%sizeN)) ) = 
            tC1 * isoZ[sizeM-1];
        C_LU.at( cToS.at(std::make_pair(0,(row+1)%sizeN)) )
            .mapprime(ULINK,sizeM+10,0);
        std::cout << TAG_C_LU <<"["<< 0 <<","<< (row+1)%sizeN <<"]";
        printfln("= %s", C_LU.at(cToS.at(std::make_pair(0,(row+1)%sizeN))) );

        /*         _________                            ________
         *  I_U1--|         |              I_U1<<I_U0--|        \
         *        |C_RU_m-11|  *contract*              |ZsizeM-1 --I_UsizeM+10
         * I_XH1--|_________|            I_XH1<<I_XH0--|_dagger_/      >>I_U1
         *            |
         *           I_R
         *
         */
        C_RU.at( cToS.at(std::make_pair(sizeM-1,(row+1)%sizeN)) ) = 
            tC2 * isoZ[sizeM-1].conj().mapprime(ULINK,0,1, HSLINK,0,1);
        C_RU.at( cToS.at(std::make_pair(sizeM-1,(row+1)%sizeN)) )
            .mapprime(ULINK,sizeM+10,1);
        std::cout << TAG_C_RU <<"["<< sizeM-1 <<","<< (row+1)%sizeN <<"]";
        printfln("= %s", C_RU.at( 
            cToS.at(std::make_pair(sizeM-1,(row+1)%sizeN))) );

        /*        ______
         *  I_U--|      |--I_U1              I_U0--|Zc-1\__I_UsizeM+10>>I_U0 
         *       |T_U_c1|       *contract*  I_XH0--|dag /
         * I_XH--|__ ___|--I_XH1   
         *          |                    I_U1--|Zc  \__I_UsizeM+10>>I_U1
         *         I_XV                 I_XH1--|    /
         *
         */
        std::cout <<"(6) ----- REDUCE T_U & X "<< std::string(47,'-') <<
            std::endl;
        for (int col=0; col<sizeM; col++) {
            std::cout <<" --- Construct reduced T_U["<< col <<","
                << (row+1)%sizeN <<"] ---"<< std::endl;
        
            // reset primes
            isoZ[(col-1+sizeM)%sizeM].mapprime(ULINK,1,0, HSLINK,1,0);

            tT1[col] *= isoZ[(col-1+sizeM)%sizeM];
            tT1[col].mapprime(ULINK,sizeM+10,0);

            // conjugate back
            isoZ[(col-1+sizeM)%sizeM].conj();

            isoZ[col].mapprime(ULINK,0,1, HSLINK,0,1);

            T_U.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) ) = 
                tT1[col] * isoZ[col];
            T_U.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) )
                .mapprime(ULINK,sizeM+10,1);
            
            isoZ[col].conj();

            std::cout << TAG_T_U <<"["<< col <<","<< (row+1)%sizeN <<"]";
            printfln("= %s", T_U.at( 
                cToS.at(std::make_pair(col,(row+1)%sizeN))) );
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[2] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(6) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('U', -1, row);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('U', -1, row);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    std::cout <<"##### InsURow Done "<< std::string(53,'#') << std::endl;
}

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

void CtmEnv::insDRow_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
{
    std::cout <<"##### InsDRow called "<< std::string(51,'#') << std::endl;
    // sequentialy contract lower boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*
    for (int row=sizeN-1; row>=0; row--) {
        std::chrono::steady_clock::time_point t_iso_begin = 
            std::chrono::steady_clock::now();
        /*
         * Absorb one row of network up, leading to new (unreduced) environment
         * tensors for sites one row down (+1 in Y coord)
         *
         *   I_L0>>I_L1               I_XV0>>I_XV1             I_R0>>I_R1
         *    |                         |                          | 
         *  |T_L_0n-1|--I_XH ... I_XH--|X_xn-1|--I_XH1 ... I_XH1--|T_R_m-1n-1|
         *    |                         |                          | 
         *  |C_LD_0n-1|--I_D ... I_D--|T_D_xn-1|--I_D1 ... I_D1--|C_RD_m-1n-1| 
         *
         *    V V V                  V V V V V                    V V V
         *
         *     I_L1                    I_XV1                        I_R1
         *   ___|_____                ___|____                  _____|_____
         *  |         |--I_D    I_D--|        |--I_D1    I_D1--|           |
         *  |C_LD_0n-1|              |T_D_xn-1|                |C_RD_m-1n-1|
         *  |_________|--I_XH  I_XH--|________|--I_XH1  I_XH1--|___________|
         *
         */
         std::cout <<"(1) ----- C_LD & T_L ["<< 0 <<","<< row <<"] -----"
            << std::endl;
        auto tC4 = C_LD.at( cToS.at(std::make_pair(0,row)) ) * 
            T_L.at( cToS.at(std::make_pair(0,row)) );
        tC4.prime(LLINK);
        Print(tC4);

        std::cout <<"(2) ----- C_RD & T_R ["<< sizeM-1 <<","<< row <<"] -----"
            << std::endl;
        auto tC3 = C_RD.at( cToS.at(std::make_pair(sizeM-1,row)) ) * 
            T_R.at( cToS.at(std::make_pair(sizeM-1,row)) );
        tC3.prime(RLINK);
        Print(tC3);

        std::vector<ITensor> tT3; 
        for (int col=0; col<sizeM; col++) {
            std::cout <<"(3."<< col <<") ----- T_D & X ["<< col <<","<< row <<
                "] -----"<< std::endl;

            tT3.push_back( T_D.at( cToS.at(std::make_pair(col,row)) ) * 
                sites[cToS[std::make_pair(col,row)]] );
            tT3.at(tT3.size()-1).prime(VSLINK);

            printfln("= %s", tT3[col]);
        }

        std::chrono::steady_clock::time_point t_iso_end =
            std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

       std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_D0--|             \
         *          |Z[0..sizeM-1] |--I_DsizeM+10
         *   I_XH0--|_____________/
         *
         */

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('D', -1, row);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[1] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(5) ----- Construct reduced C_LD,C_RD -----"<< std::endl;
        /*
         *    I_L1
         *   __|______                            ________
         *  |         |--I_D               I_D0--|        \
         *  |C_LD_0n-2|        *contract*        |ZsizeM-1 --I_DsizeM+10>>I_D0
         *  |_________|--I_XH             I_XH0--|________/
         *
         */
        C_LD.at( cToS.at(std::make_pair(0,(row-1+sizeN)%sizeN)) ) = 
            tC4 * isoZ[sizeM-1];
        C_LD.at( cToS.at(std::make_pair(0,(row-1+sizeN)%sizeN)) )
            .mapprime(DLINK,sizeM+10,0);
        std::cout << TAG_C_LD <<"["<< 0 <<","<< (row-1+sizeN)%sizeN <<"]";
        printfln("= %s", 
            C_LD.at(cToS.at(std::make_pair(0,(row-1+sizeN)%sizeN))) );

        /*
         *            I_R1
         *         ____|______                           ________
         *  I_D1--|           |             I_D1<<I_D0--|        \
         *        |C_RD_m-1n-2| *contract*              |ZsizeM-1 --I_DsizeM+10
         * I_XH1--|___________|           I_XH1<<I_XH0--|_dagger_/     >>I_D1
         *
         */
        C_RD.at( cToS.at(std::make_pair(sizeM-1,(row-1+sizeN)%sizeN)) ) = 
            tC3 * isoZ[sizeM-1].conj().mapprime(DLINK,0,1, HSLINK,0,1);
        C_RD.at( cToS.at(std::make_pair(sizeM-1,(row-1+sizeN)%sizeN)) )
            .mapprime(DLINK,sizeM+10,1);
        std::cout << TAG_C_RD <<"["<< sizeM-1 <<","<< (row-1+sizeN)%sizeN <<"]";
        printfln("= %s", C_RD.at( 
            cToS.at(std::make_pair(sizeM-1,(row-1+sizeN)%sizeN))) );

        /*
         *          I_XV1
         *        ___|____
         *  I_D--|        |--I_D1              I_D0--|Zc-1\__I_DsizeM+10>>I_D0 
         *       |T_D_cn-1|       *contract*  I_XH0--|dag /
         * I_XH--|________|--I_XH1   
         *                               I_D1--|Zc  \__I_DsizeM+10>>I_D1
         *                              I_XH1--|    /
         */
        std::cout <<"(6) ----- REDUCE T_D & X "<< std::string(47,'-') <<
            std::endl;
        for (int col=0; col<sizeM; col++) {
            std::cout <<" --- Construct reduced T_D["<< col <<","
                << (row-1+sizeN)%sizeN <<"] ---"<< std::endl;
        
            // reset primes
            isoZ[(col-1+sizeM)%sizeM].mapprime(DLINK,1,0, HSLINK,1,0);

            tT3[col] *= isoZ[(col-1+sizeM)%sizeM];
            tT3[col].mapprime(DLINK,sizeM+10,0);

            // conjugate back
            isoZ[(col-1+sizeM)%sizeM].conj();

            isoZ[col].mapprime(DLINK,0,1, HSLINK,0,1);

            T_D.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) ) = 
                tT3[col] * isoZ[col];
            T_D.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) )
                .mapprime(DLINK,sizeM+10,1);
            
            isoZ[col].conj();

            std::cout << TAG_T_D <<"["<< col <<","<< (row-1+sizeN)%sizeN <<"]";
            printfln("= %s", T_D.at( 
                cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN))) );
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[2] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
        
        std::cout <<"(6) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('D', -1, row);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('D', -1, row);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    std::cout <<"##### InsDRow Done "<< std::string(53,'#') << std::endl;
}

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

void CtmEnv::insLCol_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT)
{
    std::cout <<"##### InsLCol called "<< std::string(51,'#') << std::endl;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*
    for (int col=0; col<sizeM; col++) {
        std::chrono::steady_clock::time_point t_iso_begin = 
            std::chrono::steady_clock::now();
        /*
         * Absorb one column of network to the left, leading to new (unreduced)
         * environment tensors for sites in column to the right (+1 in X coord)
         *   _______    ______      
         *  |C_LU_00|--|T_U_00|--I_U1 ==> |C_L_10|--I_U1>>I_U
         *    |           |                 |   |
         *   I_L         I_XV              I_L I_XV
         *
         *   I_L       I_XV               I_L  I_XV
         *   _|____     |                  |    |  
         *  |T_L_00|--|X_00|--I_XH1   ==> |T_L_10|--I_XH1>>I_XH   
         *    |         |                  |    |
         *   I_L1      I_XV1              I_L1 I_XV1
         *   ...
         *   I_L         I_XV               I_L  I_XV 
         *   _|______     |                  |    |
         *  |T_L_0n-1|--|X_0n-1|--I_XH1 ==> |T_L_1n-1|--I_XH1>>I_XH
         *    |           |                  |    |
         *   I_L1        I_XV1              I_L1 I_XV1 

         *   I_L1          I_XV1               I_L1   I_XV1 
         *   _|_______     |                   |     |
         *  |C_LD_0n-1|--|T_D_0n-1|--I_D1 ==> |C_LD_1n-1|--I_D1>>I_D
         *
         */
        std::cout <<"(1) ----- C_LU & T_U ["<< col <<","<< 0 <<"] -----"
            << std::endl;
        auto tC1 = C_LU.at( cToS.at(std::make_pair(col,0)) ) * 
            T_U.at( cToS.at(std::make_pair(col,0)) );
        tC1.prime(ULINK, -1);
        Print(tC1);

        std::cout <<"(2) ----- C_LD & T_D ["<< col <<","<< sizeN-1 <<"] -----"
            << std::endl;
        auto tC4 = C_LD.at( cToS.at(std::make_pair(col,sizeN-1)) ) * 
            T_D.at( cToS.at(std::make_pair(col,sizeN-1)) );
        tC4.prime(DLINK, -1);
        Print(tC4);

        std::vector<ITensor> tT4; 
        for (int row=0; row<sizeN; row++) {
            std::cout <<"(3."<< row <<") ----- T_L & X ["<< col <<","<< row <<
                "] -----"<< std::endl;

            tT4.push_back( T_L.at( cToS.at(std::make_pair(col,row)) ) * 
                sites[cToS[std::make_pair(col,row)]] );
            tT4.at(tT4.size()-1).prime(HSLINK, -1);

            printfln("= %s", tT4[row]);
        }

        std::chrono::steady_clock::time_point t_iso_end = 
            std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_L0--|             \
         *          |Z[0..sizeN-1] |--I_LsizeN+10
         *   I_XV0--|_____________/
         *
         */

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('L', col, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[1] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(5) ----- Construct reduced C_LU,C_LD -----"<< std::endl;
        /*                                    ________
         * |C_LD_10|--I_U              I_L0--|        \
         *  |    |         *contract*        |ZsizeN-1 --I_LsizeN+10>>I_L0
         * I_L0  I_XV0                I_XV0--|________/
         *
         */
        C_LU.at( cToS.at(std::make_pair((col+1)%sizeM,0)) ) = 
            tC1 * isoZ[sizeN-1];
        C_LU.at( cToS.at(std::make_pair((col+1)%sizeM,0)) )
            .mapprime(LLINK,sizeN+10,0);
        std::cout << TAG_C_LU <<"["<< (col+1)%sizeM <<","<< 0 <<"]";
        printfln("= %s", C_LU.at(cToS.at(std::make_pair((col+1)%sizeM,0))) );

        /*                                             ________
         * I_L1  I_XV1                    I_L1<<I_L0--|        \
         *  |    |           *contract*               |ZsizeN-1 --I_LsizeN+10
         * |C_LD_1n-1|--I_D             I_XV1<<I_XV0--|_dagger_/      >>I_L1
         *
         */
        C_LD.at( cToS.at(std::make_pair((col+1)%sizeM,sizeN-1)) ) = 
            tC4 * isoZ[sizeN-1].conj().mapprime(LLINK,0,1, VSLINK,0,1);
        C_LD.at( cToS.at(std::make_pair((col+1)%sizeM,sizeN-1)) )
            .mapprime(LLINK,sizeN+10,1);
        std::cout << TAG_C_LD <<"["<< (col+1)%sizeM <<","<< sizeN-1 <<"]";
        printfln("= %s", C_LD.at( 
            cToS.at(std::make_pair((col+1)%sizeM,sizeN-1))) );

        /* 
         * I_L0 I_XV0                  I_L0--|Zr-1\__I_LsizeN+10>>I_L0 
         *  |    |                    I_XV0--|dag /
         * |T_L_1r|--I_XH  *contract*  
         *  |    |                     I_L1--|Zr  \__I_LsizeN+10>>I_L1
         * I_L1 I_XV1                 I_XV1--|    /
         *
         */
        std::cout <<"(6) ----- REDUCE T_L & X "<< std::string(47,'-') <<
            std::endl;
        for (int row=0; row<sizeN; row++) {
            std::cout <<" --- Construct reduced T_L["<< (col+1)%sizeM <<","
                << row <<"] ---"<< std::endl;
        
            // reset primes
            isoZ[(row-1+sizeN)%sizeN].mapprime(LLINK,1,0, VSLINK,1,0);

            tT4[row] *= isoZ[(row-1+sizeN)%sizeN];
            tT4[row].mapprime(LLINK,sizeN+10,0);

            // conjugate back
            isoZ[(row-1+sizeN)%sizeN].conj();

            isoZ[row].mapprime(LLINK,0,1, VSLINK,0,1);

            T_L.at( cToS.at(std::make_pair((col+1)%sizeM,row)) ) = 
                tT4[row] * isoZ[row];
            T_L.at( cToS.at(std::make_pair((col+1)%sizeM,row)) )
                .mapprime(LLINK,sizeN+10,1);
            
            isoZ[row].conj();

            std::cout << TAG_T_L <<"["<< (col+1)%sizeM <<","<< row <<"]";
            printfln("= %s", T_L.at( 
                cToS.at(std::make_pair((col+1)%sizeM,row))) );
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[2] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(6) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('L', col, -1);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('L', col, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    std::cout <<"##### InsLCol Done "<< std::string(53,'#') << std::endl;
}

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

void CtmEnv::insRCol_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT) 
{
    std::cout <<"##### InsRCol called "<< std::string(51,'#') << std::endl;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*
    for (int col=sizeM-1; col>=0; col--) {
        std::chrono::steady_clock::time_point t_iso_begin = 
            std::chrono::steady_clock::now();
        /*
         * Absorb one column of network to the right, leading to new 
         * (unreduced) environment tensors for sites in column to the 
         * left (-1 in X coord)
         *
         *   I_U--|T_U_m-10|--|C_RU_m-10| ==> I_U1<<I_U--|C_RU_m-20|
         *          |           |                          |     |
         *         I_XV        I_R                        I_XV  I_R
         *
         *         I_XV    I_R                         I_XV  I_R
         *         _|__     |                           |    |  
         *  I_XH--|X_00|--|T_R_m-10| ==> I_XH1<<I_XH--|T_R_m-20|   
         *          |       |                           |    |
         *         I_XV1   I_R1                        I_XV1 I_R1
         *   ...
         *         I_XV    I_R                           I_XV  I_R
         *         _|__     |                             |    |  
         *  I_XH--|X_00|--|T_R_m-1n-1| ==> I_XH1<<I_XH--|T_R_m-2n-1|   
         *          |       |                             |    |
         *         I_XV1   I_R1                          I_XV1 I_R1

         *        I_XV1         I_R1                         I_XV1   I_R1 
         *        _|_______      |                            |     |
         *  I_D--|T_D_m-1n-1|--|C_RD_m-1n-1| ==> I_D1<<I_D--|C_RD_m-2n-1|
         *
         */
        std::cout <<"(1) ----- C_RU & T_U ["<< col <<","<< 0 <<"] -----"
            << std::endl;
        auto tC2 = C_RU.at( cToS.at(std::make_pair(col,0)) ) * 
            T_U.at( cToS.at(std::make_pair(col,0)) );
        tC2.prime(ULINK);
        Print(tC2);

        std::cout <<"(2) ----- C_RD & T_D ["<< col <<","<< sizeN-1 <<"] -----"
            << std::endl;
        auto tC3 = C_RD.at( cToS.at(std::make_pair(col,sizeN-1)) ) * 
            T_D.at( cToS.at(std::make_pair(col,sizeN-1)) );
        tC3.prime(DLINK);
        Print(tC3);

        std::vector<ITensor> tT2; 
        for (int row=0; row<sizeN; row++) {
            std::cout <<"(3."<< row <<") ----- T_R & X ["<< col <<","<< row <<
                "] -----"<< std::endl;

            tT2.push_back( T_R.at( cToS.at(std::make_pair(col,row)) ) * 
                sites[cToS[std::make_pair(col,row)]] );
            tT2.at(tT2.size()-1).prime(HSLINK);

            printfln("= %s", tT2[row]);
        }

        std::chrono::steady_clock::time_point t_iso_end = 
            std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_R0--|             \
         *          |Z[0..sizeN-1] |--I_RsizeN+10
         *   I_XV0--|_____________/
         *
         */

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('R', col, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[1] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(5) ----- Construct reduced C_RU,C_RD -----"<< std::endl;
        /*                                       ________
         * I_U1--|C_RU_m-20|              I_R0--|        \
         *         |     |    *contract*        |ZsizeN-1 --I_RsizeN+10>>I_R0
         *        I_XV0  I_R0            I_XV0--|________/
         *
         */
        C_RU.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,0)) ) = 
            tC2 * isoZ[sizeN-1];
        C_RU.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,0)) )
            .mapprime(RLINK,sizeN+10,0);
        std::cout << TAG_C_RU <<"["<< (col-1+sizeM)%sizeM <<","<< 0 <<"]";
        printfln("= %s",
            C_RU.at(cToS.at(std::make_pair((col-1+sizeM)%sizeM,0))) );

        /*                                               ________
         *        I_XV1  I_R1               I_R1<<I_R0--|        \
         *         |      |    *contract*               |ZsizeN-1 --I_RsizeN+10
         * I_D1--|C_RD_m-2n-1|            I_XV1<<I_XV0--|_dagger_/     >>I_R1
         *
         */
        C_RD.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,sizeN-1)) ) = 
            tC3 * isoZ[sizeN-1].conj().mapprime(RLINK,0,1, VSLINK,0,1);
        C_RD.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,sizeN-1)) )
            .mapprime(RLINK,sizeN+10,1);
        std::cout << TAG_C_RD <<"["<< (col-1+sizeM)%sizeM <<","<< sizeN-1 <<"]";
        printfln("= %s", C_RD.at( 
            cToS.at(std::make_pair((col-1+sizeM)%sizeM,sizeN-1))) );

        /* 
         *       I_XV0 I_R0                 I_R0--|Zr-1\__I_RsizeN+10>>I_R0 
         *         |    |                  I_XV0--|dag /
         * I_XH1--|T_R_m-2r|  *contract*  
         *         |    |                   I_R1--|Zr  \__I_RsizeN+10>>I_R1
         *       I_XV1 I_R1                I_XV1--|    /
         *
         */
        std::cout <<"(6) ----- REDUCE T_R & X "<< std::string(47,'-') <<
            std::endl;
        for (int row=0; row<sizeN; row++) {
            std::cout <<" --- Construct reduced T_R["<< (col-1+sizeM)%sizeM 
                <<","<< row <<"] ---"<< std::endl;
        
            // reset primes
            isoZ[(row-1+sizeN)%sizeN].mapprime(RLINK,1,0, VSLINK,1,0);

            tT2[row] *= isoZ[(row-1+sizeN)%sizeN];
            tT2[row].mapprime(RLINK,sizeN+10,0);

            // conjugate back
            isoZ[(row-1+sizeN)%sizeN].conj();

            isoZ[row].mapprime(RLINK,0,1, VSLINK,0,1);

            T_R.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) ) = 
                tT2[row] * isoZ[row];
            T_R.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) )
                .mapprime(RLINK,sizeN+10,1);

            T_R.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) );
            
            isoZ[row].conj();

            std::cout << TAG_T_R <<"["<< (col-1+sizeM)%sizeM <<","<< row <<"]";
            printfln("= %s", T_R.at( 
                cToS.at(std::make_pair((col-1+sizeM)%sizeM,row))) );
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[2] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"(6) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('R', col, -1);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('R', col, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    std::cout <<"##### InsRCol Done "<< std::string(53,'#') << std::endl;
}

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

std::vector<ITensor> CtmEnv::isoT1(char ctmMove, int col, int row) {
    std::cout <<"----- ISO_T1 called for "<< ctmMove <<" at ["<< col
        <<","<< row <<"] -----"<<std::endl;

    std::vector<ITensor> isoZ;
    ITensor tRDM, S, V;
    Index svdI;
    Spectrum spec;

    switch(ctmMove) {
        case 'L': {
            isoZ = std::vector<ITensor>(sizeN, ITensor(I_L, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                // build RDM
                tRDM = build_2x2_RDM('L', col, r);
                // Perform SVD
                spec = svd(tRDM, isoZ[r], S, V, {"Maxm",x});
                Print(spec);
                // relabel isometry index
                svdI = commonIndex(isoZ[r],S);
                isoZ[r] *= delta(svdI, prime(I_L, sizeN+10));
                Print(isoZ[r]);
            }
            break;
        }
        case 'U': {
            isoZ = std::vector<ITensor>(sizeM, ITensor(I_U, I_XH));
            for (int c=0; c<sizeM; c++) {
                tRDM = build_2x2_RDM('U', (c+1)%sizeM, row);
                spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x});
                svdI = commonIndex(isoZ[c],S);
                isoZ[c] *= delta(svdI, prime(I_U, sizeM+10));
            }
            break;
        }
        case 'R': {
            isoZ = std::vector<ITensor>(sizeN, ITensor(I_R, I_XV));
            for (int r=0; r<sizeN; r++) {
                tRDM = build_2x2_RDM('R', col, (r+1)%sizeN);
                spec = svd(tRDM, isoZ[r], S, V, {"Maxm",x});
                svdI = commonIndex(isoZ[r],S);
                isoZ[r] *= delta(svdI, prime(I_R, sizeN+10));
            }
            break;
        }
        case 'D': {
            isoZ = std::vector<ITensor>(sizeM, ITensor(I_D, I_XH));
            for (int c=0; c<sizeM; c++) {
                tRDM = build_2x2_RDM('D', c, row);
                spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x});
                svdI = commonIndex(isoZ[c],S);
                isoZ[c] *= delta(svdI, prime(I_D, sizeM+10));
            }
            break;
        }
    }

    return isoZ;   
}

ITensor CtmEnv::build_2x2_RDM(char ctmMove, int col, int row) const {

    ITensor rdm;

    switch (ctmMove) {
        case 'L': {
            // build left upper corner
            rdm = build_corner('1', col, row);
            rdm.mapprime(LLINK,1,10, VSLINK,1,10); // Indices along cut
            // build right upper corner
            rdm *= build_corner('2', (col+1)%sizeM, row);
            // build right lower corner
            rdm *= build_corner('3', (col+1)%sizeM, (row+1)%sizeN );
            // build left lower corner
            rdm *= build_corner('4', col, (row+1)%sizeN );
            rdm.mapprime(LLINK,10,0, VSLINK,10,0); // Indices along cut
            break;
        }
        case 'U': {
            // build right upper corner
            rdm = build_corner('2', col, row);
            rdm.mapprime(ULINK,0,10, HSLINK,0,10); // Indices along cut    
            // build right lower corner
            rdm *= build_corner('3', col, (row+1)%sizeN );
            // build left lower corner
            rdm *= build_corner('4', (col-1+sizeM)%sizeM, (row+1)%sizeN );
            // build left upper corner
            rdm *= build_corner('1', (col-1+sizeM)%sizeM, row );
            rdm.mapprime(ULINK,0,1, HSLINK,0,1); // Indices along cut
            rdm.mapprime(ULINK,10,0, HSLINK,10,0); // Indices along cut
            break;
        }
        case 'R': {
            // build right lower corner
            rdm = build_corner('3', col, row);
            rdm.mapprime(RLINK,0,10, VSLINK,0,10); // Indices along cut
            // build left lower corner
            rdm *= build_corner('4', (col-1+sizeM)%sizeM, row);
            // build left upper corner
            rdm *= build_corner('1', (col-1+sizeM)%sizeM,
                (row-1+sizeN)%sizeN );
            // build right upper corner
            rdm *= build_corner('2', col, (row-1+sizeN)%sizeN );
            rdm.mapprime(RLINK,0,1, VSLINK,0,1); // Indices along cut
            rdm.mapprime(RLINK,10,0, VSLINK,10,0); // Indices along cut
            break;
        }
        case 'D': {
            // build left lower corner
            rdm = build_corner('4', col, row);
            rdm.mapprime(DLINK,1,10, HSLINK,1,10); // Indices along cut    
            // build left upper corner
            rdm *= build_corner('1', col, (row-1+sizeN)%sizeN );
            // build right upper corner
            rdm *= build_corner('2', (col+1)%sizeM, (row-1+sizeN)%sizeN );
            // build right lower corner
            rdm *= build_corner('3', (col+1)%sizeM, row );
            rdm.mapprime(DLINK,10,0, HSLINK,10,0); // Indices along cut
            break;
        }
    }

    // symmetrize rdm to obtain hermitian matrix
    rdm = 0.5*( rdm + swapPrime(rdm, 0, 1).conj() );

    return rdm;
}

ITensor CtmEnv::build_corner(char corner, int col, int row) const {
    ITensor ct;
    switch(corner) {
        case '1': {
            // build left upper corner
            ct = T_L.at( cToS.at(std::make_pair(col,row)) );
            ct *= C_LU.at( cToS.at(std::make_pair(col,row)) );
            ct *= sites[cToS.at(std::make_pair(col,row))];
            ct *= T_U.at( cToS.at(std::make_pair(col,row)) );
            ct.mapprime(ULINK,1,0, HSLINK,1,0);
            break;
        }
        case '2': {
            // build right upper corner
            ct = T_U.at( cToS.at(std::make_pair(col,row)) );
            ct *= C_RU.at( cToS.at(std::make_pair(col,row)) );
            ct *= sites[cToS.at(std::make_pair(col,row))];
            ct *= T_R.at( cToS.at(std::make_pair(col,row)) );
            ct.mapprime(RLINK,1,0, VSLINK,1,0);
            break;
        }
        case '3': {
            // build right lower corner
            ct = T_R.at( cToS.at(std::make_pair(col,row)) );
            ct *= C_RD.at( cToS.at(std::make_pair(col,row)) );
            ct *= sites[cToS.at(std::make_pair(col,row))];
            ct *= T_D.at( cToS.at(std::make_pair(col,row)) );
            ct.mapprime(DLINK,0,1, HSLINK,0,1);
            break;
        }
        case '4': {
            // build left lower corner
            ct = T_D.at( cToS.at(std::make_pair(col,row)) );
            ct *= C_LD.at( cToS.at(std::make_pair(col,row)) ); 
            ct *= sites[cToS.at(std::make_pair(col,row))];
            ct *= T_L.at( cToS.at(std::make_pair(col,row)) );
            ct.mapprime(LLINK,0,1, VSLINK,0,1);
            break;
        }
    }
    return ct;
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
void CtmEnv::normalizeBLE_ctmStep(char ctmMove, int col, int row) {

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

    std::cout <<"----- normalizeBLE_ctmStep called for "<< ctmMove 
        <<" ["<< col <<","<< row <<"]-----"<< std::endl;

    switch(ctmMove) {
        case 'U': { // col is ignored as we iterate over cols
            normalizeBLE_T( C_LU.at( 
                cToS.at( std::make_pair(0, (row+1)%sizeN) ) ) );
            
            for (int c=0; c<sizeM; c++) {
                normalizeBLE_T( T_U.at( 
                    cToS.at( std::make_pair(c, (row+1)%sizeN) ) ) );
            }

            normalizeBLE_T( C_RU.at( 
                cToS.at( std::make_pair(sizeM-1, (row+1)%sizeN) ) ) );
            break;
        }
        case 'R': { // row is ignored as we iterate over rows
            normalizeBLE_T( C_RU.at( 
                cToS.at( std::make_pair((col-1+sizeM)%sizeM, 0) ) ) );
            
            for (int r=0; r<sizeN; r++) {
                normalizeBLE_T( T_R.at( 
                    cToS.at( std::make_pair((col-1+sizeM)%sizeM, r) ) ) );
            }

            normalizeBLE_T( C_RD.at( 
                cToS.at( std::make_pair((col-1+sizeM)%sizeM, sizeN-1) ) ) );
            break;
        }
        case 'D': { // col is ignored as we iterate over cols
            normalizeBLE_T( C_LD.at( 
                cToS.at( std::make_pair(0, (row-1+sizeN)%sizeN) ) ) );
            
            for (int c=0; c<sizeM; c++) {
                normalizeBLE_T( T_D.at( 
                    cToS.at( std::make_pair(c, (row-1+sizeN)%sizeN) ) ) );
            }

            normalizeBLE_T( C_RD.at( 
                cToS.at( std::make_pair(sizeM-1, (row-1+sizeN)%sizeN) ) ) );
            break;
        }
        case 'L': { // row is ignored as we iterate over rows
            normalizeBLE_T( C_LU.at( 
                cToS.at( std::make_pair((col+1)%sizeM, 0) ) ) );
            std::cout <<"C_LU ["<< (col+1)%sizeM <<","<< 0 <<"]"<< std::endl;

            for (int r=0; r<sizeN; r++) {
                normalizeBLE_T( T_L.at( 
                    cToS.at( std::make_pair((col+1)%sizeM, r) ) ) );
                std::cout <<"T_L ["<< (col+1)%sizeM <<","<< r <<"]"
                    << std::endl;
            }

            normalizeBLE_T( C_LD.at( 
                cToS.at( std::make_pair((col+1)%sizeM, sizeN-1) ) ) );
            std::cout <<"C_LD ["<< (col+1)%sizeM <<","<< sizeN-1 <<"]"
                << std::endl;

            break;
        }
        default: {
            std::cout <<"Unsupported ctmMove type - expecting one of "
                <<" U,R,D or L"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }

    std::cout <<"----- normalizeBLE_ctmStep for "<< ctmMove <<" END -----"
        << std::endl;
}

/*
 * Normalize the tensors Proportional To (Frobenius) Norm (PTN)
 *
 */
void CtmEnv::normalizePTN() {

    for ( auto& t : C_LU ) { t /= norm(t); }
    for ( auto& t : C_RU ) { t /= norm(t); }
    for ( auto& t : C_RD ) { t /= norm(t); }
    for ( auto& t : C_LD ) { t /= norm(t); }
    for ( auto& t : T_U ) { t /= norm(t); }
    for ( auto& t : T_R ) { t /= norm(t); }
    for ( auto& t : T_D ) { t /= norm(t); }
    for ( auto& t : T_L ) { t /= norm(t); }

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
void CtmEnv::normalizePTN_ctmStep(char ctmMove, int col, int row) {

    auto normalizePTN_T = [](ITensor& t) { t /= norm(t); };

    std::cout <<"----- normalizePTN_ctmStep called for "<< ctmMove 
        <<" ["<< col <<","<< row <<"]-----"<< std::endl;

    switch(ctmMove) {
        case 'U': { // col is ignored as we iterate over cols
            normalizePTN_T( C_LU.at( 
                cToS.at( std::make_pair(0, (row+1)%sizeN) ) ) );
            
            for (int c=0; c<sizeM; c++) {
                normalizePTN_T( T_U.at( 
                    cToS.at( std::make_pair(c, (row+1)%sizeN) ) ) );
            }

            normalizePTN_T( C_RU.at( 
                cToS.at( std::make_pair(sizeM-1, (row+1)%sizeN) ) ) );
            break;
        }
        case 'R': { // row is ignored as we iterate over rows
            normalizePTN_T( C_RU.at( 
                cToS.at( std::make_pair((col-1+sizeM)%sizeM, 0) ) ) );
            
            for (int r=0; r<sizeN; r++) {
                normalizePTN_T( T_R.at( 
                    cToS.at( std::make_pair((col-1+sizeM)%sizeM, r) ) ) );
            }

            normalizePTN_T( C_RD.at( 
                cToS.at( std::make_pair((col-1+sizeM)%sizeM, sizeN-1) ) ) );
            break;
        }
        case 'D': { // col is ignored as we iterate over cols
            normalizePTN_T( C_LD.at( 
                cToS.at( std::make_pair(0, (row-1+sizeN)%sizeN) ) ) );
            
            for (int c=0; c<sizeM; c++) {
                normalizePTN_T( T_D.at( 
                    cToS.at( std::make_pair(c, (row-1+sizeN)%sizeN) ) ) );
            }

            normalizePTN_T( C_RD.at( 
                cToS.at( std::make_pair(sizeM-1, (row-1+sizeN)%sizeN) ) ) );
            break;
        }
        case 'L': { // row is ignored as we iterate over rows
            normalizePTN_T( C_LU.at( 
                cToS.at( std::make_pair((col+1)%sizeM, 0) ) ) );
            std::cout <<"C_LU ["<< (col+1)%sizeM <<","<< 0 <<"]"<< std::endl;

            for (int r=0; r<sizeN; r++) {
                normalizePTN_T( T_L.at( 
                    cToS.at( std::make_pair((col+1)%sizeM, r) ) ) );
                std::cout <<"T_L ["<< (col+1)%sizeM <<","<< r <<"]"
                    << std::endl;
            }

            normalizePTN_T( C_LD.at( 
                cToS.at( std::make_pair((col+1)%sizeM, sizeN-1) ) ) );
            std::cout <<"C_LD ["<< (col+1)%sizeM <<","<< sizeN-1 <<"]"
                << std::endl;

            break;
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
     * I_U1--|C_RU|--I_R
     *
     */
    U = ITensor(prime(I_U,1));
    svd( C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_cru[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_R1--|C_RD|--I_D1
     *
     */
    U = ITensor(prime(I_R,1));
    svd( C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) ), U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_crd[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_D--|C_LD|--I_L1
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
 * TODO Using indexing convention from ctm-cluster-env.cc
 */
CtmData CtmEnv::getCtmData() const {
    std::vector< itensor::ITensor > tT_U;
    std::vector< itensor::ITensor > tT_R;
    std::vector< itensor::ITensor > tT_D;
    std::vector< itensor::ITensor > tT_L;

    for( int col=0; col<sizeM; col++) {
        tT_U.push_back( prime(
            T_U.at( cToS.at(std::make_pair(col,0)) ), ULINK, col) );
        tT_D.push_back( mapprime(
            T_D.at( cToS.at(std::make_pair(col,sizeN-1)) ), 
            VSLINK, 1, sizeN,
            DLINK, 0, col,
            DLINK, 1, col+1) );
    }
    for( int row=0; row<sizeN; row++) {
        tT_L.push_back( prime(
            T_L.at( cToS.at(std::make_pair(0,row)) ), LLINK, row) );
        tT_R.push_back( mapprime(
            T_R.at( cToS.at(std::make_pair(sizeM-1,row)) ),
            HSLINK, 1, sizeM,
            RLINK, 0, row,
            RLINK, 1, row+1) );
    }

    CtmData ctmData = {
        x, d, sizeN, sizeM,
        sites, cToS, 
        tT_U, tT_R, tT_D, tT_L,
        C_LU.at( cToS.at(std::make_pair(0,0)) ),
        mapprime( C_RU.at( cToS.at(std::make_pair(sizeM-1,0)) ),
            ULINK, 1, sizeM),
        mapprime( C_RD.at( cToS.at(std::make_pair(sizeM-1,sizeN-1)) ),
            RLINK, 1, sizeN,
            DLINK, 1, sizeM ),
        mapprime( C_LD.at( cToS.at(std::make_pair(0,sizeN-1)) ),
            LLINK, 1, sizeN),
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
            <<" -> "<< siteIds[cToSEntry.second] << std::endl;
    }
    s <<"]"<< std::endl;
    
    s <<"sites: ["<< std::endl;
    for( std::size_t i=0; i<sites.size(); i++) {
        s << WS4 << siteIds[i] <<" -> "<< i;
        printfln(" = %s", sites[i]);
    }
    s <<"]"<< std::endl;

    // Loop over inequivalent sites and print their environment
    for (std::size_t i=0; i<sites.size(); i++) {
        s <<"===="<<" BEGIN ENV OF SITE "<< siteIds[i] <<" "<< 
            std::string(47,'=') << std::endl;
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

        s <<"----"<< siteIds[i] <<" start HALF-ROW TENSORS--"<< std::endl;
        s << TAG_T_L << i;
        printfln(" = %s", T_L[i]);
        s << TAG_T_R << i;
        printfln(" = %s", T_R[i]);
        s <<"------end HALF-ROW TENSORS--"<< std::endl;

        s <<"----"<< siteIds[i] <<" start HALF-COL TENSORS--"<< std::endl;
        s << TAG_T_U << i;
        printfln(" = %s", T_U[i]);
        s << TAG_T_D << i;
        printfln(" = %s", T_D[i]);
        s <<"------end HALF-COL TENSORS--"<< std::endl;
        s <<"===="<<" END ENV OF SITE "<< siteIds[i] <<" "<< 
            std::string(49,'=') << std::endl;
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