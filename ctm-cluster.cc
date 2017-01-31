#include "ctm-cluster.h"

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
     *   ____            ____                 ______          ____
     *  |C_LU|----I_U0--|T_U0|--...--I_Um-1--|T_Um-1|--I_Um--|C_RU|
     *    |               |                    |               |
     *   I_L0            I_XV0                I_XV0           I_R0
     *   _|__             |                    |               |
        |T_L0|----I_XH0--                           --I_XHm--|T_R0|
     *    |                                                    |
     *   ...                                                  ...  
     *    |                                                    |
         I_Ln-1                                               I_Rn-1
          |                                                    |
        |T_Ln-1|--I_XH0--                           --I_XHm--|T_Rn-1|
          |               |                    |               |
         I_Ln            I_XVn                I_XVn           I_Rn
     *   _|__            _|_                  _|____          _|__
     *  |C_LD|----I_D0--|T_D0|--...--I_Dm-1--|T_Dm-1|--I_Dm--|C_RD|
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

    // Construct tensors "C_*"
    C_LU = ITensor(I_L, I_U);
    C_RU = ITensor(prime(I_U,sizeM), I_R);
    C_RD = ITensor(prime(I_R,sizeN), prime(I_D,sizeM));
    C_LD = ITensor(I_D, prime(I_L,sizeN));

    // Construct half-row tensor T_L* and T_R*
    for( int i=0; i<sizeN; i++ ) {
        T_L.push_back( ITensor(prime(I_L,i), prime(I_L,i+1), I_XH) );
        T_R.push_back( ITensor(prime(I_R,i), prime(I_R,i+1), 
            prime(I_XH,1)) );
    }

    // Construct half-column tensor T_D* and T_R*
    for( int i=0; i<sizeM; i++ ) {
        T_U.push_back( ITensor(prime(I_U,i), prime(I_U,i+1), I_XV) );
        T_D.push_back( ITensor(prime(I_D,i), prime(I_D,i+1),
            prime(I_XV,1)) );
    }
    
    // Iterate over siteIds as given in Cluster c
    for( const auto& siteIdEntry : c.siteIds ) {
        sites.push_back( contractOST(c.sites.at(siteIdEntry)) );

        // add entries for map from cluster site to tensor 
        for ( const auto& cToSEntry : c.cToS ) {
            if( cToSEntry.second == siteIdEntry ) {
                cToS[ cToSEntry.first ] = sites.size()-1;
            }
        }
    }
    
    I_Fxd = Index(TAG_I_FXD, x*d);
    I_Tx  = Index(TAG_I_TX, x);
    R = ITensor(I_Fxd, I_Tx);
    for(int i=1; i<=x; i++) { R.set(I_Fxd(i), I_Tx(i), 1.+0._i); }

    // Construct vectors holding SVD spectrum of corner matrices
    spec = {
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x),
        std::vector< double >(x)
    };
}

CtmEnv::CtmEnv (std::string t_name, CtmData const& ctmD, Cluster const& c)
    : m_name(t_name), d(ctmD.auxDimSite), x(ctmD.auxDimEnv),
    sizeN(ctmD.sizeN), sizeM(ctmD.sizeM),
    // half-row/column tensors
    T_U(ctmD.T_U), T_R(ctmD.T_R), T_D(ctmD.T_D), T_L(ctmD.T_L),
    // corner matrices 
    C_LD(ctmD.C_LD), C_LU(ctmD.C_LU), C_RU(ctmD.C_RU), C_RD(ctmD.C_RD), 
    // environment indices
    I_U(ctmD.I_U), I_R(ctmD.I_R), I_D(ctmD.I_D), I_L(ctmD.I_L),
    // on-site indices
    I_XH(ctmD.I_XH), I_XV(ctmD.I_XV) {

    // consistency checks
    if ( ctmD.sizeN != c.sizeN || ctmD.sizeM != c.sizeM ) {
        std::cout <<"Dimensions of cluster and environment "<<
            "are not in agreement"<< std::endl;
        exit(EXIT_FAILURE);
    }
    if ( ctmD.auxDimSite != c.auxBondDim*c.auxBondDim ) {
        std::cout <<"AuxDimSite of environment and auxBondDim**2 "
            <<"of cluster are not in agreement"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Iterate over siteIds as given in Cluster c
    for( const auto& siteIdEntry : c.siteIds ) {
        sites.push_back( contractOST(c.sites.at(siteIdEntry)) );

        // add entries for map from cluster site to tensor 
        for ( const auto& cToSEntry : c.cToS ) {
            if( cToSEntry.second == siteIdEntry ) {
                cToS[ cToSEntry.first ] = sites.size()-1;
            }
        }
    }
    
    I_Fxd = Index(TAG_I_FXD, x*d);
    I_Tx  = Index(TAG_I_TX, x);
    R = ITensor(I_Fxd, I_Tx);
    for(int i=1; i<=x; i++) { R.set(I_Fxd(i), I_Tx(i), 1.+0._i); }

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
    auto auxI = noprime(findtype(T.inds(), Link));

    if(auxI.m()*auxI.m() != d) {
        std::cout <<"ctmEnv.d does not agree with onSiteT.dimD^2"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Define combiner tensors Y*
    auto C04 = combiner(auxI, prime(auxI,4));
    auto C15 = prime(C04,1);
    auto C26 = prime(C04,2);
    auto C37 = prime(C04,3);

    auto X = (T*( conj(T).prime(Link,4) ))*C04*C15*C26*C37;

    // Define delta tensors D* to relabel combiner indices to I_XH, I_XV
    auto DH0 = delta(I_XH, commonIndex(X,C04));
    auto DV0 = delta(I_XV, commonIndex(X,C15));
    auto DH1 = delta(prime(I_XH,1), commonIndex(X,C26));
    auto DV1 = delta(prime(I_XV,1), commonIndex(X,C37));

    return X*DH0*DV0*DH1*DV1;
}

// ########################################################################
// environment initalization methods
/*
 * TODO? implement enum INIT_ENV to string for purpose of debug output 
 *
 */

void CtmEnv::initMockEnv() {
    C_LU.fill(1.+0._i);
    C_RU.fill(1.+0._i);
    C_RD.fill(1.+0._i);
    C_LD.fill(1.+0._i);
    for ( auto& t_u : T_U ) { t_u.fill(1.+0._i); }
    for ( auto& t_r : T_R ) { t_r.fill(1.+0._i); }
    for ( auto& t_d : T_D ) { t_d.fill(1.+0._i); }
    for ( auto& t_l : T_L ) { t_l.fill(1.+0._i); }

    CtmEnv::computeSVDspec();

    std::cout <<"INIT_ENV_const1 with all elements of C's and T's"<<
        " set to constant"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

void CtmEnv::initRndEnv(bool isComplex) {
    //Fill with random numbers (real or complex)
    randomize(C_LU, {"Complex", isComplex});
    randomize(C_RU, {"Complex", isComplex});
    randomize(C_RD, {"Complex", isComplex});
    randomize(C_LD, {"Complex", isComplex});
    for ( auto& t_u : T_U ) { randomize(t_u, {"Complex", isComplex}); }
    for ( auto& t_r : T_R ) { randomize(t_r, {"Complex", isComplex}); }
    for ( auto& t_d : T_D ) { randomize(t_d, {"Complex", isComplex}); }
    for ( auto& t_l : T_L ) { randomize(t_l, {"Complex", isComplex}); }

    CtmEnv::computeSVDspec();

    std::cout <<"INIT_ENV_rnd with all C's and T's random (complex ? " 
        << isComplex <<")"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

/*
 * TODO expose parameters - scale of "noise", real or complex 
 *
 */
// void CtmEnv::initFromFile(int chi, IO_ENV_FMT ioFmt, std::string TAG) {
//     std::cout << "INIT_ENV_file with prefix " << TAG << "\n";
//     CtmData inCtmD = readEnv(ioFmt, TAG);

//     // If the desired dimension of environment is higher, then
//     // the dimension of environment stored within input files,
//     // use the environment of lower dimensions as "Ansatz"
//     if( chi > inCtmD.chi ) {
//         std::cout <<" ENV_file( chi="<< inCtmD.chi
//             <<" ) < ENV_target( chi="<< chi <<"). Using ENV_file as ANSATZ"
//             <<"\n";

//         bool isComplex = false;

//         // Initalize ENV(chi) rnd/sym
//         initSymEnv(isComplex);
        
//         double mag = 0.001;

//         // Scale "noise" by mag
//         C_lu = mag*C_lu;
//         C_ru = mag*C_ru;
//         C_rd = mag*C_rd;
//         C_ld = mag*C_ld;
//         T_u = mag*T_u;
//         T_r = mag*T_r;
//         T_d = mag*T_d;
//         T_l = mag*T_l;

//         std::cout << "Current ENV" << "\n";
//         std::cout << getCtmData();

//         // Add inCtmD(==ENV_file) to ENV_target
//         C_lu = C_lu + delta(i_Tl, inCtmD.i_Tl)*inCtmD.C_lu
//             *delta(inCtmD.i_Tu, i_Tu);
//         C_ru = C_ru + prime(delta(i_Tu, inCtmD.i_Tu),1)*inCtmD.C_ru
//             *delta(inCtmD.i_Tr, i_Tr);
//         C_rd = C_rd + prime(delta(i_Tr, inCtmD.i_Tr),1)*inCtmD.C_rd
//             *prime(delta(i_Td, inCtmD.i_Td),1);
//         C_ld = C_ld + delta(i_Td, inCtmD.i_Td)*inCtmD.C_ld
//             *prime(delta(inCtmD.i_Tl, i_Tl),1);

//         std::cout << "C's generated" << "\n";

//         inCtmD.T_u = inCtmD.T_u*delta(i_Xv, inCtmD.i_Xv);
//         inCtmD.T_r = inCtmD.T_r*prime(delta(i_Xh, inCtmD.i_Xh),1);
//         inCtmD.T_d = inCtmD.T_d*prime(delta(i_Xv, inCtmD.i_Xv),1);
//         inCtmD.T_l = inCtmD.T_l*delta(i_Xh, inCtmD.i_Xh);

//         T_u = T_u + delta(i_Tu, inCtmD.i_Tu)*inCtmD.T_u
//             *prime(delta(inCtmD.i_Tu, i_Tu),1);
//         T_r = T_r + delta(i_Tr, inCtmD.i_Tr)*inCtmD.T_r
//             *prime(delta(inCtmD.i_Tr, i_Tr),1);
//         T_d = T_d + delta(i_Td, inCtmD.i_Td)*inCtmD.T_d
//             *prime(delta(inCtmD.i_Td, i_Td),1);
//         T_l = T_l + delta(i_Tl, inCtmD.i_Tl)*inCtmD.T_l
//             *prime(delta(inCtmD.i_Tl, i_Tl),1);

//         std::cout << "T's generated" << "\n";
    
//     } else if( chi < inCtmD.chi ) {
//         // TODO
//         std::cout << "ENV_file( chi="<< inCtmD.chi
//             <<" ) > ENV_target( chi="<< chi <<")" << "\n";
//         exit(EXIT_FAILURE);
//     } else {
//         setCtmData(inCtmD);
//     }

//     std::cout << std::string(72,'=') << "\n";
// }

// ########################################################################
// CTM iterative methods

void CtmEnv::insURow_DBG(CtmEnv::ISOMETRY iso_type) {
    std::cout <<"##### InsURow called "<< std::string(51,'#') << std::endl;

    // sequentialy contract upper boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*
    for (int row=0; row<sizeN; row++) {

        /*
         * Insert a copy of row into network to obtain following expanded TN:
         *     ____         ____                 ______          ____
         *  A |C_LU|--I_U--|T_U0|--...--I_Um-1--|T_Um-1|--I_Um--|C_RU|  A
         *  A   |            |                    |               |     A
         *  A  I_L0         I_XV                 I_XV            I_R0   A
         *  A  _|__                                              _|__   A 
         *  A |T_L0|--I_XH                               I_XH1--|T_R0|  A    
         *  A   |                                                 |     A
         *  A  I_L1                                              I_R1   A
         *
         * Focus on a part of the TN highlighted by A's. Absorb the 
         * inserted tensors to obtain TN of the original form
         *
         * 1) Contract C_LU with T_L(row)
         *   ____                 ____
         *  |C_LU|--I_U          |    |--I_U
         *    |               => |C_LU| 
         *   I_L(row)         => |____|--I_XH
         *   _|______         =>    |
         *  |T_L(row)|--I_XH       I_L(row+1)
         *    |
         *   I_L(row+1)
         * 
         * 2) Contract C_RU with T_R(row)
         *         ____                  ____
         *  I_Um--|C_RU|          I_Um--|    |
         *          |         =>        |C_RU|  
         *         I_R(row)   => I_XH1--|____|
         *         _|______   =>           |
         * I_XH1--|T_R(row)|              I_R(row+1) 
         *          |
         *         I_R(row+1)
         *
         */
        std::cout <<"(1) ----- C_LU & T_L"<< row <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_LU);
        Print(T_L[row]);

        C_LU *= T_L[row];

        std::cout <<"After contraction"<< std::endl;
        Print(C_LU);

        std::cout <<"(2) ----- C_RU & T_R"<< row <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_RU);
        Print(T_R[row]);

        C_RU *= T_R[row];

        std::cout <<"After contraction"<< std::endl;
        Print(C_RU);

        /*
         * compute one of the "trivial" isometries
         *
         * TODO? how to include more elaborate isometries
         * TODO? call isometry creation as a function
         *
         */
        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;

        std::pair< ITensor, ITensor > isoPair;
        
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoPair = isoT1( IndexSet(I_U, I_XH), 
                    std::make_pair(prime(I_L,row+1),prime(I_R,row+1)),
                    C_LU, C_RU);
                break;
            }
            case ISOMETRY_T2: {
                isoPair = isoT2( IndexSet(I_U, I_XH),
                    C_LU, C_LD, C_RD, C_RU);
                break;
            }
            case ISOMETRY_T3: {
                isoPair = isoT3(
                    std::make_pair(I_U, prime(I_U,sizeM)),
                    std::make_pair(I_XH, prime(I_XH,1)),
                    C_LU, C_RU);
                break;
            }
            default: {
                std::cout <<"Unsupported Isometry type"<< std::endl;
                exit(EXIT_FAILURE);
                break;
            }
        }

        // references to individual isometry tensors
        ITensor& tU = isoPair.first;
        ITensor& tV = isoPair.second;

        /*
         * Obtain new C_LU,C_RU by contraction of C_LU,C_RU 
         * with an isometry U,V and reduction matrix R
         *   ____          _          
         *  |    |--I_U---| \
         *  |C_LU|        |tU--I_SVD--|R|--I_U = |C_LU|--I_U  
         *  |____|--I_XH--|_/                      |
         *    |                                   I_L(row+1)
         *   I_L(row+1)
         *                                     _           ____ 
         *                                    / |---I_Um--|    |
         *   I_Um--|C_RU|= I_Um--|R|--I_SVD1--tV|         |C_RU|
         *           |                        \_|--I_XH1--|____|
         *          I_R(row+1)                              |
         *                                                 I_R(row+1)
         * 
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(5) ----- Construct reduced C_LU,C_RU -----"<< std::endl;
        auto I_SVD_U = findtype(tU.inds(),SVD_LINK);
        auto I_SVD_V = findtype(tV.inds(),SVD_LINK);
        C_LU = ( (C_LU*tU) *delta(I_SVD_U,I_Fxd)*R )*delta(I_Tx,I_U);
        C_RU = ( (C_RU*tV) *delta(I_SVD_V,I_Fxd)*R )
            *delta(I_Tx,prime(I_U,sizeM));

        Print(C_LU);
        Print(C_RU);

        /* 
         * Contract T_U[0..sizeM-1] with X[row,col] to obtain new T_U
         *            ___                           ___
         * I_U(col)--|T_U|--I_U(col+1)   I_U(col)--|   |--I_U(col+1)
         *             |                           |T_U|  
         *            I_XV         =>        I_XH--|___|--I_XH1
         *            _|_          =>                |
         *     I_XH--| X |--I_XH1                   I_XV1 
         *             |
         *            I_XV1
         *
         * Obtain new T_U by contraction of T_U with an isometry tU, tV and
         * reduction matrix R as follows (since C_LU was reduced by tU--R--
         * from the right)
         *                          _              ___ 
         *                         / |--I_U(col)--|   |--I_U(col+1)
         *  I_U(col)--|R|--I_SVD1--tV|            |T_U|
         *                         \_|------I_XH--|___|--I_XH1
         *                                          |                     
         *                                        I_XV1
         * and from the right
         *            ___                _
         * I_U(col)--|   |--I_U(col+1)--| \
         *           |T_U|              |tU--I_SVD--|R|--I_U(col+1)  
         *           |___|--I_XH1-------|_/
         *             |
         *            I_XV1 
         *
         *
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(6) ----- T_U & X "<< std::string(54,'-') << std::endl;
        tV.noprime();
        tU.prime(ULINK,HSLINK);
        for (int col=0; col<sizeM; col++) {
            std::cout <<"--- Before contraction T_U["<< col <<"] & X["<< row
                <<","<< col <<"] ---"<< std::endl;
            std::cout << TAG_T_U <<"["<< col <<"]";
            printfln("= %s",T_U[col]);
            std::cout <<"("<< row <<","<< col <<") -> sites["<< 
                cToS[std::make_pair(row,col)] <<"]"<< std::endl;

            T_U[col] *= sites[cToS[std::make_pair(row,col)]];

            std::cout <<"After contraction Col="<< col << std::endl;
            std::cout << TAG_T_U <<"["<< col <<"]";
            printfln("= %s",T_U[col]);
        
            std::cout <<" --- Construct reduced T_U["<< col <<"] ---"
                << std::endl;
            
            T_U[col] = ( ( T_U[col]*tV.mapprime(ULINK,col-1,col) ) 
                *delta(I_SVD_V,I_Fxd)*R ) *delta(I_Tx,prime(I_U,col));

            T_U[col] = ( ( T_U[col]*tU.mapprime(ULINK,col,col+1) ) 
                *delta(I_SVD_U,I_Fxd)*R ) *delta(I_Tx,prime(I_U,col+1));

            T_U[col].noprime(VSLINK);
            std::cout << TAG_T_U <<"["<< col <<"]";
            printfln("= %s",T_U[col]);
        }

        std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    // End of cluster absorption
    C_LU.noprime(LLINK);
    C_RU.noprime(RLINK);
 
    std::cout <<"##### InsURow Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insDRow_DBG(CtmEnv::ISOMETRY iso_type) {
    std::cout <<"##### InsURow called "<< std::string(51,'#') << std::endl;
    // sequentialy contract lower boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*
    for (int row=sizeN-1; row>=0; row--) {

        /*
         * Insert a copy of row into network to obtain following expanded TN:
         *
         *  A  I_Ln-1                                            I_Rn-1  A
         *  A  _|____                                            _|____  A 
         *  A |T_Ln-1|--I_XH                             I_XH1--|T_Rn-1| A    
         *  A   |                                                 |      A
         *  A  I_Ln         I_XV1                I_XV1           I_Rn    A
         *  A  _|__         _|__                 _|____          _|__    A
         *  A |C_LD|--I_D--|T_D0|--...--I_Dm-1--|T_Dm-1|--I_Dm--|C_RD|   A
         *
         * Focus on a part of the TN highlighted by A's. Absorb the 
         * inserted tensors to obtain TN of the original form
         *
         * 1) Contract C_LU with T_L(row)
         *
         *   I_L(row)            I_(row)
         *   _|______            _|__
         *  |T_L(row)|--I_XH    |    |--I_XH
         *    |              => |C_LD| 
         *   I_L(row+1)      => |____|--I_D
         *   _|__            => 
         *  |C_LD|------I_D
         *
         * 2) Contract C_RU with T_R(row)
         *         
         *         I_R(row)               I_R(row)
         *         _|______               _|__
         * I_XH1--|T_R(row)|      I_XH1--|    |
         *          |         =>         |C_RU|  
         *         I_R(row+1) =>   I_Dm--|____|
         *         _|__       =>
         *  I_Dm--|C_RD|
         *
         */
        std::cout <<"(1) ----- C_LD & T_L"<< row <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_LD);
        Print(T_L[row]);

        C_LD *= T_L[row];

        std::cout <<"After contraction"<< std::endl;
        Print(C_LD);

        std::cout <<"(2) ----- C_RD & T_R"<< row <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_RD);
        Print(T_R[row]);

        C_RD *= T_R[row];

        std::cout <<"After contraction"<< std::endl;
        Print(C_RD);

        /*
         * compute one of the "trivial" isometries
         *
         */
        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
        std::pair< ITensor, ITensor > isoPair;
        
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoPair = isoT1( IndexSet(I_D, I_XH), 
                    std::make_pair( prime(I_L,row),prime(I_R,row) ),
                    C_LD, C_RD);
                break;
            }
            case ISOMETRY_T2: {
                isoPair = isoT2( IndexSet(I_D, I_XH),
                    C_LD, C_LU, C_RU, C_RD);
                break;
            }
            case ISOMETRY_T3: {
                isoPair = isoT3(
                    std::make_pair(I_D, prime(I_D,sizeM)),
                    std::make_pair(I_XH, prime(I_XH,1)),
                    C_LD, C_RD);
                break;
            }
            default: {
                std::cout <<"Unsupported Isometry type"<< std::endl;
                exit(EXIT_FAILURE);
                break;
            }
        }

        // references to individual isometry tensors
        ITensor& tU = isoPair.first;
        ITensor& tV = isoPair.second;

        /*
         * Obtain new C_LD,C_RD by contraction of C_LD,C_RD 
         * with an isometry U,V and reduction matrix R
         *
         *   I_L(row)
         *   _|__          _                      I_L(row)
         *  |    |--I_D---| \                      |
         *  |C_LD|        |tU--I_SVD--|R|--I_D = |C_LD|--I_D  
         *  |____|--I_XH--|_/  
         *
         *                                                 I_R(row)
         *          I_R(row)                   _           _|__ 
         *           |                        / |---I_Dm--|    |
         *   I_Dm--|C_RD|= I_Dm--|R|--I_SVD1--tV|         |C_RD|
         *                                    \_|--I_XH1--|____|
         * 
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(5) ----- Construct reduced C_LD,C_RD -----"<< std::endl;
        auto I_SVD_U = findtype(tU.inds(),SVD_LINK);
        auto I_SVD_V = findtype(tV.inds(),SVD_LINK);
        C_LD = ( (C_LD*tU) *delta(I_SVD_U,I_Fxd)*R )*delta(I_Tx,I_D);
        C_RD = ( (C_RD*tV) *delta(I_SVD_V,I_Fxd)*R )
            *delta(I_Tx,prime(I_D,sizeM));

        Print(C_LD);
        Print(C_RD);
        
        /* 
         * Contract T_D[0..sizeM-1] with X[row,col] to obtain new T_D
         *
         *            I_XV                           I_XV  
         *            _|_                            _|_
         *     I_XH--| X |--I_XH1             I_XH--|   |--I_XH1
         *             |                            |T_D|  
         *            I_XV1            => I_D(col)--|___|--I_D(col+1)
         *            _|_              =>
         * I_D(col)--|T_D |--I_D(col+1)
         *
         * Obtain new T_D by contraction of T_D with an isometry tU, tV and
         * reduction matrix R as follows (since C_LD was reduced by tU--R--
         * from the right)
         *
         *                                         I_XV 
         *                          _              _|_ 
         *                         / |--I_XH------|   |--I_XH1
         *  I_D(col)--|R|--I_SVD1--tV|            |T_D|
         *                         \_|--I_D(col)--|___|--I_D(col+1)
         * and from the right
         *
         *            I_XV
         *            _|_                _
         * I_D(col)--|   |--I_XH1-------| \
         *           |T_D|              |tU--I_SVD--|R|--I_D(col+1)  
         *           |___|--I_D(col+1)--|_/
         *
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(6) ----- T_D & X "<< std::string(54,'-') << std::endl;
        tV.noprime();
        tU.prime(DLINK,HSLINK);
        for (int col=0; col<sizeM; col++) {
            std::cout <<"--- Before contraction T_D["<< col <<"] & X["<< row
                <<","<< col <<"] ---"<< std::endl;
            std::cout << TAG_T_D <<"["<< col <<"]";
            printfln("= %s",T_D[col]);
            std::cout <<"("<< row <<","<< col <<") -> sites["<< 
                cToS[std::make_pair(row,col)] <<"]"<< std::endl;

            T_D[col] *= sites[cToS[std::make_pair(row,col)]];

            std::cout <<"After contraction Col="<< col << std::endl;
            std::cout << TAG_T_D <<"["<< col <<"]";
            printfln("= %s",T_D[col]);
        
            std::cout <<" --- Construct reduced T_D["<< col <<"] ---"
                << std::endl;
            
            T_D[col] = ( ( T_D[col]*tV.mapprime(DLINK,col-1,col) ) 
                *delta(I_SVD_V,I_Fxd)*R ) *delta(I_Tx,prime(I_D,col));

            T_D[col] = ( ( T_D[col]*tU.mapprime(DLINK,col,col+1) ) 
                *delta(I_SVD_U,I_Fxd)*R ) *delta(I_Tx,prime(I_D,col+1));

            T_D[col].prime(VSLINK);
            std::cout << TAG_T_D <<"["<< col <<"]";
            printfln("= %s",T_D[col]);
        }

        std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    // End of cluster absorption
    C_LD.prime(LLINK,sizeN);
    C_RD.prime(RLINK,sizeN);

    std::cout <<"##### InsDRow Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insLCol_DBG(CtmEnv::ISOMETRY iso_type) {
    std::cout <<"##### InsLCol called "<< std::string(51,'#') << std::endl;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*
    for (int col=0; col<sizeM; col++) {
        
        /*
         * Insert a copy of column into network to obtain following expanded TN:
         *
         *  AAAAAAAAAAAAAAAAAAAAAAAAAAAA
         *   ____              ________      
         *  |C_LU|--I_U(col)--|T_U(col)|--I_U(col+1)
         *    |                 |
         *   I_L               I_XV
         *   _|__
         *  |T_L0|--I_XH   
         *    |
         *   I_L1
         *    |
         *   ...
         *    |
         *   I_Ln-1
         *   _|
         *  |T_Ln-1|--I_XH
         *    |
         *   I_Ln              I_XV1  
         *   _|__              _|______
         *  |C_LD|--I_D(col)--|T_D(col)|--I_D(col+1)
         *
         *  AAAAAAAAAAAAAAAAAAAAAAAAAAAA   
         *
         * Focus on a part of the TN highlighted by A's. Absorb the 
         * inserted tensors to obtain TN of the original form
         *
         * 1) Contract C_LU with T_U(col)
         *   ____            ___                 ______
         *  |C_LU|-I_U(col)-|T_U|-I_U(col+1) => |C_LU__|--I_U(col+1)  
         *    |               |                   |   |    
         *   I_L             I_XV                I_L I_XV    
         *
         * 2) Contract C_LD with T_D(col)
         *
         *   I_Ln              I_XV1                I_Ln  I_XV1
         *   _|__              _|_                  _|_____|_
         *  |C_LD|--I_D(col)--|T_D|--I_U(col+1) => |___C_LD__|--I_U(col+1)
         *
         */
        std::cout <<"(1) ----- C_LU & T_U"<< col <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_LU);
        Print(T_U[col]);

        C_LU *= T_U[col];

        std::cout <<"After contraction"<< std::endl;
        Print(C_LU);

        std::cout <<"(2) ----- C_LD & T_D"<< col <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_LD);
        Print(T_D[col]);

        C_LD *= T_D[col];

        std::cout <<"After contraction"<< std::endl;
        Print(C_LD);

        /*
         * compute one of the "trivial" isometries
         *
         */
        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
        std::pair< ITensor, ITensor > isoPair;
        
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoPair = isoT1( IndexSet(I_L, I_XV), 
                    std::make_pair( prime(I_U,col+1),prime(I_D,col+1) ),
                    C_LU, C_LD);
                break;
            }
            case ISOMETRY_T2: {
                isoPair = isoT2( IndexSet(I_L, I_XV),
                    C_LU, C_RU, C_RD, C_LD);
                break;
            }
            case ISOMETRY_T3: {
                isoPair = isoT3(
                    std::make_pair(I_L, prime(I_L,sizeN)),
                    std::make_pair(I_XV, prime(I_XV,1)),
                    C_LU, C_LD);
                break;
            }
            default: {
                std::cout <<"Unsupported Isometry type"<< std::endl;
                exit(EXIT_FAILURE);
                break;
            }
        }

        // references to individual isometry tensors
        ITensor& tU = isoPair.first;
        ITensor& tV = isoPair.second;

        /*
         * Obtain new C_LU,C_LD by contraction of C_LU,C_LD 
         * with an isometry U,V and reduction matrix R
         *
         *   I_U(col+1)
         *   _|__          _                      I_U(col+1)
         *  |    |--I_L---| \                      |
         *  |C_LU|        |tU--I_SVD--|R|--I_L = |C_LU|--I_L  
         *  |____|--I_XV--|_/  
         *
         *                                                 I_D(col+1)
         *          I_D(col+1)                 _           _|__ 
         *           |                        / |---I_Ln--|    |
         *   I_Ln--|C_LD|= I_Ln--|R|--I_SVD1--tV|         |C_LD|
         *                                    \_|--I_XV1--|____|
         * 
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(5) ----- Construct reduced C_LU,C_LD -----"<< std::endl;
        auto I_SVD_U = findtype(tU.inds(),SVD_LINK);
        auto I_SVD_V = findtype(tV.inds(),SVD_LINK);
        C_LU = ( (C_LU*tU) *delta(I_SVD_U,I_Fxd)*R )*delta(I_Tx,I_L);
        C_LD = ( (C_LD*tV) *delta(I_SVD_V,I_Fxd)*R )
            *delta(I_Tx,prime(I_L,sizeN));

        Print(C_LU);
        Print(C_LD);

        /* 
         * Contract T_L[0..sizeN-1] with X[row,col] to obtain new T_L
         *
         *            I_XH1                          I_XH1  
         *            _|_                            _|_
         *     I_XV--| X |--I_XV1             I_XV--|   |--I_XV1
         *             |                            |T_L|  
         *            I_XH             => I_L(row)--|___|--I_L(row+1)
         *            _|_              =>
         * I_L(row)--|T_L|--I_L(row+1)
         *
         * Obtain new T_L by contraction of T_L with an isometry tU, tV and
         * reduction matrix R as follows (since C_LU was reduced by tU--R--
         * from the bottom)
         *
         *                                         I_XH1 
         *                          _              _|_ 
         *                         / |--I_XV------|   |--I_XV1
         *  I_L(row)--|R|--I_SVD1--tV|            |T_L|
         *                         \_|--I_L(row)--|___|--I_L(row+1)
         * and from the bottom
         *
         *            I_XH1
         *            _|_                _
         * I_L(row)--|   |--I_XV1-------| \
         *           |T_L|              |tU--I_SVD--|R|--I_L(row+1)  
         *           |___|--I_L(row+1)--|_/
         *
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(6) ----- T_L & X "<< std::string(54,'-') << std::endl;
        tV.noprime();
        tU.prime(LLINK,VSLINK);
        for (int row=0; row<sizeN; row++) {
            std::cout <<"--- Before contraction T_L["<< row <<"] & X["<< row
                <<","<< col <<"] ---"<< std::endl;
            std::cout << TAG_T_L <<"["<< row <<"]";
            printfln("= %s",T_L[row]);
            std::cout <<"("<< row <<","<< col <<") -> sites["<< 
                cToS[std::make_pair(row,col)] <<"]"<< std::endl;

            T_L[row] *= sites[cToS[std::make_pair(row,col)]];

            std::cout <<"After contraction Row="<< row << std::endl;
            std::cout << TAG_T_L <<"["<< row <<"]";
            printfln("= %s",T_L[row]);
        
            std::cout <<" --- Construct reduced T_L["<< row <<"] ---"
                << std::endl;
            
            T_L[row] = ( ( T_L[row]*tV.mapprime(LLINK,row-1,row) ) 
                *delta(I_SVD_V,I_Fxd)*R ) *delta(I_Tx,prime(I_L,row));

            T_L[row] = ( ( T_L[row]*tU.mapprime(LLINK,row,row+1) ) 
                *delta(I_SVD_U,I_Fxd)*R ) *delta(I_Tx,prime(I_L,row+1));

            T_L[row].noprime(HSLINK);
            std::cout << TAG_T_L <<"["<< row <<"]";
            printfln("= %s",T_L[row]);
        }

        std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    // End of cluster absorption
    C_LU.noprime(ULINK);
    C_LD.noprime(DLINK);

    std::cout <<"##### InsLCol Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insRCol_DBG(CtmEnv::ISOMETRY iso_type) {
    std::cout <<"##### InsRCol called "<< std::string(51,'#') << std::endl;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*
    for (int col=sizeM-1; col>=0; col--) {
        
        /*
         * Insert a copy of column into network to obtain following expanded TN:
         *
         *             AAAAAAAAAAAAAAAAAAAAAAAAAAAA
         *             ________                ____      
         *  I_U(col)--|T_U(col)|--I_U(col+1)--|C_RU|
         *              |                       | 
         *             I_XV                    I_R
         *                                     _|__
         *                             I_XH1--|T_R0|
         *                                      |
         *                                     I_R1
         *                                      |
         *                                     ...
         *                                      |
         *                                     I_Rn-1
         *                                     _|____
         *                             I_XH1--|T_Rn-1|
         *                                      |
         *             I_XV1                   I_Rn
         *             _|______                _|__
         *  I_D(col)--|T_D(col)|--I_D(col+1)--|C_RD|
         *
         *              AAAAAAAAAAAAAAAAAAAAAAAAAAAA   
         *
         * Focus on a part of the TN highlighted by A's. Absorb the 
         * inserted tensors to obtain TN of the original form
         *
         * 1) Contract C_RU with T_U(col)
         *             ________                ____                 ______
         *  I_U(col)--|T_U(col)|--I_U(col+1)--|C_RU| =>  I_U(col)--| C_RU |
         *              |                       |    =>             |    |
         *             I_XV                    I_R                 I_XV I_R
         *
         * 2) Contract C_LD with T_D(col)
         *
         *             I_XV1                   I_Rn                I_XV1 I_Rn
         *             _|______                _|__                 |_____|_
         *  I_D(col)--|T_D(col)|--I_D(col+1)--|C_RU| =>  I_U(col)--|_C_RU___|
         *
         */
        std::cout <<"(1) ----- C_RU & T_U"<< col <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_RU);
        Print(T_U[col]);

        C_RU *= T_U[col];

        std::cout <<"After contraction"<< std::endl;
        Print(C_RU);

        std::cout <<"(2) ----- C_RD & T_D"<< col <<"-----"<< std::endl;
        std::cout <<"Before contraction"<< std::endl;
        Print(C_RD);
        Print(T_D[col]);

        C_RD *= T_D[col];

        std::cout <<"After contraction"<< std::endl;
        Print(C_RD);

        /*
         * compute one of the "trivial" isometries
         *
         */
        std::cout <<"(4) ----- Computing Isometry -----"<< std::endl;
    
        std::pair< ITensor, ITensor > isoPair;
        
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoPair = isoT1( IndexSet(I_R, I_XV), 
                    std::make_pair( prime(I_U,col),prime(I_D,col) ),
                    C_RU, C_RD);
                break;
            }
            case ISOMETRY_T2: {
                isoPair = isoT2( IndexSet(I_R, I_XV),
                    C_RU, C_LU, C_LD, C_RD);
                break;
            }
            case ISOMETRY_T3: {
                isoPair = isoT3(
                    std::make_pair(I_R, prime(I_R,sizeN)),
                    std::make_pair(I_XV, prime(I_XV,1)),
                    C_RU, C_RD);
                break;
            }
            default: {
                std::cout <<"Unsupported Isometry type"<< std::endl;
                exit(EXIT_FAILURE);
                break;
            }
        }

        // references to individual isometry tensors
        ITensor& tU = isoPair.first;
        ITensor& tV = isoPair.second;

        /*
         * Obtain new C_LU,C_LD by contraction of C_LU,C_LD 
         * with an isometry U,V and reduction matrix R
         *
         *   I_U(col)
         *   _|__          _                      I_U(col)
         *  |    |--I_R---| \                      |
         *  |C_RU|        |tU--I_SVD--|R|--I_L = |C_RU|--I_R  
         *  |____|--I_XV--|_/  
         *
         *                                                 I_D(col)
         *          I_D(col)                   _           _|__ 
         *           |                        / |---I_Rn--|    |
         *   I_Rn--|C_RD|= I_Rn--|R|--I_SVD1--tV|         |C_RD|
         *                                    \_|--I_XV1--|____|
         * 
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(5) ----- Construct reduced C_RU,C_RD -----"<< std::endl;
        auto I_SVD_U = findtype(tU.inds(),SVD_LINK);
        auto I_SVD_V = findtype(tV.inds(),SVD_LINK);
        C_RU = ( (C_RU*tU) *delta(I_SVD_U,I_Fxd)*R )*delta(I_Tx,I_R);
        C_RD = ( (C_RD*tV) *delta(I_SVD_V,I_Fxd)*R )
            *delta(I_Tx,prime(I_R,sizeN));

        Print(C_RU);
        Print(C_RD);

        /* 
         * Contract T_R[0..sizeN-1] with X[row,col] to obtain new T_R
         *
         *              I_XH                            I_XH
         *              _|_                             _|_
         *      I_XV1--| X |--I_XV              I_XV1--|   |--I_XV
         *               |                             |T_R|  
         *              I_XH1           => I_R(row+1)--|___|--I_R(row)
         *              _|_             =>
         * I_R(row+1)--|T_R|--I_R(row)
         *
         * Obtain new T_R by contraction of T_R with an isometry tU, tV and
         * reduction matrix R as follows (since C_RU was reduced by tU--R--
         * from the bottom)
         *
         *                                         I_XH 
         *                          _              _|_ 
         *                         / |--I_XV------|   |--I_XV1
         *  I_R(row)--|R|--I_SVD1--tV|            |T_R|
         *                         \_|--I_R(row)--|___|--I_R(row+1)
         * and from the bottom
         *
         *            I_XH
         *            _|_                _
         * I_R(row)--|   |--I_XV1-------| \
         *           |T_R|              |tU--I_SVD--|R|--I_R(row+1)  
         *           |___|--I_R(row+1)--|_/
         *
         * and using appropriate delta-matrices to relabel auxiliary reduction 
         * indices to the original ones of the environment
         *
         */
        std::cout <<"(6) ----- T_R & X "<< std::string(54,'-') << std::endl;
        tV.noprime();
        tU.prime(RLINK,VSLINK);
        for (int row=0; row<sizeN; row++) {
            std::cout <<"--- Before contraction T_R["<< row <<"] & X["<< row
                <<","<< col <<"] ---"<< std::endl;
            std::cout << TAG_T_R <<"["<< row <<"]";
            printfln("= %s",T_R[row]);
            std::cout <<"("<< row <<","<< col <<") -> sites["<< 
                cToS[std::make_pair(row,col)] <<"]"<< std::endl;

            T_R[row] *= sites[cToS[std::make_pair(row,col)]];

            std::cout <<"After contraction Row="<< row << std::endl;
            std::cout << TAG_T_R <<"["<< row <<"]";
            printfln("= %s",T_R[row]);
        
            std::cout <<" --- Construct reduced T_R["<< row <<"] ---"
                << std::endl;
            
            T_R[row] = ( ( T_R[row]*tV.mapprime(RLINK,row-1,row) ) 
                *delta(I_SVD_V,I_Fxd)*R ) *delta(I_Tx,prime(I_R,row));

            T_R[row] = ( ( T_R[row]*tU.mapprime(RLINK,row,row+1) ) 
                *delta(I_SVD_U,I_Fxd)*R ) *delta(I_Tx,prime(I_R,row+1));

            T_R[row].prime(HSLINK);
            std::cout << TAG_T_R <<"["<< row <<"]";
            printfln("= %s",T_R[row]);
        }

        std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    // End of cluster absorption
    C_RU.prime(ULINK,sizeM);
    C_RD.prime(DLINK,sizeM);

    std::cout <<"##### InsLCol Done "<< std::string(53,'#') << std::endl;
}

// ########################################################################
// isometries

std::pair<ITensor,ITensor> CtmEnv::isoT1(IndexSet const& iS_tU, 
    std::pair< Index, Index > const& iS_delta, ITensor const& t1, 
    ITensor const& t2) const {
    /*
     * Compute the "isometry" Z which will be used to reduce
     * contracted tensors to original dimensions
     *
     * We use the following method [DUMB?]
     * 
     * i_E - environment index (I_U, I_R, I_D or I_L) 
     * i_S - aux index of site (I_XH or I_XV)
     * are grouped in arg iS_tU
     *
     * iS_delta holds indices which are contracted through delta tensor
     *        ____                     ____
     *  i_E--|    |                   |    |--i_E?
     *       | t1 |--DELTA(iS_delta)--| t2 |        =
     *  i_S--|____|                   |____|--i_S1  
     *           __                        __
     *     i_E--|  \          _           /  |--I_E?
     * =        |tU --I_SVD--|S|--I_SVD1-- tV|
     *     i_S--|__/                      \__|--I_S1
     *
     */
    // Prepare matrices for SVD - isoPair holds (tU,tV), S holds SVD values
    std::pair< ITensor, ITensor > isoPair = 
       std::make_pair(ITensor(iS_tU), ITensor());
    ITensor S;

    // Perform SVD
    svd(t1*delta( iS_delta.first, iS_delta.second )*t2, isoPair.first,
        S,isoPair.second, {"IndexType",SVD_LINK});
    Print(isoPair.first);
    Print(S);
    Print(isoPair.second);

    return isoPair;
}

std::pair<ITensor,ITensor> CtmEnv::isoT2(IndexSet const& iS_tU,
    ITensor const& t1, ITensor const& t2, 
    ITensor const& t3, ITensor const& t4) const {
    /*
     * Compute the "isometry" Z which will be used to reduce
     * contracted tensors to original dimensions
     * 
     * i_E - environment index (I_U, I_R, I_D or I_L) 
     * i_S - aux index of site (I_XH or I_XV)
     * are grouped in arg iS_tU
     *
     * intermediate indices are contracted through delta(D) tensor
     * if necessary
     *        ____                            ____
     *  i_E--|    |                          |    |--i_E?
     *       | t1 |--D1--|t2|--D2--|t3|--D3--| t4 |       =
     *  i_S--|____|                          |____|--i_S1  
     *           __                        __
     *     i_E--|  \          _           /  |--I_E?
     * =        |tU --I_SVD--|S|--I_SVD1-- tV|
     *     i_S--|__/                      \__|--I_S1
     *
     */
    // Prepare matrices for SVD - isoPair holds (tU,tV), S holds SVD values
    std::pair< ITensor, ITensor > isoPair = 
       std::make_pair(ITensor(iS_tU), ITensor());
    ITensor S;

    // possible delta tensors (or simply scalars of value 1)
    ITensor D1, D2, D3;

    // Find indices of same IndexType on t1 & t2
    for ( auto const& i : t2.inds() ) {
        Index I = findtype(t1, i.type());
        if( I ) {
            if( I.primeLevel() == i.primeLevel() ) {
                D1 = ITensor(1.);
            } else {
                D1 = delta(I,i);
            }
            break;
        }
    }

    // Find indices of same IndexType on t2 & t3
    for ( auto const& i : t3.inds() ) {
        Index I = findtype(t2, i.type());
        if( I ) {
            if( I.primeLevel() == i.primeLevel() ) {
                D2 = ITensor(1.);
            } else {
                D2 = delta(I,i);
            }
            break;
        }
    }

    // Find indices of same IndexType on t3 & t4
    for ( auto const& i : t3.inds() ) {
        Index I = findtype(t4, i.type());
        if( I ) {
            if( I.primeLevel() == i.primeLevel() ) {
                D3 = ITensor(1.);
            } else {
                D3 = delta(I,i);
            }
            break;
        }
    }

    auto T = t1*D1*t2;
    Print(T);
    T = T*D2*t3;
    Print(T);
    T = T*D3*t4;
    Print(T);

    // Perform SVD
    svd(T, isoPair.first,
        S,isoPair.second, {"IndexType",SVD_LINK});
    Print(isoPair.first);
    Print(S);
    Print(isoPair.second);

    return isoPair;
}

std::pair<ITensor,ITensor> CtmEnv::isoT3(
    std::pair< Index, Index > const& iS_Elink,
    std::pair< Index, Index > const& iS_Slink,
    ITensor const& t1, ITensor const& t2) const {
    /*
     * Compute the "isometry" Z which will be used to reduce
     * contracted tensors to original dimensions
     * 
     * iS_Elink - holds pair of indices to define delta tensor
     *      for relabeling iS_Elink.second to iS_Elink.first on t2
     * iS_Slink - holds pair of indices to define delta tensor
     *      for relabeling iS_Slink.second to iS_Slink.first on t2
     *            ____              ____
     *      i_E--|    |            |    |--i_E'
     *           | t1 |--I_[U..L]--| t1*|        +
     *      i_S--|____|            |____|--i_S'  
     *                   ____              ___
     *    D(i_E,i_E~')--|    |            |   |--D(i_E~, i_E')
     * +                |t2* |--I_[U..L]--|t2 |               =
     *    D(i_S,i_S~')--|____|            |___|--D(i_S~, i_S')
     *           __                     ___
     *     i_E--|  \         _         /   |--i_E'--D(i_E', i_E~)
     * =        |tZ --I(D)--|D|--I(D)-- tZ*|
     *     I_S--|__/                   \___|--i_S'--D(i_E', i_S~)
     *
     */

    ITensor T;
    ITensor D_E1_E2p, D_E2_E1p, D_S1_S2p, D_S2_S1p;

    D_E2_E1p = delta( iS_Elink.second,prime(iS_Elink.first) );
    D_S2_S1p = delta( iS_Slink.second,prime(iS_Slink.first) );

    D_E1_E2p = delta( iS_Elink.first,prime(iS_Elink.second) );
    D_S1_S2p = delta( iS_Slink.first,prime(iS_Slink.second) );

    // Prepare Hermitian matrix to be diagonalized
    T = t1*(prime(t1, iS_Elink.first, iS_Slink.first).conj()) + 
        ((prime(t2, iS_Elink.second, iS_Slink.second).conj())
        *D_E1_E2p*D_S1_S2p)*(t2*D_E2_E1p*D_S2_S1p);

    ITensor tZ, Diag;

    diagHermitian(T, tZ, Diag);

    Print(tZ);
    PrintData(Diag);

    auto tZstar = conj(tZ);
    tZstar = tZstar*delta(iS_Elink.first, iS_Elink.second)*
        delta(iS_Slink.first,iS_Slink.second);
    Print(tZstar);

    auto cI = commonIndex(tZ, tZstar);
    tZ *= delta(cI, Index("dummy_diag",cI.m(),SVD_LINK));
    tZstar *= delta(cI, Index("dummy_diag",cI.m(),SVD_LINK));

    return std::make_pair(tZ, tZstar);
}

// ########################################################################
// environment normalization methods

/*
 * TODO implement optimization according to largest dim of cluster
 * TODO more detailed comments about algorithm for contracting TN
 * 
 */
double CtmEnv::getNorm() const {
    auto Norm = C_LU;

    /*
     * Suppose sizeM >= sizeN, contract say left boundary of environment 
     *
     *   |C_LU|----I_U0
     *     |
     *   |T_L0|----I_XH(sizeN-1)*2
     *     |
     *    ...
     *     |
     * |T_LsizeN|--I_XH0
     *     |
     *   |C_LD|----I_D0 
     *
     */
    for ( auto const& t : T_L ) {
        Norm.prime(HSLINK,2);
        Norm *= t;
    }
    Norm *= C_LD;

    for ( int col=0; col<sizeM; col++ ) {
        Norm *= T_D[col];

        /*
         * Contract the cluster+environment column by column
         *
         *   |C_LU|----I_U0
         *     |
         *   |T_L0|----I_XH(sizeN-1)*2
         *     |
         *    ...              I_XV
         *     |                | 
         * |T_LsizeN|--I_XH0--| X  |--I_XH1
         *     |                |
         *     |               I_XV1
         *     |                |
         *   |C_LD|----I_D0---|T_D0|--I_D1 
         *
         */
        for ( int row=sizeN-1; row>=0; row-- ) {
            Norm.mapprime(0,1,VSLINK);
            Norm *= prime(sites.at(cToS.at(std::make_pair(row,col))), HSLINK,
                2*(sizeN-1-row));
        }
        Norm.prime(HSLINK,-1);
        Norm *= T_U[col];

    }

    Norm *= C_RU;
    for ( int row=sizeN-1; row>=0; row-- ) {
        Norm.mapprime(2*row,1,HSLINK);
        Norm *= T_R[row];
    }
    Norm *= C_RD;

    return norm(Norm);
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

    normalizeBLE_T(C_LU);
    normalizeBLE_T(C_RU);
    normalizeBLE_T(C_RD);
    normalizeBLE_T(C_LD);

    for (int i=0; i<sizeM; i++) {
        normalizeBLE_T(T_U[i]);
        normalizeBLE_T(T_D[i]);
    }

    for (int i=0; i<sizeN; i++) {
        normalizeBLE_T(T_L[i]);
        normalizeBLE_T(T_R[i]);
    }
}

/*
 * Normalize the tensors Proportional To (Frobenius) Norm (PTN)
 *
 */
void CtmEnv::normalizePTN() {
    double nrm = CtmEnv::getNorm();
    // Euclidean norms of C_* & T_* matrices
    double nClu = norm(C_LU);
    double nCru = norm(C_RU);
    double nCrd = norm(C_RD);
    double nCld = norm(C_LD);
    std::vector<double> nTu, nTr, nTd, nTl;
    
    double nSum = nClu + nCru + nCrd + nCld;

    for ( int i=0; i<sizeN; i++ ) {
        nTl.push_back(norm(T_L[i]));
        nSum += nTl[i];
        nTr.push_back(norm(T_R[i]));
        nSum += nTr[i];
    }

    for ( int i=0; i<sizeM; i++ ) {
        nTu.push_back(norm(T_U[i]));
        nSum += nTu[i];
        nTd.push_back(norm(T_D[i]));
        nSum += nTd[i];
    }

    C_LU *= std::pow(1.0/nrm, nClu/nSum);
    C_RU *= std::pow(1.0/nrm, nCru/nSum);
    C_RD *= std::pow(1.0/nrm, nCrd/nSum);
    C_LD *= std::pow(1.0/nrm, nCld/nSum);
    
    for ( int i=0; i<sizeN; i++ ) {
        T_R[i] *= std::pow(1.0/nrm, nTr[i]/nSum);
        T_L[i] *= std::pow(1.0/nrm, nTl[i]/nSum);
    }

    for ( int i=0; i<sizeM; i++ ) {
        T_U[i] *= std::pow(1.0/nrm, nTu[i]/nSum);
        T_D[i] *= std::pow(1.0/nrm, nTd[i]/nSum);
    }

}

// void CtmEnv::normalizeCs() {
//     double norm = CtmEnv::getNorm();
//     double sqrt4N = std::pow(norm, 1.0/4.0);
//     C_lu = C_lu /sqrt4N;
//     C_ru = C_ru /sqrt4N;
//     C_rd = C_rd /sqrt4N;
//     C_ld = C_ld /sqrt4N;
// }

// void CtmEnv::normalize() {
//     double norm  = CtmEnv::getNorm();
//     double sqrt8N = std::pow(norm, 1.0/8.0);
//     C_lu = C_lu /sqrt8N;
//     C_ru = C_ru /sqrt8N;
//     C_rd = C_rd /sqrt8N;
//     C_ld = C_ld /sqrt8N;
//     T_u  =  T_u /sqrt8N;
//     T_r  =  T_r /sqrt8N;
//     T_d  =  T_d /sqrt8N;
//     T_l  =  T_l /sqrt8N;
// } 

// void CtmEnv::printENorms(std::string tag) {
//     std::cout << tag <<" "<< norm(C_lu) <<" "<< norm(T_u) <<" "
//         << norm(C_ru) <<" "<< norm(T_r) <<" "<< norm(C_rd) 
//         <<" "<< norm(T_d) <<" "<< norm(C_ld) <<" "<< norm(T_l) <<"\n";
// }

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
    svd(C_LU, U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_clu[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_Um--|C_RU|--I_R
     *
     */
    U = ITensor(prime(I_U,sizeM));
    svd(C_RU, U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_cru[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_Rn--|C_RD|--I_Dm
     *
     */
    U = ITensor(prime(I_R,sizeN));
    svd(C_RD, U, S, V);
    iS = S.inds();
    for(int i=1; i<=x; i++) {
        spec.spec_crd[i-1] = S.real(iS[0](i), iS[1](i));
    }

    /*
     * I_D--|C_LD|--I_Ln
     *
     */
    U = ITensor(I_D);
    svd(C_LD, U, S, V);
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

// Index const& CtmEnv::getIXH() const { return i_Xh; }
// Index const& CtmEnv::getIXV() const { return i_Xv; }

CtmEnv::CtmSpec CtmEnv::getCtmSpec() const {
    return spec;
}

CtmData CtmEnv::getCtmData() const {
    CtmData ctmData = {
        x, d, sizeN, sizeM, 
        T_U, T_R, T_D, T_L,
        C_LU, C_RU, C_RD, C_LD,
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

    // TODO cToS, sites

    s <<"----start CORNER TENSORS----"<< std::endl;
    s << TAG_C_LU;
    printfln(" = %s", C_LU);
    s << TAG_C_RU;
    printfln(" = %s", C_RU);
    s << TAG_C_RD;
    printfln(" = %s", C_RD);
    s << TAG_C_LD;
    printfln(" = %s", C_LD);
    s <<"------end CORNER TENSORS----"<< std::endl;

    s <<"----start HALF-ROW TENSORS--"<< std::endl;
    for( std::size_t i=0; i<T_L.size(); i++ ) {
        s << TAG_T_L << i;
        printfln(" = %s", T_L[i]);
    }
    for( std::size_t i=0; i<T_R.size(); i++ ) {
        s << TAG_T_R << i;
        printfln(" = %s", T_R[i]);
    }
    s <<"------end HALF-ROW TENSORS--"<< std::endl;

    s <<"----start HALF-COL TENSORS--"<< std::endl;
    for( std::size_t i=0; i<T_U.size(); i++ ) {
        s << TAG_T_U << i;
        printfln(" = %s", T_U[i]);
    }
    for( std::size_t i=0; i<T_D.size(); i++ ) {
        s << TAG_T_D << i;
        printfln(" = %s", T_D[i]);
    }
    s <<"------end HALF-COL TENSORS--"<< std::endl;

    return s;
}

std::ostream& operator<<(std::ostream& s, CtmEnv const& c) {
    return c.print(s);
}

std::ostream& operator<<(std::ostream& s, CtmEnv::CtmSpec const& spec) {
    s << "C_lu: ";
    for (std::complex<double> const& sv : spec.spec_clu) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ru: ";
    for (std::complex<double> const& sv : spec.spec_cru) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_rd: ";
    for (std::complex<double> const& sv : spec.spec_crd) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ld: ";
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