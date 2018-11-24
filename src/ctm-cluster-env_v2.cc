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

//CtmEnv::CtmEnv () { solver = SvdSolver() }

CtmEnv::CtmEnv (std::string t_name, int t_x, Cluster const& c, 
    SvdSolver & ssolver, Args const& args) 
    : m_name(t_name), solver(ssolver), 
    d(c.auxBondDim*c.auxBondDim), x(t_x), 
    sizeN(c.sizeN), sizeM(c.sizeM) 
    {

    isoPseudoInvCutoff = args.getReal("isoPseudoInvCutoff",1.0e-14);
    isoMinElemWarning  = args.getReal("isoMinElemWarning",1.0e-4);
    isoMaxElemWarning  = args.getReal("isoMaxElemWarning",1.0e4);
    SVD_METHOD         = args.getString("SVD_METHOD","itensor");
    rsvd_power         = args.getInt("rsvd_power",2);
    rsvd_reortho       = args.getInt("rsvd_reortho",1);
    rsvd_oversampling  = args.getInt("rsvd_oversampling",10);
    DBG     = args.getBool("dbg",false);
    DBG_LVL = args.getInt("dbgLevel",0);

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
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) )
    };
}

/*
 * Assume the whole environment is delivered as a set of simple
 * environments for each of non-equivalent sites within cluster
 * Preserves compatibility with LEGACY ctm-cluster-io.h
 *
 */
CtmEnv::CtmEnv (std::string t_name,  std::vector<CtmData> const& ctmD, 
    Cluster const& c, SvdSolver & ssolver, Args const& args)
    : m_name(t_name), solver(ssolver), d(ctmD[0].auxDimSite), 
    x(ctmD[0].auxDimEnv), sizeN(c.sizeN), sizeM(c.sizeM),
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
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) ),
        std::vector< std::vector<double> >(c.siteIds.size(), std::vector<double>(x) )
    };
}

ITensor CtmEnv::contractOST(ITensor const& T, bool expose) const {
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
    auto auxI = noprime(findtype(T, AUXLINK));

    if(auxI.m()*auxI.m() != d) {
        std::cout <<"ctmEnv.d does not agree with onSiteT.dimD^2"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Define combiner tensors Y*
    auto C04 = combiner(auxI, prime(auxI,4));
    auto C15 = prime(C04,1);
    auto C26 = prime(C04,2);
    auto C37 = prime(C04,3);

    ITensor X;
    if (expose)
        X = (T*( conj(T).prime(AUXLINK,4).prime(PHYS,1) ))*C04*C15*C26*C37;
    else
        X = (T*( conj(T).prime(AUXLINK,4) ))*C04*C15*C26*C37;

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

    auto shift05 = [](double r){ return r-0.5; };

    for ( auto& t : C_LU ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_RU ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_RD ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : C_LD ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_U ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_R ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_D ) { randomize(t, {"Complex", isComplex}); }
    for ( auto& t : T_L ) { randomize(t, {"Complex", isComplex}); }

    // for ( auto& t : C_LU ) { t.apply(shift05); }
    // for ( auto& t : C_RU ) { t.apply(shift05); }
    // for ( auto& t : C_RD ) { t.apply(shift05); }
    // for ( auto& t : C_LD ) { t.apply(shift05); }
    // for ( auto& t : T_U ) { t.apply(shift05); }
    // for ( auto& t : T_R ) { t.apply(shift05); }
    // for ( auto& t : T_D ) { t.apply(shift05); }
    // for ( auto& t : T_L ) { t.apply(shift05); }

    normalizePTN();

    computeSVDspec();

    std::cout <<"INIT_ENV_rnd with all C's and T's random (complex ? " 
        << isComplex <<")"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

void CtmEnv::initCtmrgEnv(bool dbg) {
    std::cout <<"===== INIT_ENV_ctmrg called "<< std::string(44,'=') 
        << std::endl;
    
    //Define "contractor" tensor
    int D = round(sqrt(d));
    auto cI = Index("C",d);
    auto CT = ITensor(cI);
    for ( int i=1; i<= D; i++ ) {
        CT.set(cI(i+D*(i-1)),1.0);
    }
    // OBC (?)
    // for ( int i=1; i<= D*D; i++ ) {
    //     CT.set(cI(i),1.0);
    // }

    if(dbg) PrintData(CT);

    for ( size_t i=0; i<sites.size(); i++ ) {
        if(dbg) std::cout <<"----- generating init env for site "<< siteIds[i]
            <<" -----"<< std::endl;
        // Locate the first appearance of given site within cluster
        int row, col;
        for ( const auto& cToSEntry : cToS ) {
            if ( cToSEntry.second == i) {
                col = cToSEntry.first.first;
                row = cToSEntry.first.second;
                break;
            }
        }
        if(dbg) std::cout <<"Found "<< siteIds[i] <<" at ["<< col <<","<< row
            <<"]"<< std::endl;

        //Construct corner matrices
        std::pair<int,int> site = 
            std::make_pair((col-1+sizeM)%sizeM,(row-1+sizeN)%sizeN);
        C_LU[i] = ( ( ( sites[ cToS.at(site) ]
            *(CT*delta(cI,I_XH)) )*(CT*delta(cI,I_XV)) )
            *delta( prime(I_XH,1), I_U ) )
            *delta( prime(I_XV,1), I_L);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LU;
        printfln(" = %s", C_LU[i]); }
        
        site = 
            std::make_pair((col+1)%sizeM,(row-1+sizeN)%sizeN);
        C_RU[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,I_XV)) )
            *delta( I_XH, prime(I_U,1) ) )
            *delta( prime(I_XV,1), I_R);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RU;
        printfln(" = %s", C_RU[i]); }

        site = 
            std::make_pair((col+1)%sizeM,(row+1)%sizeN);
        C_RD[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,prime(I_XV,1))) )
            *delta( I_XH, prime(I_D,1) ) )
            *delta( I_XV, prime(I_R,1));
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RD;
        printfln(" = %s", C_RD[i]); }

        site = 
            std::make_pair((col-1+sizeM)%sizeM,(row+1)%sizeN);
        C_LD[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,I_XH)) )*(CT*delta(cI,prime(I_XV,1))) )
            *delta( prime(I_XH,1), I_D ) )
            *delta( I_XV, prime(I_L,1));
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LD;
        printfln(" = %s", C_LD[i]); }
    
        //Construct half-row/col matrices
        site = 
            std::make_pair(col,(row-1+sizeN)%sizeN);
        T_U[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XV)) )
            *delta(I_XH, I_U) )*delta(prime(I_XH,1), prime(I_U,1));
        T_U[i].prime(VSLINK,-1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_U;
        printfln(" = %s", T_U[i]); }

        site = 
            std::make_pair((col+1)%sizeM,row);
        T_R[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XH,1))) )
            *delta(I_XV, I_R) )*delta(prime(I_XV,1), prime(I_R,1));
        T_R[i].prime(I_XH,1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_R;
        printfln(" = %s", T_R[i]); }

        site = 
            std::make_pair(col,(row+1)%sizeN);
        T_D[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XV,1))) )
            *delta(I_XH, I_D) )*delta(prime(I_XH,1), prime(I_D,1));
        T_D[i].prime(I_XV,1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_D;
        printfln(" = %s", T_D[i]); }

        site = 
            std::make_pair((col-1+sizeM)%sizeM,row);
        T_L[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XH)) )
            *delta(I_XV, I_L) )*delta(prime(I_XV,1), prime(I_L,1));
        T_L[i].prime(HSLINK,-1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_L;
        printfln(" = %s", T_L[i]); }
    }

    //    normalizePTN();

    computeSVDspec();

    std::cout <<"===== INIT_ENV_ctmrg done "<< std::string(46,'=') 
        << std::endl; 
}

void CtmEnv::initOBCEnv(bool dbg) {
    std::cout <<"===== INIT_ENV_obc called "<< std::string(44,'=') 
        << std::endl;
    
    //Define "contractor" tensor
    int D = round(sqrt(d));
    auto cI = Index("C",d);
    auto CT = ITensor(cI);
    for ( int i=1; i<= D*D; i++ ) {
        CT.set(cI(i),1.0);
    }
    // OBC (?)
    // for ( int i=1; i<= D*D; i++ ) {
    //     CT.set(cI(i),1.0);
    // }

    if(dbg) PrintData(CT);

    for ( size_t i=0; i<sites.size(); i++ ) {
        if(dbg) std::cout <<"----- generating init env for site "<< siteIds[i]
            <<" -----"<< std::endl;
        // Locate the first appearance of given site within cluster
        int row, col;
        for ( const auto& cToSEntry : cToS ) {
            if ( cToSEntry.second == i) {
                col = cToSEntry.first.first;
                row = cToSEntry.first.second;
                break;
            }
        }
        if(dbg) std::cout <<"Found "<< siteIds[i] <<" at ["<< col <<","<< row
            <<"]"<< std::endl;

        //Construct corner matrices
        std::pair<int,int> site = 
            std::make_pair((col-1+sizeM)%sizeM,(row-1+sizeN)%sizeN);
        C_LU[i] = ( ( ( sites[ cToS.at(site) ]
            *(CT*delta(cI,I_XH)) )*(CT*delta(cI,I_XV)) )
            *delta( prime(I_XH,1), I_U ) )
            *delta( prime(I_XV,1), I_L);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LU;
        printfln(" = %s", C_LU[i]); }
        
        site = 
            std::make_pair((col+1)%sizeM,(row-1+sizeN)%sizeN);
        C_RU[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,I_XV)) )
            *delta( I_XH, prime(I_U,1) ) )
            *delta( prime(I_XV,1), I_R);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RU;
        printfln(" = %s", C_RU[i]); }

        site = 
            std::make_pair((col+1)%sizeM,(row+1)%sizeN);
        C_RD[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,prime(I_XH,1))) )*(CT*delta(cI,prime(I_XV,1))) )
            *delta( I_XH, prime(I_D,1) ) )
            *delta( I_XV, prime(I_R,1));
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_RD;
        printfln(" = %s", C_RD[i]); }

        site = 
            std::make_pair((col-1+sizeM)%sizeM,(row+1)%sizeN);
        C_LD[i] = ( ( ( sites[ cToS.at(site)]
            *(CT*delta(cI,I_XH)) )*(CT*delta(cI,prime(I_XV,1))) )
            *delta( prime(I_XH,1), I_D ) )
            *delta( I_XV, prime(I_L,1));
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_C_LD;
        printfln(" = %s", C_LD[i]); }
    
        //Construct half-row/col matrices
        site = 
            std::make_pair(col,(row-1+sizeN)%sizeN);
        T_U[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XV)) )
            *delta(I_XH, I_U) )*delta(prime(I_XH,1), prime(I_U,1));
        T_U[i].prime(VSLINK,-1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_U;
        printfln(" = %s", T_U[i]); }

        site = 
            std::make_pair((col+1)%sizeM,row);
        T_R[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XH,1))) )
            *delta(I_XV, I_R) )*delta(prime(I_XV,1), prime(I_R,1));
        T_R[i].prime(I_XH,1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_R;
        printfln(" = %s", T_R[i]); }

        site = 
            std::make_pair(col,(row+1)%sizeN);
        T_D[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,prime(I_XV,1))) )
            *delta(I_XH, I_D) )*delta(prime(I_XH,1), prime(I_D,1));
        T_D[i].prime(I_XV,1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_D;
        printfln(" = %s", T_D[i]); }

        site = 
            std::make_pair((col-1+sizeM)%sizeM,row);
        T_L[i] = (( sites[ cToS.at(site) ] * (CT*delta(cI,I_XH)) )
            *delta(I_XV, I_L) )*delta(prime(I_XV,1), prime(I_L,1));
        T_L[i].prime(HSLINK,-1);
        if(dbg) { std::cout << siteIds[cToS.at(site)] <<" -> "<< TAG_T_L;
        printfln(" = %s", T_L[i]); }
    }

    //    normalizePTN();

    computeSVDspec();

    std::cout <<"===== INIT_ENV_obc done "<< std::string(46,'=') 
        << std::endl; 
}

void CtmEnv::initPWREnv(bool dbg) {
    double pwr = 1.0;

    for ( auto& t : C_LU ) { for (int i=1; i<=x; i++) t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double) i,-pwr)); }
    for ( auto& t : C_RU ) { for (int i=1; i<=x; i++) t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double) i,-pwr)); }
    for ( auto& t : C_RD ) { for (int i=1; i<=x; i++) t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double) i,-pwr)); }
    for ( auto& t : C_LD ) { for (int i=1; i<=x; i++) t.set(t.inds().front()(i), t.inds().back()(i), std::pow((double) i,-pwr)); }
    for ( auto& t : T_U ) { 
        for (int i=1; i<=x; i++) {
            for (int j=1; j<=d; j++) { t.set(I_U(i), prime(I_U,1)(i), I_XV(j), 1.0); } 
        }
    }
    for ( auto& t : T_R ) { 
        for (int i=1; i<=x; i++) {
            for (int j=1; j<=d; j++) { t.set(I_R(i), prime(I_R,1)(i), prime(I_XH,1)(j), 1.0); } 
        } 
    }
    for ( auto& t : T_D ) { 
        for (int i=1; i<=x; i++) {
            for (int j=1; j<=d; j++) { t.set(I_D(i), prime(I_D,1)(i), prime(I_XV,1)(j), 1.0); } 
        }
    }
    for ( auto& t : T_L ) { 
        for (int i=1; i<=x; i++) {
            for (int j=1; j<=d; j++) { t.set(I_L(i), prime(I_L,1)(i), I_XH(j), 1.0); } 
        } 
    }

    // normalizePTN();

    computeSVDspec();

    std::cout <<"INIT_ENV_pwr called"<< std::endl;
    std::cout << std::string(72,'=') << std::endl;
}

void CtmEnv::symmetrizeEnv(bool dbg) {
 
    //Define "contractor" tensor
    int D = round(sqrt(d));
    auto cI  = Index("C",d);
    auto cIp = prime(cI);
    auto CT = ITensor(cI,cIp);
    for ( int i=0; i<D; i++ ) {
        for ( int j=0; j<D; j++ ) {
            CT.set(cI(1+i*D),cIp(1+i+j*D),1.0);
        }
    }

    for ( size_t i=0; i<sites.size(); i++ ) {
        if(dbg) std::cout <<"----- symmetrizing T's for site "<< siteIds[i]
            <<" -----"<< std::endl;
        // Locate the first appearance of given site within cluster
        int row, col;
        for ( const auto& cToSEntry : cToS ) {
            if ( cToSEntry.second == i) {
                col = cToSEntry.first.first;
                row = cToSEntry.first.second;
                break;
            }
        }
        if(dbg) std::cout <<"Found "<< siteIds[i] <<" at ["<< col <<","<< row
            <<"]"<< std::endl;

        //Construct half-row/col matrices
        T_U[i] = 0.5*( T_U[i] + (T_U[i]*(delta(cI,I_XV)*CT))*delta(cIp,I_XV) );
        if(dbg) { printfln(" = %s", T_U[i]); }

        T_R[i] = 0.5*( T_R[i] + (T_R[i]*(delta(cI,prime(I_XH))*CT))*delta(cIp,prime(I_XH)) );
        if(dbg) { printfln(" = %s", T_R[i]); }

        T_D[i] = 0.5*( T_D[i] + (T_D[i]*(delta(cI,prime(I_XV))*CT))*delta(cIp,prime(I_XV)) );
        if(dbg) { printfln(" = %s", T_D[i]); }

        T_L[i] = 0.5*( T_L[i] + (T_L[i]*(delta(cI,I_XH)*CT))*delta(cIp,I_XH) );
        if(dbg) { printfln(" = %s", T_L[i]); }
    }
}

void CtmEnv::testCtmrgEnv() {
    
    double m = 0.;
    auto max_m = [&m](double d) {
        if(std::abs(d) > m) m = std::abs(d);
    };

    ITensor temp;
    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = C_LU[a] - C_LU[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"C_LU["<<a<<"]-C_LU["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = C_RU[a] - C_RU[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"C_RU["<<a<<"]-C_RU["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(C_LU[0]);
    Print(C_RU[0]);
    ITensor ttemp = (C_RU[0] * delta(I_R, I_L))
        * delta(prime(I_U,1),I_U);
    temp = C_LU[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = C_RD[a] - C_RD[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"C_RD["<<a<<"]-C_RD["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(C_LU[0]);
    Print(C_RD[0]);
    ttemp = (C_RD[0] * delta(prime(I_R,1), I_L))
        * delta(prime(I_D,1), I_U);
    temp = C_LU[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = C_LD[a] - C_LD[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"C_LD["<<a<<"]-C_LD["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(C_LU[0]);
    Print(C_LD[0]);
    ttemp = (C_LD[0] * delta(prime(I_L,1), I_L))
        * delta(I_D, I_U);
    temp = C_LU[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = T_L[a] - T_L[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"T_L["<<a<<"]-T_L["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = T_U[a] - T_U[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"T_U["<<a<<"]-T_U["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(T_L[0]);
    Print(T_U[0]);
    ttemp = ((T_U[0] * delta(I_U, I_L))
        * delta(prime(I_U,1), prime(I_L,1))) 
        * delta(I_XV, I_XH);
    temp = T_L[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = T_R[a] - T_R[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"T_R["<<a<<"]-T_R["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(T_L[0]);
    Print(T_R[0]);
    ttemp = ((T_R[0] * delta(I_R, I_L))
        * delta(prime(I_R,1), prime(I_L,1))) 
        * delta(prime(I_XH,1), I_XH);
    temp = T_L[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;

    for (int a=0; a<sites.size(); a++) {
        for (int b=a+1; b<sites.size(); b++) {
            temp = T_D[a] - T_D[b];
            m = 0.;
            temp.visit(max_m);
            std::cout<<"T_D["<<a<<"]-T_D["<<b<<"] Max element: "<< m <<std::endl;
        }
    }

    Print(T_L[0]);
    Print(T_D[0]);
    ttemp = ((T_D[0] * delta(I_D, I_L))
        * delta(prime(I_D,1), prime(I_L,1))) 
        * delta(prime(I_XV,1), I_XH);
    temp = T_L[0] - ttemp;
    m = 0.;
    temp.visit(max_m);
    std::cout<<"Max element: "<< m <<std::endl;
}

// ########################################################################
// CTM iterative methods

void CtmEnv::insURow_DBG(CtmEnv::ISOMETRY iso_type, 
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT, bool dbg) 
{
    if(dbg) std::cout <<"##### InsURow called "<< std::string(51,'#') << std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;

    // sequentialy contract upper boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*
    for (int row=0; row<sizeN; row++) {

        if(dbg) std::cout <<"(1) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_U0--|             \
         *          |Z[0..sizeM-1] |--I_UsizeM+10
         *   I_XH0--|_____________/
         *
         */
        t_iso_begin = std::chrono::steady_clock::now(); 

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('U', -1, row);
                break;
            }
            case ISOMETRY_T2: {
                isoZ = isoT2('U', -1, row, accT);
                break;
            }
            case ISOMETRY_T3: {
                isoZ = isoT3('U', -1, row, accT);
                break;
            }
            case ISOMETRY_T4: {
                isoZ = isoT4('U', -1, row, accT);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        for (int col=0; col<sizeM; col++) {
            /*
             * Absorb one row of network up, leading to new (unreduced) 
             * environment tensors for sites one row down (+1 in Y coord)
             *   _______               ______                _________
             *  |C_LU_00|--I_U   I_U--|T_U_x0|--I_U1  I_U1--|C_RU_m-10|
             *   _|____                 |                    _|______
             *  |T_L_00|--I_XH  I_XH--|X_x0|--I_XH1  I_XH1--|T_R_m-10|   
             *    |                     |                     | 
             *   I_L1>>I_L            I_XV1>>I_XV           I_R1>>I_R
             *
             *    V V V                V V V V V                  V V V
             *   _______                ______                  _________
             *  |       |--I_U    I_U--|      |--I_U1    I_U1--|         |
             *  |C_LU_01|              |T_U_x1|                |C_RU_m-11|
             *  |_______|--I_XH  I_XH--|______|--I_XH1  I_XH1--|_________|
             *     |                      |                        |
             *    I_L                    I_XV                     I_R
             *
             */
            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(2."<< col <<".1) ----- C_LU & T_L ["<<
                col <<","<< row <<"] -----"<< std::endl;
            auto tC1 = C_LU.at( cToS.at(std::make_pair(col,row)) ) * 
                T_L.at( cToS.at(std::make_pair(col,row)) );
            tC1.prime(LLINK, -1);
            if(dbg) Print(tC1);

            if(dbg) std::cout <<"(2."<< col <<".2) ----- T_U & X ["<<
                col <<","<< row <<"] -----"<< std::endl;
            // auto tT1 = ( T_U.at( cToS.at(std::make_pair(col,row)) ) * 
            //     sites[cToS[std::make_pair(col,row)]] );
            // tT1.prime(VSLINK, -1);
            //if(dbg) printfln("= %s", tT1);
            
            if(dbg) std::cout <<"(2."<< col <<".3) ----- C_RU & T_R ["<<
                col <<","<< row <<"] -----"<< std::endl;
            auto tC2 = C_RU.at( cToS.at(std::make_pair(col,row)) ) * 
                T_R.at( cToS.at(std::make_pair(col,row)) );
            tC2.prime(RLINK, -1);
            if(dbg) Print(tC2);
        
            t_iso_end = std::chrono::steady_clock::now();
            accT[1] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(3."<< col <<
                ".1) ----- Construct reduced C_LU -----"<<" isoZ["<< (2*col)%(2*sizeM) 
                <<"]"<< std::endl;
            /*   _______                           ________
             *  |       |--I_U              I_U0--|        \
             *  |C_LU_01|       *contract*        |ZsizeM-1 --I_UsizeM+10>>I_U0
             *  |_______|--I_XH            I_XH0--|________/
             *     |
             *    I_L
             *
             */
            C_LU.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) ) = 
                tC1 * isoZ[(2*col)%(2*sizeM)];
            C_LU.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) )
                .mapprime(ULINK,sizeM+10,0);
            if(dbg) { std::cout << TAG_C_LU <<"["<< col <<","<< (row+1)%sizeN <<"]";
            printfln("= %s", 
                C_LU.at(cToS.at(std::make_pair(col,(row+1)%sizeN))) ); }

            /*        ______
             *  I_U--|      |--I_U1             I_U0--|Zc-1\__I_UsizeM+10>>I_U0 
             *       |T_U_c1|       *contract* I_XH0--|dag /
             * I_XH--|__ ___|--I_XH1   
             *          |                    I_U1--|Zc  \__I_UsizeM+10>>I_U1
             *         I_XV                 I_XH1--|    /
             *
             */
            if(dbg) std::cout <<"(3."<< col <<".2) ----- REDUCE T_U["<< col <<","
                << (row+1)%sizeN <<"] ----- isoZ["<< (2*col+1)%(2*sizeM) <<"] & "
                <<"isoZ["<< (2*col+2)%(2*sizeM) <<"]"<<std::endl;
            
            auto tT1 = T_U.at( cToS.at(std::make_pair(col,row)) )*isoZ[(2*col+1)%(2*sizeM)];
            tT1 = (tT1 * sites[cToS[std::make_pair(col,row)]]).prime(VSLINK, -1);
            tT1.mapprime(ULINK,sizeM+10,0);

            T_U.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) ) = 
                tT1 * isoZ[(2*col+2)%(2*sizeM)].prime();
            T_U.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) )
                .mapprime(ULINK,sizeM+10+1,1);

            isoZ[(2*col+2)%(2*sizeM)].prime(-1);

            if(dbg) { std::cout << TAG_T_U <<"["<< col <<","<< (row+1)%sizeN <<"]";
            printfln("= %s", T_U.at( 
               cToS.at(std::make_pair(col,(row+1)%sizeN))) ); }
        
            if(dbg) std::cout <<"(3."<< col <<".3) ----- Construct reduced C_RU -----"
                <<"isoZ["<< (2*col+3)%(2*sizeM) <<"]"<< std::endl;
            /*         _______                           ________
             *  I_U1--|       |             I_U1<<I_U0--|        \
             *        |C_RU_c1| *contract*              |ZsizeM-1 --I_UsizeM+10
             * I_XH1--|_______|           I_XH1<<I_XH0--|_dagger_/      >>I_U1
             *            |
             *           I_R
             *
             */
            C_RU.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) ) = 
                tC2 * isoZ[(2*col+3)%(2*sizeM)].prime();
            C_RU.at( cToS.at(std::make_pair(col,(row+1)%sizeN)) )
                .mapprime(ULINK,sizeM+10+1,1);

            isoZ[(2*col+3)%(2*sizeM)].prime(-1);

            t_iso_end = std::chrono::steady_clock::now();
            accT[2] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            if(dbg) { std::cout << TAG_C_RU <<"["<< col <<","<< (row+1)%sizeN <<"]";
            printfln("= %s", C_RU.at( 
               cToS.at(std::make_pair(col,(row+1)%sizeN))) ); }
        }
        
        if(dbg) std::cout <<"(4) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        
        t_iso_begin = std::chrono::steady_clock::now();
        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('U', -1, (row+1)%sizeN);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('U', -1, (row+1)%sizeN);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        if(dbg) std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    if(dbg) std::cout <<"##### InsURow Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insDRow_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT, bool dbg) 
{
    if(dbg) std::cout <<"##### InsDRow called "<< std::string(51,'#') << std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    // sequentialy contract lower boundary of environment with 
    // sizeN rows of cluster + half-row matrices T_L* and T_R*

    for (int row=sizeN-1; row>=0; row--) {

        if(dbg) std::cout <<"(1) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_D0--|             \
         *          |Z[0..sizeM-1] |--I_DsizeM+10
         *   I_XH0--|_____________/
         *
         */

        t_iso_begin = std::chrono::steady_clock::now();

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('D', -1, row);
                break;
            }
            case ISOMETRY_T2: {
                isoZ = isoT2('D', -1, row, accT);
                break;
            }
            case ISOMETRY_T3: {
                isoZ = isoT3('D', -1, row, accT);
                break;
            }
            case ISOMETRY_T4: {
                isoZ = isoT4('D', -1, row, accT);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        for (int col=0; col<sizeM; col++) {
            /*
             * Absorb one row of network up, leading to new (unreduced) 
             * environment tensors for sites one row down (+1 in Y coord)
             *
             *   I_L0>>I_L1              I_XV0>>I_XV1            I_R0>>I_R1
             *    |                       |                       | 
             *  |T_L_0n-1|--I_XH  I_XH--|X_0n-1|--I_XH1  I_XH1--|T_R_0n-1|
             *    |                       |                       | 
             *  |C_LD_0n-1|--I_D   I_D--|T_D_0n-1|--I_D1  I_D1--|C_RD_0n-1| 
             *
             *    V V V                  V V V V V                 V V V
             *
             *     I_L1                    I_XV1                      I_R1
             *   ___|_____                ___|____                  ___|_____
             *  |         |--I_D    I_D--|        |--I_D1    I_D1--|         |
             *  |C_LD_0n-1|              |T_D_0n-1|                |C_RD_0n-1|
             *  |_________|--I_XH  I_XH--|________|--I_XH1  I_XH1--|_________|
             *
             */
            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(2."<< col <<".1) ----- C_LD & T_L ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC4 = C_LD.at( cToS.at(std::make_pair(col,row)) ) * 
                T_L.at( cToS.at(std::make_pair(col,row)) );
            tC4.prime(LLINK);
            if(dbg) Print(tC4);

            if(dbg) std::cout <<"(2."<< col <<".2) ----- T_D & X ["<< col <<","<< row <<
                "] -----"<< std::endl;
            // auto tT3 = ( T_D.at( cToS.at(std::make_pair(col,row)) ) * 
            //     sites[cToS[std::make_pair(col,row)]] );
            // tT3.prime(VSLINK);
            // if(dbg) printfln("= %s", tT3);

            if(dbg) std::cout <<"(2."<< col <<".3) ----- C_RD & T_R ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC3 = C_RD.at( cToS.at(std::make_pair(col,row)) ) * 
                T_R.at( cToS.at(std::make_pair(col,row)) );
            tC3.prime(RLINK);
            if(dbg) Print(tC3);

            t_iso_end = std::chrono::steady_clock::now();
            accT[1] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            t_iso_begin = std::chrono::steady_clock::now(); 

            if(dbg) std::cout <<"(3."<< col <<".1) ----- Construct reduced C_LD -----"
                <<"isoZ["<< (2*col)%(2*sizeM) <<"]"<< std::endl;
            /*
             *    I_L1
             *   __|______                         ________
             *  |         |--I_D            I_D0--|        \
             *  |C_LD_0n-2|      *contract*       |ZsizeM-1 --I_DsizeM+10>>I_D0
             *  |_________|--I_XH          I_XH0--|________/
             *
             */
            C_LD.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) ) = 
                tC4 * isoZ[(2*col)%(2*sizeM)];
            C_LD.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) )
                .mapprime(DLINK,sizeM+10,0);
            if(dbg) { std::cout << TAG_C_LD <<"["<< col <<","<< (row-1+sizeN)%sizeN <<"]";
            printfln("= %s", 
               C_LD.at(cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN))) );

            std::cout <<"(3."<< col <<".2) ----- REDUCE T_D["<< col <<","
            << (row-1+sizeN)%sizeN <<"] ----- isoZ["<< (2*col+1)%(2*sizeM) <<"] & "
                <<"isoZ["<< (2*col+2)%(2*sizeM) <<"]"<<std::endl; }
            /*
             *          I_XV1
             *        ___|____
             *  I_D--|        |--I_D1            I_D0--|Zc-1\__I_DsizeM+10>>I_D0 
             *       |T_D_cn-1|      *contract* I_XH0--|dag /
             * I_XH--|________|--I_XH1   
             *                               I_D1--|Zc  \__I_DsizeM+10>>I_D1
             *                              I_XH1--|    /
             */

            // reset primes
            auto tT3 = T_D.at( cToS.at(std::make_pair(col,row)) )*isoZ[(2*col+1)%(2*sizeM)];
            tT3 = (tT3 * sites[cToS[std::make_pair(col,row)]] ).prime(VSLINK);
            tT3.mapprime(DLINK,sizeM+10,0);

            T_D.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) ) = 
                tT3 * isoZ[(2*col+2)%(2*sizeM)].prime();
            T_D.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) )
                .mapprime(DLINK,sizeM+10+1,1);

            isoZ[(2*col+2)%(2*sizeM)].prime(-1);

            if(dbg) { std::cout << TAG_T_D <<"["<< col <<","<< (row-1+sizeN)%sizeN <<"]";
            printfln("= %s", T_D.at( 
               cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN))) );

            std::cout <<"(3."<< col <<".3) ----- Construct reduced C_LD -----"
                <<" isoZ["<< (2*col+3)%(2*sizeM) <<"]"<< std::endl; }
            /*
             *            I_R1
             *         ____|____                          ________
             *  I_D1--|         |            I_D1<<I_D0--|        \
             *        |C_RD_0n-2| *contract*             |ZsizeM-1 --I_DsizeM+10
             * I_XH1--|_________|          I_XH1<<I_XH0--|_dagger_/     >>I_D1
             *
             */
            C_RD.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) ) = 
                tC3 * isoZ[(2*col+3)%(2*sizeM)].prime();
            C_RD.at( cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN)) )
                .mapprime(DLINK,sizeM+10+1,1);

            isoZ[(2*col+3)%(2*sizeM)].prime(-1);

            t_iso_end = std::chrono::steady_clock::now();
            accT[2] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            if(dbg) { std::cout << TAG_C_RD <<"["<< col <<","<< (row-1+sizeN)%sizeN <<"]";
            printfln("= %s", C_RD.at( 
               cToS.at(std::make_pair(col,(row-1+sizeN)%sizeN))) ); }

        }

        if(dbg) std::cout <<"(6) ----- NORMALIZE "<< std::string(47,'-') << std::endl;
        
        t_iso_begin = std::chrono::steady_clock::now();

        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('D', -1, (row-1+sizeN)%sizeN);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('D', -1, (row-1+sizeN)%sizeN);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        if(dbg) std::cout <<"Row "<< row <<" done"<< std::endl;
    }

    if(dbg) std::cout <<"##### InsDRow Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insLCol_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT, bool dbg)
{
    if(dbg) std::cout <<"##### InsLCol called "<< std::string(51,'#') << std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*
    
    for (int col=0; col<sizeM; col++) {

        if(dbg) std::cout <<"(1) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_L0--|             \
         *          |Z[0..sizeN-1] |--I_LsizeN+10
         *   I_XV0--|_____________/
         *
         */

        t_iso_begin = std::chrono::steady_clock::now();

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('L', col, -1);
                break;
            }
            case ISOMETRY_T2: {
                isoZ = isoT2('L', col, -1, accT, dbg);
                break;
            }
            case ISOMETRY_T3: {
                isoZ = isoT3('L', col, -1, accT, dbg);
                break;
            }
            case ISOMETRY_T4: {
                isoZ = isoT4('L', col, -1, accT, dbg);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        for (int row=0; row<sizeN; row++) {
            /*
             * Absorb one column of network to the left, leading to new 
             * (unreduced) environment tensors for sites in column 
             * to the right (+1 in X coord)
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
             *
             *   I_L1       I_XV1               I_L1   I_XV1 
             *   _|_____     |                   |     |
             *  |C_LD_00|--|T_D_00|--I_D1 ==> |C_LD_10|--I_D1>>I_D
             *
             */
            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(2."<< row <<".1) ----- C_LU & T_U ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC1 = C_LU.at( cToS.at(std::make_pair(col,row)) ) * 
                T_U.at( cToS.at(std::make_pair(col,row)) );
            tC1.prime(ULINK, -1);
            if(dbg) Print(tC1);

            if(dbg) std::cout <<"(2."<< row <<".2) ----- T_L & X ["<< col <<","
                << row <<"] -----"<< std::endl;
            // auto tT4 = ( T_L.at( cToS.at(std::make_pair(col,row)) ) * 
            //     sites[cToS[std::make_pair(col,row)]] );
            // tT4.prime(HSLINK, -1);
            // if(dbg) printfln("= %s", tT4);

            if(dbg) std::cout <<"(2."<< row <<".3) ----- C_LD & T_D ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC4 = C_LD.at( cToS.at(std::make_pair(col,row)) ) * 
                T_D.at( cToS.at(std::make_pair(col,row)) );
            tC4.prime(DLINK, -1);
            if(dbg) Print(tC4);

            t_iso_end = std::chrono::steady_clock::now();
            accT[1] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(3."<< row <<".1) ----- Construct reduced C_LU -----"
                <<" isoZ["<< (2*row)%(2*sizeN) <<"]"<< std::endl;
            /*                                    ________
             * |C_LD_10|--I_U              I_L0--|        \
             *  |    |         *contract*        |ZsizeN-1 --I_LsizeN+10>>I_L0
             * I_L0  I_XV0                I_XV0--|________/
             *
             */
            C_LU.at( cToS.at(std::make_pair((col+1)%sizeM,row)) ) = 
                tC1 * isoZ[(2*row)%(2*sizeN)];
            C_LU.at( cToS.at(std::make_pair((col+1)%sizeM,row)) )
                .mapprime(LLINK,sizeN+10,0);
            if(dbg) { std::cout << TAG_C_LU <<"["<< (col+1)%sizeM <<","<< row <<"]";
            printfln("= %s", 
                C_LU.at(cToS.at(std::make_pair((col+1)%sizeM,row))) ); }

            /* 
             * I_L0 I_XV0                  I_L0--|Zr-1\__I_LsizeN+10>>I_L0 
             *  |    |                    I_XV0--|dag /
             * |T_L_10|--I_XH  *contract*  
             *  |    |                     I_L1--|Zr  \__I_LsizeN+10>>I_L1
             * I_L1 I_XV1                 I_XV1--|    /
             *
             */
            if(dbg) std::cout <<"(3."<< row <<".2) ----- REDUCE T_L ["<< (col+1)%sizeM 
                <<","<< row <<"] ----- isoZ["<< (2*row+1)%(2*sizeN) <<"] & "
                <<" isoZ["<< (2*row+2)%(2*sizeN) <<"]"<< std::endl;
            
            auto tT4 = T_L.at(cToS.at(std::make_pair(col,row))) *isoZ[(2*row+1)%(2*sizeN)]; 
            tT4 = (tT4 * sites[cToS[std::make_pair(col,row)]] ).prime(HSLINK, -1);
            tT4.mapprime(LLINK,sizeN+10,0);

            T_L.at( cToS.at(std::make_pair((col+1)%sizeM,row)) ) = 
                tT4 * isoZ[(2*row+2)%(2*sizeN)].prime();
            T_L.at( cToS.at(std::make_pair((col+1)%sizeM,row)) )
                .mapprime(LLINK,sizeN+10+1,1);

            isoZ[(2*row+2)%(2*sizeN)].prime(-1);

            if(dbg) { std::cout << TAG_T_L <<"["<< (col+1)%sizeM <<","<< row <<"]";
            printfln("= %s", T_L.at( 
                cToS.at(std::make_pair((col+1)%sizeM,row))) );

            std::cout <<"(3."<< row <<".3) ----- Construct reduced C_LD -----"
                <<" isoZ["<< (2*row+3)%(2*sizeN) <<"]"<< std::endl; }
            /*                                           ________
             * I_L1  I_XV1                  I_L1<<I_L0--|        \
             *  |    |         *contract*               |ZsizeN-1 --I_LsizeN+10
             * |C_LD_10|--I_D             I_XV1<<I_XV0--|_dagger_/      >>I_L1
             *
             */
            C_LD.at( cToS.at(std::make_pair((col+1)%sizeM,row)) ) = 
                tC4 * isoZ[(2*row+3)%(2*sizeN)].prime();
            C_LD.at( cToS.at(std::make_pair((col+1)%sizeM,row)) )
                .mapprime(LLINK,sizeN+10+1,1);

            isoZ[(2*row+3)%(2*sizeN)].prime(-1);

            t_iso_end = std::chrono::steady_clock::now();
            accT[2] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            if(dbg) { std::cout << TAG_C_LD <<"["<< (col+1)%sizeM <<","<< row <<"]";
            printfln("= %s", C_LD.at( 
               cToS.at(std::make_pair((col+1)%sizeM,row))) ); }
        }

        if(dbg) std::cout <<"(4) ----- NORMALIZE "<< std::string(47,'-') << std::endl;

        t_iso_begin = std::chrono::steady_clock::now();

        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('L', (col+1)%sizeM, -1);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('L', (col+1)%sizeM, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        if(dbg) std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    if(dbg) std::cout <<"##### InsLCol Done "<< std::string(53,'#') << std::endl;
}

void CtmEnv::insRCol_DBG(CtmEnv::ISOMETRY iso_type,
    CtmEnv::NORMALIZATION norm_type, std::vector<double> & accT, bool dbg) 
{
    if(dbg) std::cout <<"##### InsRCol called "<< std::string(51,'#') << std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    // sequentialy contract left boundary of environment with 
    // sizeM rows of cluster + half-row matrices T_U* and T_D*

    for (int col=sizeM-1; col>=0; col--) {
        
        if(dbg) std::cout <<"(1) ----- Computing Isometry -----"<< std::endl;
        /*
         * Obtain the set of isometries with index format
         *           _____________
         *    I_R0--|             \
         *          |Z[0..sizeN-1] |--I_RsizeN+10
         *   I_XV0--|_____________/
         *
         */

        t_iso_begin = std::chrono::steady_clock::now();

        std::vector<ITensor> isoZ;
        switch(iso_type) {
            case ISOMETRY_T1: {
                isoZ = isoT1('R', col, -1);
                break;
            }
            case ISOMETRY_T2: {
                isoZ = isoT2('R', col, -1, accT, dbg);
                break;
            }
            case ISOMETRY_T3: {
                isoZ = isoT3('R', col, -1, accT, dbg);
                break;
            }
            case ISOMETRY_T4: {
                isoZ = isoT4('R', col, -1, accT, dbg);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[0] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        for (int row=0; row<sizeN; row++) {
        
            /*
             * Absorb one column of network to the right, leading to new 
             * (unreduced) environment tensors for sites in column to the 
             * left (-1 in X coord)
             *
             *   I_U--|T_U_m-10|--|C_RU_m-10| ==> I_U1<<I_U--|C_RU_m-20|
             *          |           |                          |     |
             *         I_XV        I_R                        I_XV  I_R
             *
             *         I_XV    I_R                            I_XV  I_R
             *         _|__     |                              |    |  
             *  I_XH--|X_00|--|T_R_m-10| ==>    I_XH1<<I_XH--|T_R_m-20|   
             *          |       |                              |    |
             *         I_XV1   I_R1                           I_XV1 I_R1
             *
             *        I_XV1         I_R1                     I_XV1 I_R1 
             *        _|_______      |                        |     |
             *  I_D--|T_D_m-10|--|C_RD_m-10| ==> I_D1<<I_D--|C_RD_m-20|
             *
             */
            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(2."<< row <<".1) ----- C_RU & T_U ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC2 = C_RU.at( cToS.at(std::make_pair(col,row)) ) * 
                T_U.at( cToS.at(std::make_pair(col,row)) );
            tC2.prime(ULINK);
            if(dbg) Print(tC2);

            if(dbg) std::cout <<"(3."<< row <<".2) ----- T_R & X ["<< col <<","<< row <<
                "] -----"<< std::endl;
            // auto tT2 = ( T_R.at( cToS.at(std::make_pair(col,row)) ) * 
            //     sites[cToS[std::make_pair(col,row)]] );
            // tT2.prime(HSLINK);
            // if(dbg) printfln("= %s", tT2);

            if(dbg) std::cout <<"(2."<< row <<".3) ----- C_RD & T_D ["<< col <<","
                << row <<"] -----"<< std::endl;
            auto tC3 = C_RD.at( cToS.at(std::make_pair(col,row)) ) * 
                T_D.at( cToS.at(std::make_pair(col,row)) );
            tC3.prime(DLINK);
            if(dbg) Print(tC3);

            t_iso_end = std::chrono::steady_clock::now();
            accT[1] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            t_iso_begin = std::chrono::steady_clock::now();

            if(dbg) std::cout <<"(3."<< row <<".1) ----- Construct reduced C_RU -----"
                <<" isoZ["<< (2*row)%(2*sizeN) <<"]"<< std::endl;
            /*                                      ________
             * I_U1--|C_RU_m-20|             I_R0--|        \
             *         |     |    *contract*       |ZsizeN-1 --I_RsizeN+10>>I_R0
             *        I_XV0  I_R0           I_XV0--|________/
             *
             */
            C_RU.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) ) = 
                tC2 * isoZ[(2*row)%(2*sizeN)];
            C_RU.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) )
                .mapprime(RLINK,sizeN+10,0);
            if(dbg) { std::cout << TAG_C_RU <<"["<< (col-1+sizeM)%sizeM <<","<< row <<"]";
            printfln("= %s",
               C_RU.at(cToS.at(std::make_pair((col-1+sizeM)%sizeM,row))) );
            
            std::cout <<"(3."<< row <<".2) ----- REDUCE T_R ["<< 
                (col-1+sizeM)%sizeM <<","<< row <<"] ----- isoZ["
                << (2*row+1)%(2*sizeN) <<"] & "<<" isoZ["<< (2*row+2)%(2*sizeN) 
                <<"]"<< std::endl; }
            /* 
             *       I_XV0 I_R0                 I_R0--|Zr-1\__I_RsizeN+10>>I_R0 
             *         |    |                  I_XV0--|dag /
             * I_XH1--|T_R_m-2r|  *contract*  
             *         |    |                   I_R1--|Zr  \__I_RsizeN+10>>I_R1
             *       I_XV1 I_R1                I_XV1--|    /
             *
             */
            auto tT2 = T_R.at( cToS.at(std::make_pair(col,row)) ) *isoZ[(2*row+1)%(2*sizeN)]; 
            tT2 = ( tT2 * sites[cToS[std::make_pair(col,row)]] ).prime(HSLINK);
            tT2.mapprime(RLINK,sizeN+10,0);

            T_R.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) ) = 
                tT2 * isoZ[(2*row+2)%(2*sizeN)].prime();
            T_R.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) )
                .mapprime(RLINK,sizeN+10+1,1);

            isoZ[(2*row+2)%(2*sizeN)].prime(-1);

            if(dbg) { std::cout << TAG_T_R <<"["<< (col-1+sizeM)%sizeM <<","<< row <<"]";
            printfln("= %s", T_R.at( 
               cToS.at(std::make_pair((col-1+sizeM)%sizeM,row))) );

            std::cout <<"(3."<< row <<".3) ----- Construct reduced C_RD -----"
                <<" isoZ["<< (2*row+3)%(2*sizeN) <<"]"<< std::endl; }
            /*                                             ________
             *       I_XV1 I_R1               I_R1<<I_R0--|        \
             *        |     |    *contract*               |ZsizeN-1 --I_RsizeN+10
             * I_D1--|C_RD_m-20|            I_XV1<<I_XV0--|_dagger_/     >>I_R1
             *
             */
            isoZ[row].conj();
            C_RD.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) ) = 
                tC3 * isoZ[(2*row+3)%(2*sizeN)].prime();
            C_RD.at( cToS.at(std::make_pair((col-1+sizeM)%sizeM,row)) )
                .mapprime(RLINK,sizeN+10+1,1);

            isoZ[(2*row+3)%(2*sizeN)].prime(-1);

            t_iso_end = std::chrono::steady_clock::now();
            accT[2] += std::chrono::duration_cast
                <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            if(dbg) { std::cout << TAG_C_RD <<"["<< (col-1+sizeM)%sizeM <<","<< row <<"]";
            printfln("= %s", C_RD.at( 
               cToS.at(std::make_pair((col-1+sizeM)%sizeM,row))) ); }
        }

        if(dbg) std::cout <<"(4) ----- NORMALIZE "<< std::string(47,'-') << std::endl;

        t_iso_begin = std::chrono::steady_clock::now();

        switch(norm_type) {
            case NORM_BLE: {
                normalizeBLE_ctmStep('R', (col-1+sizeM)%sizeM, -1);
                break;
            }
            case NORM_PTN: {
                normalizePTN_ctmStep('R', (col-1+sizeM)%sizeM, -1);
                break;
            }
        }

        t_iso_end = std::chrono::steady_clock::now();
        accT[3] += std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

        if(dbg) std::cout <<"Column "<< col <<" done"<< std::endl;
    }

    if(dbg) std::cout <<"##### InsRCol Done "<< std::string(53,'#') << std::endl;
}

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
                spec = svd(tRDM, isoZ[r], S, V, {"Maxm=",x,"SVDThreshold",1E-2});
                Print(spec);
                PrintData(S);
                // relabel isometry index
                svdI = commonIndex(isoZ[r],S);
                isoZ[r] *= delta(svdI, prime(I_L, sizeN+10));
            }
            break;
        }
        case 'U': {
            isoZ = std::vector<ITensor>(sizeM, ITensor(I_U, I_XH));
            for (int c=0; c<sizeM; c++) {
                tRDM = build_2x2_RDM('U', (c+1)%sizeM, row);
                spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
                Print(spec);
                PrintData(S);
                svdI = commonIndex(isoZ[c],S);
                isoZ[c] *= delta(svdI, prime(I_U, sizeM+10));
            }
            break;
        }
        case 'R': {
            isoZ = std::vector<ITensor>(sizeN, ITensor(I_R, I_XV));
            for (int r=0; r<sizeN; r++) {
                tRDM = build_2x2_RDM('R', col, (r+1)%sizeN);
                spec = svd(tRDM, isoZ[r], S, V, {"Maxm=",x,"SVDThreshold",1E-2});
                Print(spec);
                PrintData(S);
                svdI = commonIndex(isoZ[r],S);
                isoZ[r] *= delta(svdI, prime(I_R, sizeN+10));
            }
            break;
        }
        case 'D': {
            isoZ = std::vector<ITensor>(sizeM, ITensor(I_D, I_XH));
            for (int c=0; c<sizeM; c++) {
                tRDM = build_2x2_RDM('D', c, row);
                spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
                Print(spec);
                PrintData(S);
                svdI = commonIndex(isoZ[c],S);
                isoZ[c] *= delta(svdI, prime(I_D, sizeM+10));
            }
            break;
        }
    }

    return isoZ;   
}

std::vector<ITensor> CtmEnv::isoT2(char ctmMove, int col, int row, 
    std::vector<double> & accT, bool dbg) {
    
    if(dbg) std::cout <<"----- ISO_T2 called for "<< ctmMove <<" at ["<< col
        <<","<< row <<"] -----"<<std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;

    std::vector<ITensor> isoZ;
    
    std::pair<ITensor, ITensor> halves;
    
    //auto argsSVDhalves = Args("Maxm",x);
    auto argsSVDhalves = Args();
    auto argsSVDRRt    = Args("Maxm",x,"SVDThreshold",1E-2);

    ITensor R, Rt;
    Index auxIR, auxIRt, sIU, sIV;

    ITensor S, U, V;
    Index svdI;
    Spectrum spec;
    int p1, p2;

    // Take the square-root of SV's
    auto oneOverSqrtT = [](double r) { return 1.0/sqrt(r); };

    switch(ctmMove) {
        case 'L': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_L, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // STEP 1 build upper and lower half
                if(dbg) std::cout <<"Upper and lower half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('L', col, r);
                if(dbg) Print(halves.first);  // upper_h
                if(dbg) Print(halves.second); // lower_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 2 Obtain R and Rt(ilde)
                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                R = ITensor(I_L, I_XV);
                spec = svd(halves.first, R, S, U, argsSVDhalves);
                std::cout.precision( std::numeric_limits< double >::max_digits10 );
                if(dbg) PrintData(S);
                if(dbg) Print(spec);
                R *= S;
                auxIR = commonIndex(R,S);
                Rt = ITensor(I_L, I_XV);
                spec = svd(halves.second, Rt, S, U, argsSVDhalves);
                if(dbg) PrintData(S);
                if(dbg) Print(spec);
                Rt *= S;
                auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 3
                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                U = ITensor(auxIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                if(dbg) PrintData(S);

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 4
                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2+1]);
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[3] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
        case 'U': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_U, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('U', (c+1)%sizeM, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_U, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_U, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"left and right half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('U', (c+1)%sizeM, row);
                if(dbg) Print(halves.first);  // left_h
                if(dbg) Print(halves.second); // right_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                R = ITensor(I_U, I_XH);
                svd(halves.first, R, S, U, argsSVDhalves);
                R *= S;
                auxIR = commonIndex(R,S);
                Rt = ITensor(I_U, I_XH);
                svd(halves.second, Rt, S, U, argsSVDhalves);
                Rt *= S;
                auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(auxIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                if(dbg) PrintData(S);

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]);

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0; 
            }
            break;
        }
        case 'R': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_R, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"Upper and lower half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('R', col, (r+1)%sizeN);
                if(dbg) Print(halves.first);
                if(dbg) Print(halves.second);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                R = ITensor(I_R, I_XV);
                svd(halves.first, R, S, U, argsSVDhalves);
                R *= S;
                auxIR = commonIndex(R,S);
                Rt = ITensor(I_R, I_XV);
                svd(halves.second, Rt, S, U, argsSVDhalves);
                Rt *= S;
                auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(auxIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                if(dbg) PrintData(S);

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            }
            break;
        }
        case 'D': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_D, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('D', c, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_D, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_D, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"left and right half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('D', c, row);
                if(dbg) Print(halves.first);  // left_h
                if(dbg) Print(halves.second); // right_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                R = ITensor(I_D, I_XH);
                svd(halves.first, R, S, U, argsSVDhalves);
                R *= S;
                auxIR = commonIndex(R,S);
                Rt = ITensor(I_D, I_XH);
                svd(halves.second, Rt, S, U, argsSVDhalves);
                Rt *= S;
                auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(auxIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                if(dbg) PrintData(S);

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg)  Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]); 

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
    }

    return isoZ;
}

std::vector<ITensor> CtmEnv::isoT3(char ctmMove, int col, int row, 
    std::vector<double> & accT, bool dbg) {
    
    if(dbg) std::cout <<"----- ISO_T3 called for "<< ctmMove <<" at ["<< col
        <<","<< row <<"] -----"<<std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;

    std::vector<ITensor> isoZ;
    
    auto argsSVDRRt = Args(
        "Cutoff",-1.0,
        "Maxm",x,
        "SVDThreshold",1E-2,
        "SVD_METHOD",SVD_METHOD,
        "rsvd_power",rsvd_power,
        "rsvd_reortho",rsvd_reortho,
        "rsvd_oversampling",rsvd_oversampling
    );

    ITensor R, Rt;
    Index sIU, sIV;

    ITensor S, U, V;
    Index svdI;
    Spectrum spec;
    int p1, p2;

    // Take the square-root of SV's
    double loc_psdInvCutoff = isoPseudoInvCutoff;
    double max_sv;
    auto oneOverSqrtT = [&max_sv, &loc_psdInvCutoff](Real r) 
        { return (r/max_sv > loc_psdInvCutoff) ? 1.0/sqrt(r) : 0.0; };

    switch(ctmMove) {
        case 'L': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_L, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // STEP 1 build upper and lower corner
                if(dbg) std::cout <<" LEFT Upper and lower corner of cell+env: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                R = build_corner('1', col, r, dbg);
                Rt = build_corner('4', col, (r+1) % sizeN, dbg);
                R.mapprime(LLINK,1,0, VSLINK,1,0);
                Rt.mapprime(LLINK,1,0, VSLINK,1,0);

                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 2
                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                U = ITensor(I_U, I_XH);
                spec = svd(R*Rt, U, S, V, solver, argsSVDRRt);
                if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
                    S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
                    std::cout << "WARNING: CTM-Iso3 " << ctmMove << " [col:row]= ["<< col <<":"<< r
                        <<"] Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
                        << std::endl;
                }

                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 3
                t_iso_begin = std::chrono::steady_clock::now();

                // Create pseudo-inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2+1]);
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[3] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
        case 'U': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_U, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('U', (c+1)%sizeM, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_U, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_U, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build left and right corner
                if(dbg) std::cout <<"UP left and right corner of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                R = build_corner('1', c, row, dbg);
                Rt = build_corner('2', (c+1) % sizeM, row, dbg);

                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(prime(I_L,1), prime(I_XV,1));
                spec = svd(R*Rt, U, S, V, solver, argsSVDRRt);
                if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
                    S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
                    std::cout << "WARNING: CTM-Iso3 " << ctmMove << " [col:row]= ["<< c <<":"<< row
                        <<"] Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
                        << std::endl;
                }
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]);

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0; 
            }
            break;
        }
        case 'R': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_R, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"RIGHT Upper and lower half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                R = build_corner('2', col, r, dbg);
                Rt = build_corner('3', col, (r+1) % sizeN, dbg);

                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(I_U, I_XH);
                spec = svd(R*Rt, U, S, V, solver, argsSVDRRt);
                if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
                    S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
                    std::cout << "WARNING: CTM-Iso3 " << ctmMove << " [col:row]= ["<< col <<":"<< r
                        <<"] Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
                        << std::endl;
                }
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            }
            break;
        }
        case 'D': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_D, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('D', c, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_D, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_D, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"DOWN left and right half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                R = build_corner('4', c, row, dbg);
                Rt = build_corner('3', (c+1) % sizeM, row, dbg);
                R.mapprime(HSLINK,1,0, DLINK,1,0);
                Rt.mapprime(HSLINK,1,0, DLINK,1,0);

                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                U = ITensor(prime(I_L,1), prime(I_XV,1));
                spec = svd(R*Rt, U, S, V, solver, argsSVDRRt);
                if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
                    S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
                    std::cout << "WARNING: CTM-Iso3 " << ctmMove << " [col:row]= ["<< c <<":"<< row
                        <<"] Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
                        << std::endl;
                }
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg)  Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]); 

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
    }

    return isoZ;
}

std::vector<ITensor> CtmEnv::isoT4(char ctmMove, int col, int row, 
    std::vector<double> & accT, bool dbg) {
    
    if(dbg) std::cout <<"----- ISO_T4 called for "<< ctmMove <<" at ["<< col
        <<","<< row <<"] -----"<<std::endl;
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;

    std::vector<ITensor> isoZ;
    
    std::pair<ITensor, ITensor> halves;
    
    //auto argsSVDhalves = Args("Maxm",x);
    auto argsSVDhalves = Args();
    auto argsSVDRRt = Args(
        "Cutoff",-1.0,
        "Maxm",x,
        "Minm",x,
        "SVDThreshold",1E-2,
        "SVD_METHOD",SVD_METHOD,
        "rsvd_power",rsvd_power,
        "rsvd_reortho",rsvd_reortho
    );

    ITensor R, Rt;
    Index auxIR, auxIRt, sIU, sIV;

    ITensor S, U, V, cmbR, cmbRt;
    Index svdI, cmbIR, cmbIRt;
    Spectrum spec;
    int p1, p2;

    // Take the square-root of SV's
     // Take the square-root of SV's
    double loc_psdInvCutoff = isoPseudoInvCutoff;
    double max_sv;
    auto oneOverSqrtT = [&max_sv, &loc_psdInvCutoff](Real r) 
        // { return (r/max_sv > loc_psdInvCutoff) ? 1.0/sqrt(r) : 1.0e-8; };
        { return (r/max_sv > loc_psdInvCutoff) ? 1.0/sqrt(r) : 0.0; };


    switch(ctmMove) {
        case 'L': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_L, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // STEP 1 build upper and lower half
                if(dbg) std::cout <<"Upper and lower half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('L', col, r);
                if(dbg) Print(halves.first);  // upper_h
                if(dbg) Print(halves.second); // lower_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 2 Obtain R and Rt(ilde)
                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                //R = ITensor(I_L, I_XV);
                cmbR  = prime(combiner(I_R,I_XV));
                cmbIR = combinedIndex(cmbR);
                R = halves.first*cmbR;
                //spec = svd(halves.first, R, S, U, argsSVDhalves);
                //std::cout.precision( std::numeric_limits< double >::max_digits10 );
                //if(dbg) PrintData(S);
                //if(dbg) Print(spec);
                //R *= S;
                //auxIR = commonIndex(R,S);
                //Rt = ITensor(I_L, I_XV);
                cmbRt  = prime(combiner(I_R,I_XV));
                cmbIRt = combinedIndex(cmbRt);
                Rt = halves.second*cmbRt;
                //spec = svd(halves.second, Rt, S, U, argsSVDhalves);
                //if(dbg) PrintData(S);
                //if(dbg) Print(spec);
                //Rt *= S;
                //auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 3
                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;

                t_iso_begin = std::chrono::steady_clock::now();

                //U = ITensor(auxIR);
                U = ITensor(cmbIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                //if(dbg) S.apply(printSVs); //PrintData(S);
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // STEP 4
                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                //isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_L,sizeN+10));
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_L,sizeN+10));
                //PrintData(isoZ[r*2+1]);
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[3] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
        case 'U': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_U, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('U', (c+1)%sizeM, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_U, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_U, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"left and right half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('U', (c+1)%sizeM, row);
                if(dbg) Print(halves.first);  // left_h
                if(dbg) Print(halves.second); // right_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                //R = ITensor(I_U, I_XH);
                cmbR  = prime(combiner(I_D,I_XH));
                cmbIR = combinedIndex(cmbR);
                R = halves.first*cmbR;
                //svd(halves.first, R, S, U, argsSVDhalves);
                //R *= S;
                //auxIR = commonIndex(R,S);
                //Rt = ITensor(I_U, I_XH);
                cmbRt  = prime(combiner(I_D,I_XH));
                cmbIRt = combinedIndex(cmbRt);
                Rt = halves.second*cmbRt;
                //svd(halves.second, Rt, S, U, argsSVDhalves);
                //Rt *= S;
                //auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                //U = ITensor(auxIR);
                U = ITensor(cmbIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                //if(dbg) S.apply(printSVs); //PrintData(S);
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_U,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]);

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0; 
            }
            break;
        }
        case 'R': {
            isoZ = std::vector<ITensor>(2*sizeN, ITensor(I_R, I_XV));
            // iterate over rows and create isometries
            for (int r=0; r<sizeN; r++) {
                if(dbg) std::cout <<"Computing Projector for row: "<< r << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"Upper and lower half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('R', col, (r+1)%sizeN);
                if(dbg) Print(halves.first);
                if(dbg) Print(halves.second);

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                //R = ITensor(I_R, I_XV);
                cmbR  = prime(combiner(I_L,I_XV));
                cmbIR = combinedIndex(cmbR);
                R = halves.first*cmbR;
                //svd(halves.first, R, S, U, argsSVDhalves);
                //R *= S;
                //auxIR = commonIndex(R,S);
                //Rt = ITensor(I_R, I_XV);
                cmbRt  = prime(combiner(I_L,I_XV));
                cmbIRt = combinedIndex(cmbRt);
                Rt = halves.second*cmbRt;
                //svd(halves.second, Rt, S, U, argsSVDhalves);
                //Rt *= S;
                //auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                //U = ITensor(auxIR);
                U = ITensor(cmbIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                //if(dbg) S.apply(printSVs); // PrintData(S);
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(r+1))%(2*sizeN);
                p2 = (2*(r+1)+1)%(2*sizeN);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_R,sizeN+10));
                if(dbg) Print(isoZ[p2]);
            
                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

            }
            break;
        }
        case 'D': {
            // isoZ = std::vector<ITensor>(sizeM, ITensor(I_D, I_XH));
            // for (int c=0; c<sizeM; c++) {
            //     tRDM = build_2x2_RDM('D', c, row);
            //     spec = svd(tRDM, isoZ[c], S, V, {"Maxm",x,"SVDThreshold",1E-2});
            //     Print(spec);
            //     PrintData(S);
            //     svdI = commonIndex(isoZ[c],S);
            //     isoZ[c] *= delta(svdI, prime(I_D, sizeM+10));
            // }
            isoZ = std::vector<ITensor>(2*sizeM, ITensor(I_D, I_XH));
            // iterate over rows and create isometries
            for (int c=0; c<sizeM; c++) {
                if(dbg) std::cout <<"Computing Projector for col: "<< c << std::endl;
                // build upper and lower half
                if(dbg) std::cout <<"left and right half of 2x2(+Env) cell: "<< std::endl;
                
                t_iso_begin = std::chrono::steady_clock::now();

                halves = build_halves('D', c, row);
                if(dbg) Print(halves.first);  // left_h
                if(dbg) Print(halves.second); // right_h

                t_iso_end = std::chrono::steady_clock::now();
                accT[4] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                // Obtain R and Rt(ilde)

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"R and R~ obtained: "<< std::endl;
                //R = ITensor(I_D, I_XH);
                cmbR  = prime(combiner(I_U,I_XH));
                cmbIR = combinedIndex(cmbR);
                R = halves.first*cmbR;
                //svd(halves.first, R, S, U, argsSVDhalves);
                //R *= S;
                //auxIR = commonIndex(R,S);
                //Rt = ITensor(I_D, I_XH);
                cmbRt  = prime(combiner(I_U,I_XH));
                cmbIRt = combinedIndex(cmbRt);
                Rt = halves.second*cmbRt;
                //svd(halves.second, Rt, S, U, argsSVDhalves);
                //Rt *= S;
                //auxIRt = commonIndex(Rt,S);
                if(dbg) Print(R);
                if(dbg) Print(Rt);

                t_iso_end = std::chrono::steady_clock::now();
                accT[5] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                if(dbg) std::cout <<"SVD of R*R~: "<< std::endl;
                //U = ITensor(auxIR);
                U = ITensor(cmbIR);
                spec = svd(R*Rt, U, S, V, argsSVDRRt);
                //if(dbg) S.apply(printSVs); // PrintData(S);
                if(dbg || DBG) {
                    Print(S);
                    std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    for(int isv=1; isv<=S.inds().front().m(); isv++) 
                        std::cout << S.real(S.inds().front()(isv),S.inds().back()(isv)) 
                            << std::endl;
                }

                t_iso_end = std::chrono::steady_clock::now();
                accT[6] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;

                t_iso_begin = std::chrono::steady_clock::now();

                // Create inverse matrix
                sIU = commonIndex(U,S);
                sIV = commonIndex(S,V);
                max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
                S.apply(oneOverSqrtT);
                if(dbg) PrintData(S);

                p1 = (2*(c+1))%(2*sizeM);
                p2 = (2*(c+1)+1)%(2*sizeM);
                if(dbg) std::cout <<"Set Projectors "<< p1 <<" "<< p2 <<" :"<< std::endl;
                isoZ[p1] = Rt*V.conj()*S*delta(sIU, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2]);
                if(dbg)  Print(isoZ[p1]);
                isoZ[p2] = R*U.conj()*S*delta(sIV, prime(I_D,sizeM+10));
                //PrintData(isoZ[c*2+1]);
                if(dbg) Print(isoZ[p2]); 

                t_iso_end = std::chrono::steady_clock::now();
                accT[7] += std::chrono::duration_cast
                    <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0;
            }
            break;
        }
    }

    return isoZ;
}

ITensor CtmEnv::build_2x2_RDM(char ctmMove, int col, int row) const {

    ITensor rdm;

    std::cout << ctmMove <<" ";
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
            
            std::cout <<"RDM for "<< siteIds[cToS.at(std::make_pair(col, row))]
                << siteIds[cToS.at(std::make_pair(col, (row+1)%sizeN))]
                <<" created"<< std::endl;
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
            //rdm.mapprime(ULINK,0,1, HSLINK,0,1); // Indices along cut
            rdm.mapprime(ULINK,10,1, HSLINK,10,1); // Indices along cut
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
            //rdm.mapprime(RLINK,0,1, VSLINK,0,1); // Indices along cut
            rdm.mapprime(RLINK,10,1, VSLINK,10,1); // Indices along cut
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
    // rdm = 0.5*( rdm + swapPrime(rdm, 0, 1).conj() );

    std::cout << std::endl;
    return rdm;
}

std::pair<ITensor, ITensor> CtmEnv::build_halves(char ctmMove, int col,  
    int row, bool dbg) const {

    ITensor upper_h, lower_h;

    if(dbg) std::cout << ctmMove <<" ";
    switch (ctmMove) {
        case 'L': {
            // build upper half
            upper_h = build_corner('1', col, row);
            upper_h *= build_corner('2', (col+1)%sizeM, row);
            upper_h = swapPrime(upper_h, 0,1);
            // build lower half
            lower_h = build_corner('3', (col+1)%sizeM, (row+1)%sizeN );
            lower_h *= build_corner('4', col, (row+1)%sizeN );
            lower_h = swapPrime(lower_h, 0,1);

            if(dbg) std::cout << std::endl <<"UH and LH for "
                << siteIds[cToS.at(std::make_pair(col, row))]
                << siteIds[cToS.at(std::make_pair(col, (row+1)%sizeN))]
                <<" created"<< std::endl;
            break;
        }
        case 'U': {
            // build right half (lower_h)
            lower_h = build_corner('2', col, row);
            lower_h *= build_corner('3', col, (row+1)%sizeN );
            // build left half (upper_h)
            upper_h = build_corner('4', (col-1+sizeM)%sizeM, (row+1)%sizeN );
            upper_h *= build_corner('1', (col-1+sizeM)%sizeM, row );
            break;
        }
        case 'R': {
            // build lower half
            lower_h = build_corner('3', col, row);
            lower_h *= build_corner('4', (col-1+sizeM)%sizeM, row);
            // build upper half
            upper_h = build_corner('1', (col-1+sizeM)%sizeM,
                (row-1+sizeN)%sizeN );
            upper_h *= build_corner('2', col, (row-1+sizeN)%sizeN );
            break;
        }
        case 'D': {
            // build left half (upper_h)
            upper_h = build_corner('4', col, row);
            upper_h *= build_corner('1', col, (row-1+sizeN)%sizeN );
            upper_h = swapPrime(upper_h, 0,1);
            // build right half (lower_h)
            lower_h = build_corner('2', (col+1)%sizeM, (row-1+sizeN)%sizeN );
            lower_h *= build_corner('3', (col+1)%sizeM, row );
            lower_h = swapPrime(lower_h, 0,1);
            break;
        }
    }

    if(dbg) std::cout << std::endl;
    return std::make_pair(upper_h, lower_h);
}

ITensor CtmEnv::build_corner(char corner, int col, int row, bool dbg) const {
    ITensor ct;
    int siteIndex = cToS.at(std::make_pair(col,row));
    switch(corner) {
        case '1': {
            // build left upper corner
            ct = T_L.at( siteIndex );
            ct *= C_LU.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_U.at( siteIndex );
            ct *= sites[siteIndex];
            ct.mapprime(ULINK,1,0, HSLINK,1,0);
            break;
        }
        case '2': {
            // build right upper corner
            ct = T_U.at( siteIndex );
            ct *= C_RU.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_R.at( siteIndex );
            ct *= sites[siteIndex];
            ct.mapprime(RLINK,1,0, VSLINK,1,0);
            break;
        }
        case '3': {
            // build right lower corner
            ct = T_R.at( siteIndex );
            ct *= C_RD.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_D.at( siteIndex );
            ct *= sites[siteIndex];
            ct.mapprime(DLINK,0,1, HSLINK,0,1);
            break;
        }
        case '4': {
            // build left lower corner
            ct = T_D.at( siteIndex );
            ct *= C_LD.at( siteIndex ); 
            //ct *= sites[siteIndex];
            ct *= T_L.at( siteIndex );
            ct *= sites[siteIndex];
            ct.mapprime(LLINK,0,1, VSLINK,0,1);
            break;
        }
    }
    if(dbg) std::cout << siteIds.at(siteIndex);
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
void CtmEnv::normalizeBLE_ctmStep(char ctmMove, int col, int row, bool dbg) {

    double iMinEW = isoMinElemWarning;
    double iMaxEW = isoMaxElemWarning;

    auto normalizeBLE_T = [&dbg,&ctmMove,&col,&row,&iMinEW,&iMaxEW](ITensor& t)
    {
        double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        if( m > iMaxEW || m < iMinEW ) {
            std::cout << "WARNING: NormalizeBLE " << ctmMove << " [col:row]= ["
                << col <<":"<< row <<"] Max elem.: "<< m << std::endl;
        }

        if(dbg) std::cout << "MAX elem = "<< m << std::endl;
        t /= m;
    };

    if(dbg) std::cout <<"----- normalizeBLE_ctmStep called for "<< ctmMove 
        <<" ["<< col <<","<< row <<"]-----"<< std::endl;

    switch(ctmMove) {
        case 'U': { // col is ignored as we iterate over cols
            for (int c=0; c<sizeM; c++) {
                normalizeBLE_T( C_LU.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizeBLE_T( T_U.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizeBLE_T( C_RU.at( cToS.at( std::make_pair(c, row) ) ) );
            }
            break;
        }
        case 'R': { // row is ignored as we iterate over rows
            for (int r=0; r<sizeN; r++) {
                normalizeBLE_T( C_RU.at( cToS.at( std::make_pair(col, r) ) ) );
                normalizeBLE_T( T_R.at( cToS.at( std::make_pair(col, r) ) ) );
                normalizeBLE_T( C_RD.at( cToS.at( std::make_pair(col, r) ) ) );
            }
            break;
        }
        case 'D': { // col is ignored as we iterate over cols
            for (int c=0; c<sizeM; c++) {
                normalizeBLE_T( C_LD.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizeBLE_T( T_D.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizeBLE_T( C_RD.at( cToS.at( std::make_pair(c, row) ) ) );
            }
            break;
        }
        case 'L': { // row is ignored as we iterate over rows
            for (int r=0; r<sizeN; r++) {
                normalizeBLE_T( C_LU.at( 
                    cToS.at( std::make_pair(col, r) ) ) );
                if(dbg) std::cout <<"C_LU ["<< col <<","<< r <<"]"<< std::endl;

                normalizeBLE_T( T_L.at( 
                    cToS.at( std::make_pair(col, r) ) ) );
                if(dbg) std::cout <<"T_L ["<< col <<","<< r <<"]"<< std::endl;
            
                normalizeBLE_T( C_LD.at( 
                cToS.at( std::make_pair(col, r) ) ) );
                if(dbg) std::cout <<"C_LD ["<< col <<","<< r <<"]"<< std::endl;
            }
            break;
        }
        default: {
            std::cout <<"Unsupported ctmMove type - expecting one of "
                <<" U,R,D or L"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }

    if(dbg) std::cout <<"----- normalizeBLE_ctmStep for "<< ctmMove <<" END -----"
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
            for (int c=0; c<sizeM; c++) {
                normalizePTN_T( C_LU.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizePTN_T( T_U.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizePTN_T( C_RU.at( cToS.at( std::make_pair(c, row) ) ) );
            }
            break;
        }
        case 'R': { // row is ignored as we iterate over rows
            for (int r=0; r<sizeN; r++) {
                normalizePTN_T( C_RU.at( cToS.at( std::make_pair(col, r) ) ) );
                normalizePTN_T( T_R.at( cToS.at( std::make_pair(col, r) ) ) );
                normalizePTN_T( C_RD.at( cToS.at( std::make_pair(col, r) ) ) );
            }
            break;
        }
        case 'D': { // col is ignored as we iterate over cols
            for (int c=0; c<sizeM; c++) {
                normalizePTN_T( C_LD.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizePTN_T( T_D.at( cToS.at( std::make_pair(c, row) ) ) );
                normalizePTN_T( C_RD.at( cToS.at( std::make_pair(c, row) ) ) );
            }
            break;
        }
        case 'L': { // row is ignored as we iterate over rows
            for (int r=0; r<sizeN; r++) {
                normalizePTN_T( C_LU.at( cToS.at( std::make_pair(col, r) ) ) );
                std::cout <<"C_LU ["<< col <<","<< r <<"]"<< std::endl;

                normalizePTN_T( T_L.at( cToS.at( std::make_pair(col, r) ) ) );
                std::cout <<"T_L ["<< col <<","<< r <<"]"<< std::endl;

                normalizePTN_T( C_LD.at( cToS.at( std::make_pair(col, r) ) ) );
                std::cout <<"C_LD ["<< col <<","<< r <<"]"<< std::endl;
            }
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
    for(int s=0; s<sites.size(); s++) {
        svd( C_LU[s], U, S, V);
        iS = S.inds();
        for(int i=1; i<=x; i++) {
            spec.spec_clu[s][i-1] = S.real(iS[0](i), iS[1](i));
        }
    }

    /*
     * I_U1--|C_RU|--I_R
     *
     */
    U = ITensor(prime(I_U,1));
    for(int s=0; s<sites.size(); s++) {
        svd( C_RU[s], U, S, V);
        iS = S.inds();
        for(int i=1; i<=x; i++) {
            spec.spec_cru[s][i-1] = S.real(iS[0](i), iS[1](i));
        }
    }

    /*
     * I_R1--|C_RD|--I_D1
     *
     */
    U = ITensor(prime(I_R,1));
    for(int s=0; s<sites.size(); s++) {
        svd( C_RD[s], U, S, V);
        iS = S.inds();
        for(int i=1; i<=x; i++) {
            spec.spec_crd[s][i-1] = S.real(iS[0](i), iS[1](i));
        }
    }

    /*
     * I_D--|C_LD|--I_L1
     *
     */
    U = ITensor(I_D);
    for(int s=0; s<sites.size(); s++) {
        svd( C_LD[s], U, S, V);
        iS = S.inds();
        for(int i=1; i<=x; i++) {
            spec.spec_cld[s][i-1] = S.real(iS[0](i), iS[1](i));
        }
    }

}

/*
 * TODO? possibly optimize the implementation
 *
 */
void CtmEnv::printSVDspec() const {

    std::cout << "spec_clu: [0,0] ( ";
    for( const auto& val : spec.spec_clu[0] ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_cru: [0,0] ( ";
    for( const auto& val : spec.spec_cru[0] ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_crd: [0,0] ( ";
    for( const auto& val : spec.spec_crd[0] ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;

    std::cout << "spec_cld: [0,0] ( ";
    for( const auto& val : spec.spec_cld[0] ) {
        std::cout << val << " ";
    }
    std::cout <<")"<< std::endl;
}

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

CtmEnv::CtmSpec CtmEnv::getCtmSpec() const {
    return spec;
}

// TODO check consistency between input cluster c and one currently
// stored in ENV
void CtmEnv::updateCluster(Cluster const& c) {

    for( unsigned int i=0; i < siteIds.size(); i++ ) {
        sites[i] = contractOST(c.sites.at(siteIds[i]));
    }
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
            T_D.at( cToS.at(std::make_pair(col,sizeM-1)) ), 
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

CtmData CtmEnv::getCtmData_DBG() const {
    std::vector< itensor::ITensor > tT_U;
    std::vector< itensor::ITensor > tT_R;
    std::vector< itensor::ITensor > tT_D;
    std::vector< itensor::ITensor > tT_L;

    int site;

    for( int col=0; col<sizeM; col++) {
        site = cToS.at(std::make_pair(col,0));
        tT_U.push_back( prime(T_U.at( site ), ULINK, col) );
        std::cout<<"T_U["<< col <<"] <- T_U["<< site <<"=("<< col <<","<< 0
             <<")] = T_U["<< siteIds[site] <<"]"<< std::endl;

        site = cToS.at(std::make_pair(col,sizeN-1));
        tT_D.push_back( prime(T_D.at( site ), DLINK, col)
            .mapprime(VSLINK, 1, sizeN) );
        std::cout<<"T_D["<< col <<"] <- T_D["<< site <<"=("<< col <<","
            << sizeM-1 <<")] = T_D["<< siteIds[site] <<"]"<< std::endl;
    }
    for( int row=0; row<sizeN; row++) {
        site = cToS.at(std::make_pair(0,row));
        tT_L.push_back( prime(
            T_L.at( site ), LLINK, row) );
        std::cout<<"T_L["<< row <<"] <- T_L["<< site <<"=("<< 0 <<","<< row 
            <<")] = T_L["<< siteIds[site] <<"]"<< std::endl;

        site = cToS.at(std::make_pair(sizeM-1,row));
        tT_R.push_back( prime(T_R.at( site ), RLINK, row)
            .mapprime(HSLINK, 1, sizeM) );
        std::cout<<"T_R["<< row <<"] <- T_R["<< site <<"=("<< sizeM-1 <<","
            << row <<")] = T_R["<< siteIds[site] <<"]"<< std::endl;
    }

    site = cToS.at(std::make_pair(0,0));
    std::cout<<"C_LU <- C_LU["<< site <<"=("<< 0 <<","<< 0 <<")] = C_LU["
        << siteIds[site] <<"]"<< std::endl;
    site = cToS.at(std::make_pair(sizeM-1,0));
    std::cout<<"C_RU <- C_RU["<< site <<"=("<< sizeM-1 <<","<< 0 <<")] = C_RU["
        << siteIds[site] <<"]"<< std::endl;
    site = cToS.at(std::make_pair(sizeM-1,sizeN-1));
    std::cout<<"C_RD <- C_RD["<< site <<"=("<< sizeM-1 <<","<< sizeN-1 <<
        ")] = C_RD["<< siteIds[site] <<"]"<< std::endl;
    site = cToS.at(std::make_pair(0,sizeN-1));
    std::cout<<"C_LD <- C_LD["<< site <<"=("<< 0 <<","<< sizeN-1 <<")] = C_LD["
        << siteIds[site] <<"]"<< std::endl;

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

CtmData_Full CtmEnv::getCtmData_Full_DBG(bool dbg) const {
    // Indexing of T_* and C_* arrays wrt environment
    // of non-equivalent sites

    if(dbg) { std::cout <<"sites: ["<< std::endl;
    for( std::size_t i=0; i<sites.size(); i++) {
        std::cout << WS4 << siteIds[i] <<" -> "<< i << std::endl;
    }
    std::cout <<"]"<< std::endl; }

    CtmData_Full ctmDataFull = {
        x, d, sizeN, sizeM,
        sites, cToS,
        T_U, T_R, T_D, T_L,
        C_LU, C_RU, C_RD, C_LD,
        I_U, I_R, I_D, I_L,
        I_XH, I_XV };
    return ctmDataFull;
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
    for (std::complex<double> const& sv : spec.spec_clu[0]) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ru: [0,0]"; //["<< sizeM-1 <<",0] "<<;
    for (std::complex<double> const& sv : spec.spec_cru[0]) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_rd: [0,0]"; //["<< sizeM-1 <<","<< sizeN-1 <<"] ";
    for (std::complex<double> const& sv : spec.spec_crd[0]) {
        s << sv.real() << ' ';
    }
    s << std::endl;
    s << "C_ld: [0,0]"; //[0,"<< sizeN-1 <<"] ";
    for (std::complex<double> const& sv : spec.spec_cld[0]) {
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
    if( iE=="INIT_ENV_ctmrg" ) return CtmEnv::INIT_ENV_ctmrg;
    if( iE=="INIT_ENV_obc"   ) return CtmEnv::INIT_ENV_obc;
    if( iE=="INIT_ENV_pwr"   ) return CtmEnv::INIT_ENV_pwr;
    if( iE=="INIT_ENV_file"  ) return CtmEnv::INIT_ENV_file;
    std::cout << "Unsupported INIT_ENV" << std::endl;
    exit(EXIT_FAILURE);
}

CtmEnv::ISOMETRY toISOMETRY(std::string const& isoType) {
    if(isoType == "ISOMETRY_T1") return CtmEnv::ISOMETRY_T1;
    if(isoType == "ISOMETRY_T2") return CtmEnv::ISOMETRY_T2;
    if(isoType == "ISOMETRY_T3") return CtmEnv::ISOMETRY_T3;
    if(isoType == "ISOMETRY_T4") return CtmEnv::ISOMETRY_T4;
    std::cout << "Unsupported ISOMETRY" << std::endl;
    exit(EXIT_FAILURE);
}

CtmEnv::NORMALIZATION toNORMALIZATION(std::string const& normType) {
    if(normType == "NORM_BLE") return CtmEnv::NORM_BLE;
    if(normType == "NORM_PTN") return CtmEnv::NORM_PTN;
    std::cout << "Unsupported NORMALIZATION" << std::endl;
    exit(EXIT_FAILURE);
}