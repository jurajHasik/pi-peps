#include "cluster-ev-builder.h"

using namespace itensor;

//Default constructor
EVBuilder::EVBuilder () {}

EVBuilder::EVBuilder (std::string in_name, Cluster const& in_cls, 
    CtmEnv const& in_env) 
    : name(in_name), cls(in_cls), env(in_env) {}

// TODO expose the indices linking site with environment from
// CtmEnv I(X*), so one can construct TOT's without explicit need
// of constructed environment
/* 
 * If the on-site tensor T (ket) has been initialized, constructs
 * a contracted <bra|MPO|ket> tensor, with a given prime level "l"
 * 
 */
ITensor EVBuilder::getTOT_DBG(MPO mpo, itensor::ITensor const& T,
        int primeLvl) const 
{
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
    Print(T);

    // Get auxBond index of T
    auto auxI = noprime(findtype(T.inds(), Link));
    // Get physical index of T
    auto s = noprime(findtype(T.inds(), Site));

    if(auxI.m()*auxI.m() != env.d) {
        std::cout <<"ctmEnv.d does not agree with onSiteT.dimD^2"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Define combiner tensors Y*
    auto C04 = combiner(auxI, prime(auxI,4));
    auto C15 = prime(C04,1);
    auto C26 = prime(C04,2);
    auto C37 = prime(C04,3);

    // Construct MPO
    auto Op = ITensor(prime(s,1), s);

    switch(mpo) {
        case MPO_Id: {
            std::cout << ">>>>> Constructing T^dag*Id*T <<<<<"<<"\n";
            for(int i=1;i<=s.m();i++)
                Op.set(s(i), prime(s,1)(i), 1.+0._i);
            break;
        }
        case MPO_S_Z: {
            std::cout << ">>>>> Constructing T^dag*Sz*T <<<<<"<<"\n";
            for(int i=1;i<=s.m();i++)
                Op.set(s(i), prime(s,1)(i), 0.5*(-(s.m()-1) 
                            + (i-1)*2) + 0._i);
            break;
        }
        case MPO_S_Z2: {
            std::cout << ">>>>> Constructing T^dag*Sz^2*T <<<<<"<<"\n";
            for(int i=1;i<=s.m();i++)
                Op.set(s(i), prime(s,1)(i), pow(0.5*(-(s.m()-1) 
                            + (i-1)*2), 2.0) + 0._i);
            break;
        }
        /* 
         * The s^+ operator maps states with s^z = x to states with 
         * s^z = x+1 . Therefore as a matrix it must act as follows
         * on vector of basis elements of spin S representation (in 
         * this particular order) |S M> 
         *       
         *     |-S  >    C_+|-S+1>           0 1 0 0 ... 0 
         * s^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
         *      ...         ...              ...
         *     | S-1>    C_+| S  >           0    ...  0 1
         *     | S  >     0                  0    ...  0 0
         *
         * where C_+ = sqrt(S(S+1)-M(M+1))
         *
         */
        case MPO_S_P: {
            std::cout << "Op MPO_S_P" << "\n";
            for(int i=1;i<=s.m()-1;i++)
                Op.set(s(i+1), prime(s,1)(i), pow( 0.5*(s.m()-1)
                    *(0.5*(s.m()-1)+1) - (-0.5*(s.m()-1)+(i-1))
                    *(-0.5*(s.m()-1)+(i-1)+1), 0.5) + 0._i);
            break;
        }
        /* 
         * The s^- operator maps states with s^z = x to states with 
         * s^z = x-1 . Therefore as a matrix it must act as follows
         * on vector of basis elements of spin S representation (in 
         * this particular order) |S M> 
         *       
         *     |-S  >     0                  0 0 0 0 ... 0 
         * s^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
         *      ...         ...              ...
         *     | S-1>    C_-| S-2>           0   ... 1 0 0
         *     | S  >    C_-| S-1>           0   ... 0 1 0
         * 
         * where C_- = sqrt(S(S+1)-M(M-1))
         *
         */
        case MPO_S_M: {
            std::cout << "Op MPO_S_M" << "\n";
            for(int i=2;i<=s.m();i++)
                Op.set(s(i-1), prime(s,1)(i), pow( 0.5*(s.m()-1)
                    *(0.5*(s.m()-1)+1) - (-0.5*(s.m()-1)+(i-1))
                    *(-0.5*(s.m()-1)+(i-1)-1), 0.5) + 0._i);
            break;
        }
        default: {
            std::cout << "Invalid MPO selection" << "\n";
            exit(EXIT_FAILURE);
            break;
        }
    }

    PrintData(Op);

    // if(isB) {
    //     std::cout << "site B - rotation on spin index" << "\n";
    //     // Operator corresponds to "odd" site of bipartite AKLT
    //     // state - perform rotation on physical indices
    //     /*
    //      * I(s)'--|Op|--I(s) => 
    //      * 
    //      * I(s)'''--|R1|--I(s)'--|Op|--I(s)--|R2|--I(s)''
    //      *
    //      * where Rot is a real symmetric rotation matrix, thus R1 = R2
    //      * defined below. Then one has to set indices of rotated
    //      * Op to proper prime level
    //      *
    //      */         
    //     auto R1 = ITensor(prime(s,3), prime(s,1));
    //     auto R2 = ITensor(s, prime(s,2));
    //     for(int i=1;i<=dimS;i++) {
    //         R1.set(prime(s,3)(i), prime(s,1)(dimS+1-i), pow(-1,i-1));
    //         R2.set(s(dimS+1-i), prime(s,2)(i), pow(-1,i-1));
    //     }
    //     PrintData(R1);
    //     PrintData(R2);
    //     Op = R1*Op*R2;
    //     PrintData(Op);
    //     Op.prime(-2);
    // }

    auto TOT = (T*Op*( conj(T).prime(Link,4).prime(Site,1) ))*C04*C15*C26*C37;

    // Define delta tensors D* to relabel combiner indices to I_XH, I_XV
    auto DH0 = delta(env.I_XH, commonIndex(TOT,C04));
    auto DV0 = delta(env.I_XV, commonIndex(TOT,C15));
    auto DH1 = delta(prime(env.I_XH,1), commonIndex(TOT,C26));
    auto DV1 = delta(prime(env.I_XV,1), commonIndex(TOT,C37));

    return TOT*DH0*DV0*DH1*DV1;
}

/*
 * TODO consider imaginary part of the result as well
 *
 */
double EVBuilder::eV_1sO(ITensor const& op, std::pair<int,int> site) const {
    auto ev = env.C_LU;

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
    for ( auto const& t : env.T_L ) {
        ev.prime(HSLINK,2);
        ev *= t;
    }
    ev *= env.C_LD;

    for ( int col=0; col<env.sizeM; col++ ) {
        ev *= env.T_D[col];

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
        for ( int row=env.sizeN-1; row>=0; row-- ) {
            ev.mapprime(0,1,VSLINK);
            // substitute original on-site tensor for op at position site
            if ( site.first == row && site.second == col ) {
                std::cout <<"OP inserted at ("<< row <<","<< col <<")"
                    << std::endl;
                ev *= prime(op, HSLINK, 2*(env.sizeN-1-row));
            } else {
                ev *= prime(env.sites.at(env.cToS.at(std::make_pair(row,col))),
                    HSLINK, 2*(env.sizeN-1-row));
            }
        }
        ev.prime(HSLINK,-1);
        ev *= env.T_U[col];

    }

    ev *= env.C_RU;
    for ( int row=env.sizeN-1; row>=0; row-- ) {
        ev.mapprime(2*row,1,HSLINK);
        ev *= env.T_R[row];
    }
    ev *= env.C_RD;

    return sumels(ev)/env.getNorm();
}

// ExpValBuilder::Mpo2S ExpValBuilder::get2STOT(ExpValBuilder::OP2S op2s,
//         Index const& i_Xh, Index const& i_Xv)
// {
//     /*
//      * 2-site operator acting on 2 physical indices
//      *   
//      * <bra|   I(s)   I(s)''
//      *          _|______|_   
//      *         |____OP____|
//      *           |      |
//      *         I(s)'  I(s)'''  |ket>     
//      *
//      */
//     auto s0 = prime(s,0);
//     auto s1 = prime(s,1);
//     auto s2 = prime(s,2);
//     auto s3 = prime(s,3);
//     auto Op = ITensor(s0, s1, s2, s3);
//     switch(op2s) {
//         case OP2S_Id: { // Identity operator
//             std::cout << ">>>>> Constructing OP2S_Id <<<<<" << "\n";  
//             for(int i=1;i<=dimS;i++) {
//                 for(int j=1;j<=dimS;j++){
//                     Op.set(s0(i),s2(j),s1(i),s3(j), 1.+0._i);
//                 }
//             }
//             break;
//         }
//         case OP2S_AKLT_S2_H: { // H of AKLT-S2 on square lattice
//             std::cout << ">>>>> Constructing OP2S_AKLT-S2-H <<<<<" << "\n";
//             // Loop over <bra| indices
//             int rS = dimS-1; // Label of SU(2) irrep in Dyknin notation
//             int mbA, mbB, mkA, mkB;
//             double hVal;
//             for(int bA=1;bA<=dimS;bA++) {
//             for(int bB=1;bB<=dimS;bB++) {
//                 // Loop over |ket> indices
//                 for(int kA=1;kA<=dimS;kA++) {
//                 for(int kB=1;kB<=dimS;kB++) {
//                     // Use Dynkin notation to specify irreps
//                     mbA = -(rS) + 2*(bA-1);
//                     mbB = -(rS) + 2*(bB-1);
//                     mkA = -(rS) + 2*(kA-1);
//                     mkB = -(rS) + 2*(kB-1);
//                     // Loop over possible values of m given by tensor product
//                     // of 2 spin (dimS-1) irreps (In Dynkin notation)
//                     hVal = 0.0;
//                     for(int m=-2*(rS);m<=2*(rS);m=m+2) {
//                         if ((mbA+mbB == m) && (mkA+mkB == m)) {
//     //DEBUG
//     std::cout <<"<"<< mbA <<","<< mbB <<"|"<< m <<"> x <"<< m <<"|"<< mkA <<","
//     << mkB<<"> = "<< SU2_getCG(rS, rS, 2*rS, mbA, mbB, m) <<" x "
//     << SU2_getCG(rS, rS, 2*rS, mkA, mkB, m) <<"\n";

//                         hVal += SU2_getCG(rS, rS, 2*rS, mbA, mbB, m) 
//                             *SU2_getCG(rS, rS, 2*rS, mkA, mkB, m);
//                         }
//                     }
//                     if((bA == kA) && (bB == kB)) {
//                         Op.set(s0(bA),s2(bB),s1(kA),s3(kB),hVal+2.);
//                     } else {
//                         Op.set(s0(bA),s2(bB),s1(kA),s3(kB),hVal);
//                     }
//                 }}
//             }}
//             break;
//         }
//         case OP2S_SS: { // S^vec_i * S^vec_i+1
//             // s^z_i*s^z_i+1 + 1/2(s^+_i*s^-_i+1 + s^-_i*s^+_i+1)
//             std::cout << ">>>>> Constructing OP2S_SS <<<<<" << "\n";
            
//             Index sBra = Index("sBra", dimS);
//             Index sKet = Index("sKet", dimS);
//             ITensor Sz = getSpinOp(MPO_S_Z, 0, sBra, sKet, false);
//             ITensor Sp = getSpinOp(MPO_S_P, 0, sBra, sKet, false);
//             ITensor Sm = getSpinOp(MPO_S_M, 0, sBra, sKet, false);
            
//             double hVal;
//             // Loop over <bra| indices
//             for(int bA=1;bA<=dimS;bA++) {
//             for(int bB=1;bB<=dimS;bB++) {
//                 // Loop over |ket> indices
//                 for(int kA=1;kA<=dimS;kA++) {
//                 for(int kB=1;kB<=dimS;kB++) {
                
//                     hVal = Sz.real(sBra(bA),sKet(kA))
//                         *Sz.real(sBra(bB),sKet(kB))+0.5*(
//                         Sp.real(sBra(bA),sKet(kA))
//                         *Sm.real(sBra(bB),sKet(kB))+
//                         Sm.real(sBra(bA),sKet(kA))
//                         *Sp.real(sBra(bB),sKet(kB)));

//                     Op.set(s0(bA),s2(bB),s1(kA),s3(kB),hVal);
//                 }}
//             }}
//             break;
//         }
//         default: {
//             std::cout << "Invalid MPO selection" << "\n";
//             exit(EXIT_FAILURE);
//             break;
//         }
//     }

//     // Perform SVD
//     /*         __
//      * I(s)---|  |--I(s)''    =>
//      *        |OP|            =>
//      * I(s)'--|__|--I(s)'''   =>
//      *            ___                      ___  
//      * => I(s)---|   |         _          |   |--I(s)''
//      * =>        |OpA|--I(o)--|S|--I(o)'--|OpB|       
//      * => I(s)'--|___|                    |___|--I(s)'''
//      *
//      */
//     auto OpA = ITensor(s0, s1);
//     ITensor OpB, S;

//     std::cout << "Performing SVD OP2S -> OpA * S * OpB" <<"\n";
//     svd(Op, OpA, S, OpB);
//     Print(OpA);
//     Print(S);
//     Print(OpB);
//     // Absorb singular values into, say, OpB
//     OpB = S*OpB;

//     // Contract OpA & OpB with <bra| & |ket> on-site tensors T
//     /*
//      * TODO write comment about this process
//      *
//      */
//     auto Tdag = prime(conj(T), Link, 2);
//     Tdag.prime(s,1);

//     // Perform rotation on OpB (we can always assume B to correspond 
//     // to "odd" site of bipartite AKLT
//     /*
//      * I(s)''--|OpB|--I(s)''' => 
//      * 
//      * I(s)--|R1|--I(s)''--|Op|--I(s)'''--|R2|--I(s)'
//      *
//      * where Rot is a real symmetric rotation matrix, thus R1 = R2
//      * defined below. Then one has to set indices of rotated
//      * Op to proper prime level
//      *
//      */
//     std::cout << "Rotation on physical indices of OpB" << "\n";
//     auto R1 = ITensor(s, s2);
//     auto R2 = ITensor(s3,s1);
//     for(int i=1;i<=dimS;i++) {
//         R1.set(s(i), s2(dimS+1-i), pow(-1,i-1));
//         R2.set(s3(dimS+1-i), s1(i), pow(-1,i-1));
//     }
//     OpB = R1*OpB*R2;
//     Print(OpB);

//     // Define conversion tensors Y
//     auto Yh  = ITensor(h, prime(h,2), i_Xh);
//     auto Yv  = ITensor(v, prime(v,2), i_Xv);
//     auto Yhp = ITensor(hp, prime(hp,2), prime(i_Xh,1));
//     auto Yvp = ITensor(vp, prime(vp,2), prime(i_Xv,1));
//     Yh.fill(0.+0._i);
//     Yv.fill(0.+0._i);
//     Yhp.fill(0.+0._i);
//     Yvp.fill(0.+0._i);
//     for(int b=1; b<=dimD; b++) {
//         for(int k=1; k<=dimD; k++) {
//             Yh.set(h(b), prime(h,2)(k), i_Xh(dimD*(b-1)+k), 1.0+0._i);
//             Yv.set(v(b), prime(v,2)(k), i_Xv(dimD*(b-1)+k), 1.0+0._i);
//             Yhp.set(hp(b), prime(hp,2)(k), prime(i_Xh,1)(dimD*(b-1)+k),
//                     1.0+0._i);
//             Yvp.set(vp(b), prime(vp,2)(k), prime(i_Xv,1)(dimD*(b-1)+k),
//                     1.0+0._i);
//         }
//     }

//     OpA = Yh*Yv*Yhp*Yvp*Tdag*OpA*T;
//     OpB = Yh*Yv*Yhp*Yvp*Tdag*OpB*T;

//     ExpValBuilder::Mpo2S mpo2S = {OpA, OpB};
//     return mpo2S;
// }

// std::complex<double> ExpValBuilder::expVal_2sO(itensor::ITensor const& opA,
//         itensor::ITensor const& opB)
// {

//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);

//     /*
//      * Construct the "left" part tensor L
//      *  _    __                       __
//      * |C|--|T |--I(T_u)'            |  |--I(T_u)'
//      *  |    |                   ==> |  |
//      * |T|--|OA|--I(XH)'(x)I(o)  ==> |L |--I(Xh)'(x)I(o)
//      *  |    |                   ==> |  |
//      * |C|--|T |--I(T_d)'            |__|--I(T_d)'
//      *
//      * where I(o) is an additional index connecting 2 MPOs making
//      * up the original 2-site operator
//      *
//      */
//     auto LA   = env.C_lu*env.T_l*env.C_ld*env.T_u*env.T_d*opA;
//     auto LId = env.C_lu*env.T_l*env.C_ld*env.T_u*env.T_d*X;
    
    
//      * Construct the "right" part tensor R
//      *                 __    _                        __
//      *        I(T_u)--|T |--|C|              I(T_u)--|  |
//      *                 |     |   ==>                 |  |
//      *  I(o)(x)I(Xh)--|O2|--|T|  ==>   I(o)(x)I(Xh)--|R |
//      *                 |     |   ==>                 |  |
//      *        I(T_d)--|T |--|C|              I(T_d)--|__|
//      *
     
//     auto RB = env.C_ru*env.T_r*env.C_rd*env.T_u*env.T_d*opB;
//     auto RId = env.C_ru*env.T_r*env.C_rd*env.T_u*env.T_d*X;

//     LA.noprime();
//     LId.noprime();
//     // Contract L*R
//     auto eBare = LA*RB;
//     auto eNorm = LId*RId;
//     //PrintData(eBare);
//     //PrintData(eNorm);
//     std::complex<double> expVal = sumelsC(eBare)/sumelsC(eNorm);
//     std::cout << "ExpVal: " << expVal <<"\n";
//     return expVal;
// }

// std::complex<double> ExpValBuilder::expVal_1sO1sO_H(int dist, 
//         itensor::ITensor const& op1, itensor::ITensor const& op2)
// {
    
//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);
//     /*
//      * Construct the "left" part tensor L
//      *  _    __                __
//      * |C|--|T |--I(T_u)'     |  |--I(T_u)'
//      *  |    |            ==> |  |
//      * |T|--|O1|--I(XH)'  ==> |L |--I(Xh)'
//      *  |    |            ==> |  |
//      * |C|--|T |--I(T_d)'     |__|--I(T_d)'
//      *
//      */
//     auto L   = env.C_lu*env.T_l*env.C_ld;
//     L = L*env.T_u*op1*env.T_d;
//     auto LId = env.C_lu*env.T_l*env.C_ld;
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
//      *          __    _                 __
//      * I(T_u)--|T |--|C|       I(T_u)--|  |
//      *          |     |   ==>          |  |
//      *  I(Xh)--|O2|--|T|  ==>   I(Xh)--|R |
//      *          |     |   ==>          |  |
//      * I(T_d)--|T |--|C|       I(T_d)--|__|
//      *
//      */
//     auto R = env.C_ru*env.T_r*env.C_rd;
//     R = R*env.T_u*op2*env.T_d;
//     auto RId = env.C_ru*env.T_r*env.C_rd;
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

// std::complex<double> ExpValBuilder::expVal_1sO1sO_V(int dist, 
//         itensor::ITensor const& op1, itensor::ITensor const& op2)
// {
    
//     auto X = ExpValBuilder::getTOT(MPO_Id, 0, env.i_Xh, env.i_Xv,
//             false);
//     /*
//      * Construct the "Up" part tensor U
//      *  _      __      _            ________________ 
//      * |C|----|T |----|C|          |_______U________|
//      *  |      |       |      ==>    |      |     |
//      * |T|----|O1|----|T|     ==>  I(Tl)' I(Xv)'  I(Tr)'
//      *  |      |       |      ==>
//      * I(Tl)' I(Xv)'  I(Tr)'
//      *
//      */
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

// ITensor ExpValBuilder::getSpinOp(ExpValBuilder::MPO mpo, int l, 
//     Index const& i_braS, Index const& i_ketS, bool isB) {

//     if (i_braS.m() != i_ketS.m()) {
//         std::cout << "Range of bra and ket indices is not equal" << "\n";
//         exit(EXIT_FAILURE);
//     }
//     // Construct MPO
//     auto Op = ITensor(i_braS, i_ketS);

//     switch(mpo) {
//         case MPO_Id: {
//             for(int i=1;i<=dimS;i++)
//                 Op.set(i_braS(i), i_ketS(i), 1.+0._i);
//             break;
//         }
//         case MPO_S_Z: {
//             for(int i=1;i<=dimS;i++)
//                 Op.set(i_braS(i), i_ketS(i), 0.5*(-(dimS-1) 
//                             + (i-1)*2) + 0._i);
//             break;
//         }
//         case MPO_S_Z2: {
//             for(int i=1;i<=dimS;i++)
//                 Op.set(i_braS(i), i_ketS(i), pow(0.5*(-(dimS-1) 
//                             + (i-1)*2), 2.0) + 0._i);
//             break;
//         }
//         case MPO_S_P: {
//             for(int i=1;i<=dimS-1;i++)
//                 Op.set(i_braS(i), i_ketS(i+1), pow( 0.5*(dimS-1)
//                     *(0.5*(dimS-1)+1) - (-0.5*(dimS-1)+(i-1))
//                     *(-0.5*(dimS-1)+(i-1)+1), 0.5) + 0._i);
//             break;
//         }
//         case MPO_S_M: {
//             for(int i=2;i<=dimS;i++)
//                 Op.set(i_braS(i), i_ketS(i-1), pow( 0.5*(dimS-1)
//                     *(0.5*(dimS-1)+1) - (-0.5*(dimS-1)+(i-1))
//                     *(-0.5*(dimS-1)+(i-1)-1), 0.5) + 0._i);
//             break;
//         }
//         default: {
//             std::cout << "Invalid MPO selection" << "\n";
//             exit(EXIT_FAILURE);
//             break;
//         }
//     }

//     if(isB) {
//         // Operator corresponds to "odd" site of bipartite AKLT
//         // state - perform rotation on physical indices
//         /*
//          * I(braS)--|Op|--I(ketS) => 
//          * 
//          * I(braS)'--|R1|--I(braS)--|Op|--I(ketS)--|R2|--I(ketS)'
//          *
//          * where Rot is a real symmetric rotation matrix, thus R1 = R2
//          * defined below. Then one has to set indices of rotated
//          * Op to proper prime level
//          *
//          */
//         auto R1 = ITensor(prime(i_braS,2), i_braS);
//         auto R2 = ITensor(i_ketS, prime(i_ketS,2));
//         for(int i=1;i<=dimS;i++) {
//             R1.set(prime(i_braS,2)(i), i_braS(dimS+1-i), pow(-1,i-1));
//             R2.set(i_ketS(dimS+1-i), prime(i_ketS,2)(i), pow(-1,i-1));
//         }
//         Op = R1*Op*R2;
//         Op.prime(-2);
//     }

//     return Op;
// }

std::ostream& EVBuilder::print(std::ostream& s) const {
    s << "ExpValBuilder("<< name <<")";
    return s;
}

std::ostream& operator<<(std::ostream& s, EVBuilder const& ev) {
    return ev.print(s);
}

std::ostream& operator<<(std::ostream& s, EVBuilder::Mpo2S
        const& mpo2S) {
    s << "Mpo2S( "<<"OpA: "<< mpo2S.opA <<"OpB: "<< mpo2S.opB <<")";
    return s;
}
