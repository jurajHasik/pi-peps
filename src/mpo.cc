#include "mpo.h"

using namespace itensor;

// ----- MPOs definition ----------------------------------------------
OpNS::OpNS() {}

OpNS::OpNS(int n) : nSite(n), siteIds(n), pi(n) {}

MpoNS::MpoNS() {}

MpoNS::MpoNS(int n) : OpNS(n), mpo(n), ai(n-1) {}

// MpoNS::MpoNS(MpoNS const& op) : nSite() mpo(op.mpo),  {

// }

// MPO_2site::MPO_2site() : MpoNS(2), H1(mpo.at(0)), H2(mpo.at(1)), Is1(pi.at(0)), 
//     Is2(pi.at(1)), a12(ai.at(0)) {}

MPO_3site::MPO_3site() : MpoNS(3), H1(mpo.at(0)), H2(mpo.at(1)), H3(mpo.at(2)),
    Is1(pi.at(0)), Is2(pi.at(1)), Is3(pi.at(2)), a12(ai.at(0)), a23(ai.at(1)) {}

MPO_3site::MPO_3site(MPO_3site const& op) : MpoNS(op), H1(mpo.at(0)), H2(mpo.at(1)), H3(mpo.at(2)),
    Is1(pi.at(0)), Is2(pi.at(1)), Is3(pi.at(2)), a12(ai.at(0)), a23(ai.at(1)) {}


MPO_2site::MPO_2site() : MpoNS(2), H1(mpo.at(0)), H2(mpo.at(1)),
    Is1(pi.at(0)), Is2(pi.at(1)), a12(ai.at(0)) {}

MPO_2site::MPO_2site(MPO_2site const& op) : MpoNS(op), H1(mpo.at(0)), H2(mpo.at(1)),
    Is1(pi.at(0)), Is2(pi.at(1)), a12(ai.at(0)) {}
// ----- END MPOs definition ------------------------------------------


// ----- MPOs construction --------------------------------------------
MPO_2site symmMPO2Sdecomp(ITensor const& u12, Index const& s1, 
    Index const& s2, bool dbg) {
    
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP2 decompose u12 from Left to Right (LR)
    ITensor O1, O2, SVt;

    double pw;
    auto pow_T = [&pw](double r) { return std::pow(r,pw); };

    // first SVD
    O1 = ITensor(s1,s1p);
    svd(u12,O1,SVt,O2,{"Truncate",false});
    /*
     *  s1'                    s2' 
     *   |                     |
     *  |O1|--a1--<SVt>--a2--|O2|
     *   |                     |
     *  s1                     s2
     *
     */
    Index a1 = commonIndex(O1,SVt);
    Index a2 = commonIndex(SVt,O2);

    pw = 0.5;
    SVt.apply(pow_T);

    O1 = (O1*SVt) * delta(a2,a1);
    O2 = O2*SVt;

    MPO_2site mpo2s;
    // Define physical indices
    mpo2s.Is1 = Index(TAG_MPO3S_PHYS1,s1.m(),PHYS);
    mpo2s.Is2 = Index(TAG_MPO3S_PHYS2,s2.m(),PHYS);

    // Define aux indices linking the on-site MPOs
    mpo2s.a12 = Index(TAG_MPO3S_12LINK,a1.m(),MPOLINK);

    mpo2s.H1 = ((O1*delta(s1,mpo2s.Is1)) *delta(s1p,prime(mpo2s.Is1)))
        *delta(a1,mpo2s.a12);
    mpo2s.H2 = ((O2*delta(s2,mpo2s.Is2)) *delta(s2p,prime(mpo2s.Is2)))
        *delta(a1,mpo2s.a12);

    if (dbg) PrintData(mpo2s.H1*mpo2s.H2);

    return mpo2s;
}

MPO_3site symmMPO3Sdecomp(ITensor const& u123, Index const& s1, 
	Index const& s2, Index const& s3, bool dbg) {
	
	Index s1p = prime(s1);
	Index s2p = prime(s2);
	Index s3p = prime(s3);

	// STEP2 decompose u123 from Left to Right (LR) and RL
	ITensor O1t, O3t, SVt, O1, O2, O3, SV1t, SV3t, O2Lt, O2Rt;

    double pw;
	auto pow_T = [&pw](double r) { return std::pow(r,pw); };

	// first SVD
	O1t = ITensor(s1,s1p,s2);
    svd(u123,O1t,SVt,O3t);
    /*
     *  s1'                  s2' s3' 
     *   |                    \  |
     *  |O1t|--a1--<SVt>--a2--|O3t|
     *   |  \                    |
     *  s1  s2                   s3
     *
     */
    Index a1 = commonIndex(O1t,SVt);
    Index a2 = commonIndex(SVt,O3t);

    pw = 0.5;
    SVt.apply(pow_T);

    O1t = O1t*SVt;
    O3t = O3t*SVt;

    O1 = ITensor(s1,s1p);
    svd(O1t,O1,SV1t,O2Lt);
    /*
     *  s1'
     *   |
     *  |O1|--a1L--<SV1t>--a2L--|O2Lt|--a1
     *   |                        |
     *  s1                        s2
     *
     */
    Index a1L = commonIndex(O1,SV1t);
    Index a2L = commonIndex(SV1t,O2Lt);

    O3 = ITensor(s3,s3p);
    svd(O3t,O3,SV3t,O2Rt);
    /*
     *        s2'                      s3'
     *        |                        | 
     *  a2--|O2Rt|--a1R--<SV3t>--a2R--|O3|
     *                                 |
     *                                 s3
     *
     */
    Index a1R = commonIndex(O2Rt,SV3t);
    Index a2R = commonIndex(SV3t,O3);

    pw = 0.5;
    SV1t.apply(pow_T);
	SV3t.apply(pow_T);

	O1 = O1*SV1t;
	O3 = O3*SV3t;
	O2 = SV1t * (O2Lt*delta(a1,a2)) * O2Rt * SV3t;

	MPO_3site mpo3s;
	// Define physical indices
	mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,s1.m(),PHYS);
	mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,s2.m(),PHYS);
	mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,s3.m(),PHYS);

	// Define aux indices linking the on-site MPOs
	mpo3s.a12 = Index(TAG_MPO3S_12LINK,a1L.m(),MPOLINK);
	mpo3s.a23 = Index(TAG_MPO3S_23LINK,a2R.m(),MPOLINK);

	mpo3s.H1 = ((O1*delta(s1,mpo3s.Is1)) *delta(s1p,prime(mpo3s.Is1)))
		*delta(a2L,mpo3s.a12);
	mpo3s.H2 = ( ((O2*delta(s2,mpo3s.Is2)) *delta(s2p,prime(mpo3s.Is2)))
		*delta(a1L,mpo3s.a12)) *delta(a2R,mpo3s.a23);
	mpo3s.H3 = ((O3*delta(s3,mpo3s.Is3)) *delta(s3p,prime(mpo3s.Is3)))
		*delta(mpo3s.a23,a1R);

	if (dbg) PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);

	return mpo3s;
}

MPO_3site ltorMPO3Sdecomp(ITensor const& u123, 
    Index const& s1, Index const& s2, Index const& s3, bool dbg) {
    
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP2 decompose u123 from Left to Right (LR) and RL
    ITensor SV_12_LR, SV_23_LR, O1_LR, O2_LR, O3_LR, tempLR;

    double pw;
    auto pow_T = [&pw](double r) { return std::pow(r,pw); };

    // first SVD
    O1_LR = ITensor(s1,s1p);
    svd(u123,O1_LR,SV_12_LR,tempLR);
    /*
     *  s1'                    s2' s3' 
     *   |                      |  |
     *  |H1|--a1--<SV_12>--a2--|temp|
     *   |                      |  |
     *  s1                     s2  s3
     *
     */

    Index a1_LR = commonIndex(SV_12_LR,O1_LR);
    Index a2_LR = commonIndex(SV_12_LR,tempLR);

    pw = 2.0/3.0;
    SV_12_LR.apply(pow_T);

    /*
     *  s1'                                     s2' s3' 
     *   |                                       |  |
     *  |H1|--a1--<SV_12>^1/3--<SV_12>^2/3--a2--|temp|
     *   |                                       |  |
     *  s1                                      s2  s3
     *
     */
    tempLR = (tempLR * SV_12_LR); // --a1_LR
    
    pw = 1.0/2.0;
    SV_12_LR.apply(pow_T);

    O1_LR = O1_LR * SV_12_LR; // --a2_LR 

    // second SVD
    O2_LR = ITensor(s2,s2p,a1_LR);
    svd(tempLR,O2_LR,SV_23_LR,O3_LR);
    /*
     *  s1'                    s2'                    s3' 
     *   |                      |                      |
     *  |H1|--a2           a1--|H2|--a3--<SV_23>--a4--|H3|
     *   |                      |                      |
     *  s1                     s2                     s3
     *
     */

    Index a3_LR = commonIndex(SV_23_LR,O2_LR);
    Index a4_LR = commonIndex(SV_23_LR,O3_LR);

    pw = 1.0/2.0;
    SV_23_LR.apply(pow_T);

    O2_LR = O2_LR * SV_23_LR; // --a4_LR
    O3_LR = O3_LR * SV_23_LR; // --a3_LR
    
    if (dbg) { 
        PrintData(O1_LR);
        PrintData(O2_LR);
        PrintData(O3_LR);
    }

    MPO_3site mpo3s;
    // Define physical indices
    mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,s1.m(),PHYS);
    mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,s2.m(),PHYS);
    mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,s3.m(),PHYS);

    // Define aux indices linking the on-site MPOs
    mpo3s.a12 = Index(TAG_MPO3S_12LINK,a1_LR.m(),MPOLINK);
    mpo3s.a23 = Index(TAG_MPO3S_23LINK,a3_LR.m(),MPOLINK);

    /*
     *  s1'                    s2'                    s3' 
     *   |                      |                      |
     *  |H1|--a2           a1--|H2|--a4           a3--|H3|
     *   |                      |                      |
     *  s1                     s2                     s3
     *
     */
    O1_LR = O1_LR * delta(a2_LR,mpo3s.a12);
    O2_LR = (O2_LR * delta(a1_LR,mpo3s.a12)) *delta(a4_LR,mpo3s.a23);
    O3_LR = O3_LR * delta(a3_LR,mpo3s.a23);

    mpo3s.H1 = (O1_LR*delta(s1,mpo3s.Is1))*delta(s1p,prime(mpo3s.Is1));
    mpo3s.H2 = (O2_LR*delta(s2,mpo3s.Is2))*delta(s2p,prime(mpo3s.Is2));
    mpo3s.H3 = (O3_LR*delta(s3,mpo3s.Is3))*delta(s3p,prime(mpo3s.Is3));

    if (dbg) {
        PrintData(mpo3s.H1);
        PrintData(mpo3s.H2);
        PrintData(mpo3s.H3);
        PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);
    }

    return mpo3s;
}

MPO_3site ltorMPO2StoMPO3Sdecomp(ITensor const& u123, 
    Index const& s1, Index const& s2, bool dbg) {

    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP2 decompose u123 from Left to Right (LR) and RL
    ITensor SV_12, O1_L, O2_R;

    double pw;
    auto pow_T = [&pw](double r) { return std::pow(r,pw); };

    // first SVD
    O1_L = ITensor(s1,s1p);
    svd(u123,O1_L,SV_12,O2_R);
    /*
     *  s1'                    s2' 
     *   |                      |
     *  |H1|--a1--<SV_12>--a2--|temp|
     *   |                      |
     *  s1                     s2
     *
     */
    //PrintData(SV_12_LR);

    Index a1_LR = commonIndex(SV_12,O1_L);
    Index a2_LR = commonIndex(SV_12,O2_R);

    pw = 0.5;
    SV_12.apply(pow_T);

    /*
     *  s1'                                     s2' 
     *   |                                       |
     *  |H1|--a1--<SV_12>^1/2--<SV_12>^1/2--a2--|temp|
     *   |                                       |
     *  s1                                      s2
     *
     */
    O2_R = (O2_R * SV_12); // --a1_LR
    O1_L = O1_L * SV_12; // --a2_LR 

    if (dbg) {
        PrintData(O1_L);
        PrintData(O2_R);
    }

    MPO_3site mpo3s;
    // Define physical indices
    if (s1.m() != s2.m()) {
        std::cout << "ltorMPO2StoMPO3Sdecomp: dim(s1) != dim(s2)"<< std::endl;
        exit(EXIT_FAILURE);
    }
    mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,s1.m(),PHYS);
    mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,s2.m(),PHYS);
    mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,s1.m(),PHYS);

    // Define aux indices linking the on-site MPOs
    mpo3s.a12 = Index(TAG_MPO3S_12LINK,a1_LR.m(),MPOLINK);
    mpo3s.a23 = Index(TAG_MPO3S_23LINK,1,MPOLINK);

    /*
     *  s1'                    s2'                    s3' 
     *   |                      |                      |
     *  |H1|--a2           a1--|H2|--a4           a3--|H3|
     *   |                      |                      |
     *  s1                     s2                     s3
     *
     */
    O1_L = O1_L * delta(a2_LR,mpo3s.a12);
    
    ITensor tempO2(mpo3s.a23);
    tempO2.fill(1.0);
    O2_R = (O2_R * delta(a1_LR,mpo3s.a12)) * tempO2;

    mpo3s.H1 = (O1_L*delta(s1,mpo3s.Is1))*delta(s1p,prime(mpo3s.Is1));
    mpo3s.H2 = (O2_R*delta(s2,mpo3s.Is2))*delta(s2p,prime(mpo3s.Is2));
    mpo3s.H3 = tempO2*delta(mpo3s.Is3,prime(mpo3s.Is3));

    if (dbg) {
        PrintData(mpo3s.H1);
        PrintData(mpo3s.H2);
        PrintData(mpo3s.H3);
        PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);
    }

    return mpo3s;
}
// ----- END MPOs construction ----------------------------------------

std::ostream& 
operator<<(std::ostream& s, MPO_3site const& mpo3s) {
    s <<"----- BEGIN MPO_3site "<< std::string(50,'-') << std::endl;
    s << mpo3s.Is1 <<" "<< mpo3s.Is2 <<" "<< mpo3s.Is3 << std::endl;
    s <<"H1 "<< mpo3s.H1 << std::endl;
    s <<"H2 "<< mpo3s.H2 << std::endl;
    s <<"H3 "<< mpo3s.H3;
    s <<"----- END MPO_3site "<< std::string(52,'-') << std::endl;
    return s;
}