#include "models.h"

using namespace itensor;

// ----- Trotter gates (2Site, 3site, ...) MPOs -----------------------
MPO_3site symmMPO3Sdecomp(ITensor const& u123, Index const& s1, 
	Index const& s2, Index const& s3) {
	
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

	PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);

	return mpo3s;
}

MPO_3site ltorMPO3Sdecomp(ITensor const& u123, 
    Index const& s1, Index const& s2, Index const& s3) {
    
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
     
    PrintData(O1_LR);
    PrintData(O2_LR);
    PrintData(O3_LR);

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

    PrintData(mpo3s.H1);
    PrintData(mpo3s.H2);
    PrintData(mpo3s.H3);
    PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);

    return mpo3s;
}

MPO_3site ltorMPO2StoMPO3Sdecomp(ITensor const& u123, 
    Index const& s1, Index const& s2) {

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

    PrintData(O1_L);
    PrintData(O2_R);

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

    PrintData(mpo3s.H1);
    PrintData(mpo3s.H2);
    PrintData(mpo3s.H3);
    PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);

    return mpo3s;
}

MPO_3site getMPO3s_Id(int physDim) {
    MPO_3site mpo3s;
    
    // Define physical indices
    mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
    mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
    mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

    ITensor idpI1(mpo3s.Is1,prime(mpo3s.Is1,1));
    ITensor idpI2(mpo3s.Is2,prime(mpo3s.Is2,1));
    ITensor idpI3(mpo3s.Is3,prime(mpo3s.Is3,1));
    for (int i=1;i<=physDim;i++) {
        idpI1.set(mpo3s.Is1(i),prime(mpo3s.Is1,1)(i),1.0);
        idpI2.set(mpo3s.Is2(i),prime(mpo3s.Is2,1)(i),1.0);
        idpI3.set(mpo3s.Is3(i),prime(mpo3s.Is3,1)(i),1.0);
    }

    ITensor id3s = idpI1*idpI2*idpI3;

    return ltorMPO3Sdecomp(id3s, mpo3s.Is1, mpo3s.Is2, mpo3s.Is3);
}

MPO_3site getMPO3s_Id_v2(int physDim, bool dbg) {
    MPO_3site mpo3s;
    
    // Define physical indices
    mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
    mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
    mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

    // Define auxiliary indices
    mpo3s.a12 = Index(TAG_MPO3S_12LINK,1,MPOLINK);
    mpo3s.a23 = Index(TAG_MPO3S_23LINK,1,MPOLINK);

    mpo3s.H1 = ITensor(mpo3s.Is1,prime(mpo3s.Is1),mpo3s.a12);
    mpo3s.H2 = ITensor(mpo3s.Is2,prime(mpo3s.Is2),mpo3s.a12,mpo3s.a23);
    mpo3s.H3 = ITensor(mpo3s.Is3,prime(mpo3s.Is3),mpo3s.a23);
    
    for (int i=1; i<=physDim; i++) {
        mpo3s.H1.set(mpo3s.Is1(i),prime(mpo3s.Is1)(i),mpo3s.a12(1),1.0);
        mpo3s.H2.set(mpo3s.Is2(i),prime(mpo3s.Is2)(i),mpo3s.a12(1),mpo3s.a23(1),1.0);
        mpo3s.H3.set(mpo3s.Is3(i),prime(mpo3s.Is3)(i),mpo3s.a23(1),1.0);
    }

    if (dbg) {
        std::cout<<"[getMPO3s_Id_v2]"<< std::endl;
        PrintData(mpo3s.H1);
        PrintData(mpo3s.H2);
        PrintData(mpo3s.H3);
    }

    return mpo3s;
}

MPO_3site getMPO3s_Uj1j2(double tau, double J1, double J2) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP1 define exact U_123 = exp(J1(S_1.S_2 + S_2.S_3) + 2*J2(S_1.S_3))
    double a = -tau*J1/8.0;
    double b = -tau*J2/4.0;
    ITensor u123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
    double el1 = exp(2.0*a + b);
    u123.set(s1(1),s2(1),s3(1),s1p(1),s2p(1),s3p(1),el1);
    u123.set(s1(2),s2(2),s3(2),s1p(2),s2p(2),s3p(2),el1);
    double el2 = (1.0/6.0)*exp(-3.0*b)*(exp(4.0*(b-a))*(1.0+2.0*exp(6.0*a))+3.0);
    u123.set(s1(1),s2(1),s3(2),s1p(1),s2p(1),s3p(2),el2);
    u123.set(s1(1),s2(2),s3(2),s1p(1),s2p(2),s3p(2),el2);
    u123.set(s1(2),s2(1),s3(1),s1p(2),s2p(1),s3p(1),el2);
    u123.set(s1(2),s2(2),s3(1),s1p(2),s2p(2),s3p(1),el2);
    double el3 = (1.0/3.0)*exp(b-4.0*a)*(-1.0+exp(6.0*a));
    u123.set(s1(1),s2(1),s3(2),s1p(1),s2p(2),s3p(1),el3);
    u123.set(s1(1),s2(2),s3(1),s1p(1),s2p(1),s3p(2),el3);
    u123.set(s1(1),s2(2),s3(1),s1p(2),s2p(1),s3p(1),el3);
    u123.set(s1(1),s2(2),s3(2),s1p(2),s2p(1),s3p(2),el3);
    u123.set(s1(2),s2(1),s3(1),s1p(1),s2p(2),s3p(1),el3);
    u123.set(s1(2),s2(1),s3(2),s1p(1),s2p(2),s3p(2),el3);
    u123.set(s1(2),s2(1),s3(2),s1p(2),s2p(2),s3p(1),el3);
    u123.set(s1(2),s2(2),s3(1),s1p(2),s2p(1),s3p(2),el3);
    double el4 = (1.0/3.0)*exp(b-4.0*a)*(2.0+exp(6.0*a));
    u123.set(s1(1),s2(2),s3(1),s1p(1),s2p(2),s3p(1),el4);
    u123.set(s1(2),s2(1),s3(2),s1p(2),s2p(1),s3p(2),el4);
    double el5 = (1.0/6.0)*exp(-3.0*b)*(exp(4.0*(b-a))*(1.0+2.0*exp(6.0*a))-3.0);
    u123.set(s1(1),s2(1),s3(2),s1p(2),s2p(1),s3p(1),el5);
    u123.set(s1(1),s2(2),s3(2),s1p(2),s2p(2),s3p(1),el5);
    u123.set(s1(2),s2(1),s3(1),s1p(1),s2p(1),s3p(2),el5);
    u123.set(s1(2),s2(2),s3(1),s1p(1),s2p(2),s3p(2),el5);
    // definition of U_123 done

    PrintData(u123);

    return ltorMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_3site getMPO3s_Uj1j2_v2(double tau, double J1, double J2, double lambda) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP1 define exact U_123 = exp(J1(S_1.S_2 + S_2.S_3) + 2*J2(S_1.S_3))
    double a = -tau*J1/8.0;
    double b = -tau*J2/4.0;
    double el_E0 = lambda*tau;
    std::cout<<"Lambda: "<< lambda << std::endl;
    ITensor u123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
    
    // Diagonal elements
    double el1 = exp(2.0*a + b)*exp(el_E0);
    u123.set(s1(1),s2(1),s3(1),s1p(1),s2p(1),s3p(1),el1);
    u123.set(s1(2),s2(2),s3(2),s1p(2),s2p(2),s3p(2),el1);

    double el2 = (1.0/6.0)*exp(-3.0*b)*(exp(4.0*(b-a))*(1.0+2.0*exp(6.0*a))+3.0)*exp(el_E0);
    u123.set(s1(1),s2(1),s3(2),s1p(1),s2p(1),s3p(2),el2);
    u123.set(s1(1),s2(2),s3(2),s1p(1),s2p(2),s3p(2),el2);
    u123.set(s1(2),s2(1),s3(1),s1p(2),s2p(1),s3p(1),el2);
    u123.set(s1(2),s2(2),s3(1),s1p(2),s2p(2),s3p(1),el2);
    
    double el4 = (1.0/3.0)*exp(b-4.0*a)*(2.0+exp(6.0*a))*exp(el_E0);
    u123.set(s1(1),s2(2),s3(1),s1p(1),s2p(2),s3p(1),el4);
    u123.set(s1(2),s2(1),s3(2),s1p(2),s2p(1),s3p(2),el4);

    // Off-Diagonal elements
    double el3 = (1.0/3.0)*exp(b-4.0*a)*(-1.0+exp(6.0*a));
    u123.set(s1(1),s2(1),s3(2),s1p(1),s2p(2),s3p(1),el3);
    u123.set(s1(1),s2(2),s3(1),s1p(1),s2p(1),s3p(2),el3);
    u123.set(s1(1),s2(2),s3(1),s1p(2),s2p(1),s3p(1),el3);
    u123.set(s1(1),s2(2),s3(2),s1p(2),s2p(1),s3p(2),el3);
    u123.set(s1(2),s2(1),s3(1),s1p(1),s2p(2),s3p(1),el3);
    u123.set(s1(2),s2(1),s3(2),s1p(1),s2p(2),s3p(2),el3);
    u123.set(s1(2),s2(1),s3(2),s1p(2),s2p(2),s3p(1),el3);
    u123.set(s1(2),s2(2),s3(1),s1p(2),s2p(1),s3p(2),el3);
    
    double el5 = (1.0/6.0)*exp(-3.0*b)*(exp(4.0*(b-a))*(1.0+2.0*exp(6.0*a))-3.0);
    u123.set(s1(1),s2(1),s3(2),s1p(2),s2p(1),s3p(1),el5);
    u123.set(s1(1),s2(2),s3(2),s1p(2),s2p(2),s3p(1),el5);
    u123.set(s1(2),s2(1),s3(1),s1p(1),s2p(1),s3p(2),el5);
    u123.set(s1(2),s2(2),s3(1),s1p(1),s2p(2),s3p(2),el5);
    
    return ltorMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_3site getMPO3s_Uladder(double tau, double J, double Jp) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP1 define exact U_123 = exp(J1(S_1.S_2 + S_2.S_3) + 2*J2(S_1.S_3))
    ITensor h123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
    h123 += J*( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
            + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) )* delta(s3,s3p);
    h123 += Jp*delta(s1,s1p)*( SU2_getSpinOp(SU2_S_Z, s2) * SU2_getSpinOp(SU2_S_Z, s3)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s2) * SU2_getSpinOp(SU2_S_M, s3)
            + SU2_getSpinOp(SU2_S_M, s2) * SU2_getSpinOp(SU2_S_P, s3) ) );
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    // definition of U_123 done

    PrintData(u123);

    return ltorMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_3site getMPO3s_Uladder_v2(double tau, double J, double Jp) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP1 define exact U_123 = exp(J1(S_1.S_2 + S_2.S_3) + 2*J2(S_1.S_3))
    ITensor h123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
    h123 += J*( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
            + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) )* delta(s3,s3p);
    h123 += Jp*delta(s1,s1p)*( SU2_getSpinOp(SU2_S_Z, s2) * SU2_getSpinOp(SU2_S_Z, s3)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s2) * SU2_getSpinOp(SU2_S_M, s3)
            + SU2_getSpinOp(SU2_S_M, s2) * SU2_getSpinOp(SU2_S_P, s3) ) );
    
    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI); 
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_3site getMPO3s_Ising_v2(double tau, double J, double h) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    // STEP1 define exact U_123 = exp(-J(Sz_1.Sz_2 + Sz_2.Sz_3) - h(Sx_1+Sx_2+Sx_3))
    ITensor h123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
    h123 += -J*( 2*SU2_getSpinOp(SU2_S_Z, s1) * 2*SU2_getSpinOp(SU2_S_Z, s2))* delta(s3,s3p);
    h123 += -J*delta(s1,s1p)*( 2*SU2_getSpinOp(SU2_S_Z, s2) * 2*SU2_getSpinOp(SU2_S_Z, s3) );
    h123 += -h*( ((SU2_getSpinOp(SU2_S_P, s1) + SU2_getSpinOp(SU2_S_M, s1))*delta(s2,s2p))*delta(s3,s3p)
        + (delta(s1,s1p)*(SU2_getSpinOp(SU2_S_P, s2)+SU2_getSpinOp(SU2_S_M, s2)))*delta(s3,s3p)
        + delta(s1,s1p)*(delta(s2,s2p)*(SU2_getSpinOp(SU2_S_P, s3)+SU2_getSpinOp(SU2_S_M, s3))) );

    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI); 
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_3site getMPO3s_Ising_2site(double tau, double J, double h) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP1 define exact U_123 = exp(-J(Sz_1.Sz_2 + Sz_2.Sz_3) - h(Sx_1+Sx_2+Sx_3))
    ITensor h123 = ITensor(s1,s2,s1p,s2p);
    h123 += -J*( 2*SU2_getSpinOp(SU2_S_Z, s1) * 2*SU2_getSpinOp(SU2_S_Z, s2) );
    h123 += -h*( (SU2_getSpinOp(SU2_S_P, s1) + SU2_getSpinOp(SU2_S_M, s1))*delta(s2,s2p)
        + delta(s1,s1p)*(SU2_getSpinOp(SU2_S_P, s2)+SU2_getSpinOp(SU2_S_M, s2)) );

    auto cmbI = combiner(s1,s2);
    h123 = (cmbI * h123 ) * prime(cmbI);
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return ltorMPO2StoMPO3Sdecomp(u123, s1, s2);
}
// ----- END Trotter gates (2Site, 3site, ...) MPOs -------------------

// ----- Model Definitions --------------------------------------------
void getModel_J1J2(nlohmann::json & json_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_J2 = json_model["J2"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    // time step
    double arg_tau = json_model["tau"].get<double>();
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    gateMPO.push_back(
        getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda)
        );

    ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );

    if (arg_fuGateSeq == "SYM1") {
        gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
        };
    } 
    else if (arg_fuGateSeq == "SYM2") {
        gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (1 BC ABCD) 
            
            {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (2 BC BADC)
            {"D", "C", "A", "B"}, {"A", "B", "D", "C"}, // (2 AD BADC)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (3 BC CDAB)

            {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (4 BC DCBA)        
            {"D", "C", "A", "B"}, {"A", "B", "D", "C"}  // (4 AD DCBA)
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},        
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3}
        };
    }
    else if (arg_fuGateSeq == "SYM3") {
        gates = {
            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},
            {"D", "C", "A", "B"},
            {"B", "A", "C", "D"},

            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"},
            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},

            {"D", "C", "A", "B"},
            {"B", "A", "C", "D"},
            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},

            {"C", "D", "B", "A"}, 
            {"A", "B", "D", "C"},
            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"}
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,2, 0,3, 1,0, 2,1},

            {3,0, 2,3, 1,2, 0,1},
            {3,2, 0,3, 1,0, 2,1},
            {3,2, 0,3, 1,0, 2,1},
            {3,0, 2,3, 1,2, 0,1},

            {1,0, 2,1, 3,2, 0,3},
            {1,2, 0,1, 3,0, 2,3},
            {1,2, 0,1, 3,0, 2,3}, 
            {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3},
            {1,0, 2,1, 3,2, 0,3},
            {1,0, 2,1, 3,2, 0,3},
            {1,2, 0,1, 3,0, 2,3}
        };
    } 
    else if (arg_fuGateSeq == "SYM4") {
        gates = {
            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"},

            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},

            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},

            {"D", "C", "A", "B"}, 
            {"B", "A", "C", "D"}
        };

        gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}
        };
    } 
    else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_NNHLadder(nlohmann::json & json_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_alpha = json_model["alpha"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    // time step
    double arg_tau = json_model["tau"].get<double>();
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    if (arg_fuGateSeq == "SYM1") {
        gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
        };

        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_J1) );
        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_alpha*arg_J1) );

        for (int i=0; i<8; i++) ptr_gateMPO.push_back( &(gateMPO[0]) ); 
        for (int i=0; i<8; i++) ptr_gateMPO.push_back( &(gateMPO[1]) );
    } else if (arg_fuGateSeq == "2SITE") {
        gates = {
            {"A", "B", "D", "C"},
            {"B", "A", "C", "D"}, 
            
            {"C", "D", "B", "A"}, 
            {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)

            {"A", "C", "D", "B"}, {"B", "D", "C", "A"},

            {"C", "A", "B", "D"}, {"D", "B", "A", "C"}
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1},
            {3,2, 0,3, 1,0, 2,1},

            {1,2, 0,1, 3,0, 2,3},
            {1,2, 0,1, 3,0, 2,3},

            {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0},
            
            {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0}
        };

        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, 0.0) );
        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_alpha*arg_J1, 0.0) );

        for (int i=0; i<6; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
        for (int i=0; i<2; i++) ptr_gateMPO.push_back( &(gateMPO[1]) );
    } 
    else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_Ising(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    // time step
    double arg_tau    = json_model["tau"].get<double>();
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    if (arg_fuGateSeq == "SYM1") {
        gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
        };

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
        ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
    } else if (arg_fuGateSeq == "SYM3") {
        gates = {
            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},
            {"D", "C", "A", "B"},
            {"B", "A", "C", "D"},

            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"},
            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},

            {"D", "C", "A", "B"},
            {"B", "A", "C", "D"},
            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},

            {"C", "D", "B", "A"}, 
            {"A", "B", "D", "C"},
            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"}
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,2, 0,3, 1,0, 2,1},

            {3,0, 2,3, 1,2, 0,1},
            {3,2, 0,3, 1,0, 2,1},
            {3,2, 0,3, 1,0, 2,1},
            {3,0, 2,3, 1,2, 0,1},

            {1,0, 2,1, 3,2, 0,3},
            {1,2, 0,1, 3,0, 2,3},
            {1,2, 0,1, 3,0, 2,3}, 
            {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3},
            {1,0, 2,1, 3,2, 0,3},
            {1,0, 2,1, 3,2, 0,3},
            {1,2, 0,1, 3,0, 2,3}
        };

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
        ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
    } else if (arg_fuGateSeq == "SYM4") {
        gates = {
            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"},

            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},

            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},

            {"D", "C", "A", "B"}, 
            {"B", "A", "C", "D"}
        };

        gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}
        };
        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
        ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
    } else if (arg_fuGateSeq == "2SITE") {
        gates = {
            {"A", "B", "D", "C"},
            {"B", "A", "C", "D"}, 
            
            {"C", "D", "B", "A"}, 
            {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)

            {"A", "C", "D", "B"}, {"B", "D", "C", "A"},

            {"C", "A", "B", "D"}, {"D", "B", "A", "C"}
        };

        gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1},
            {3,2, 0,3, 1,0, 2,1},

            {1,2, 0,1, 3,0, 2,3},
            {1,2, 0,1, 3,0, 2,3},

            {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0},
            
            {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0}
        };

        gateMPO.push_back( getMPO3s_Ising_2site(arg_tau, arg_J1, arg_h) );
        ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    std::string arg_modelType = json_model["type"].get<std::string>(); 

    if(arg_modelType == "J1J2") {
        getModel_J1J2(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "NNHLadder") {
        getModel_NNHLadder(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "Ising") {
        getModel_Ising(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }
}
// ----- END Model Definitions ----------------------------------------