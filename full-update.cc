#include "full-update.h"

using namespace itensor;

MPO_3site getMPO3s_Id(int physDim) {
	MPO_3site mpo3s;
	
	// Define physical indices
	mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
	mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
	mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

	//create a lambda function
	//which returns the square of its argument
	auto sqrt_T = [](double r) { return sqrt(r); };

    ITensor idpI1(mpo3s.Is1,prime(mpo3s.Is1,1));
    ITensor idpI2(mpo3s.Is2,prime(mpo3s.Is2,1));
    ITensor idpI3(mpo3s.Is3,prime(mpo3s.Is3,1));
    for (int i=1;i<=physDim;i++) {
        idpI1.set(mpo3s.Is1(i),prime(mpo3s.Is1,1)(i),1.0);
        idpI2.set(mpo3s.Is2(i),prime(mpo3s.Is2,1)(i),1.0);
        idpI3.set(mpo3s.Is3(i),prime(mpo3s.Is3,1)(i),1.0);
    }

    ITensor id3s = idpI1*idpI2*idpI3;

    /*
     *  s1'                    s2' s3' 
     *   |                      |  |
     *  |H1|--a1--<SV_12>--a2--|temp|
     *   |                      |  |
     *  s1                     s2  s3
     *
     */
    mpo3s.H1 = ITensor(mpo3s.Is1,prime(mpo3s.Is1));
    ITensor SV_12,temp;
    svd(id3s,mpo3s.H1,SV_12,temp);

    PrintData(mpo3s.H1);
    PrintData(SV_12);
    Print(temp);

    Index a1 = commonIndex(mpo3s.H1,SV_12);
    Index a2 = commonIndex(SV_12,temp);

    // Define aux indices linking the on-site MPOs
	Index iMPO3s12(TAG_MPO3S_12LINK,a1.m(),MPOLINK);

	/*
	 * Split obtained SVD values symmetricaly and absorb to obtain
	 * final tensor H1 and intermediate tensor temp
	 *
     *  s1'                                     s2' s3' 
     *   |                                       |  |
     *  |H1|--a1--<SV_12>^1/2--<SV_12>^1/2--a2--|temp|
     *   |                                       |  |
     *  s1                                      s2  s3
     *
     */
    SV_12.apply(sqrt_T);
    mpo3s.H1 = ( mpo3s.H1 * SV_12 )*delta(a2,iMPO3s12);
    temp = ( temp * SV_12 )*delta(a1,iMPO3s12);
    Print(mpo3s.H1);
    Print(temp);
    
    /*
     *  s1'    s2'                    s3' 
     *   |     |                      |
     *  |H1|--|H2|--a3--<SV_23>--a4--|H3|
     *   |     |                      |
     *  s1     s2                     s3
     *
     */
	mpo3s.H2 = ITensor(mpo3s.Is2,prime(mpo3s.Is2,1),iMPO3s12);
	ITensor SV_23;
    svd(temp,mpo3s.H2,SV_23,mpo3s.H3);

    Print(mpo3s.H2);
    PrintData(SV_23);
    Print(mpo3s.H3);

	Index a3 = commonIndex(mpo3s.H2,SV_23);
	Index a4 = commonIndex(SV_23,mpo3s.H3);
  
	/*
	 *cSplit obtained SVD values symmetricaly and absorb to obtain
	 * final tensor H1 and H3
	 *
     *  s1'    s2'                                     s3' 
     *   |     |                                       |
     *  |H1|--|H2|--a3--<SV_23>^1/2--<SV_23>^1/2--a4--|H3|
     *   |     |                                       |
     *  s1     s2                                      s3
     *
     */
	SV_23.apply(sqrt_T);
	Index iMPO3s23(TAG_MPO3S_23LINK,a3.m(),MPOLINK);
	mpo3s.H2 = (mpo3s.H2 * SV_23)*delta(a4,iMPO3s23);
	mpo3s.H3 = (mpo3s.H3 * SV_23)*delta(a3,iMPO3s23);

	PrintData(mpo3s.H1);
	PrintData(mpo3s.H2);
	PrintData(mpo3s.H3);

	return mpo3s;
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
    //PrintData(SV_12_LR);

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
    
    // apply(pow_T); // x^(2/3*1/2) = x^1/3
	pw = 1.0/2.0;
	SV_12_LR.apply(pow_T);

	O1_LR = O1_LR * SV_12_LR; // --a2_LR 

	// second SVD
    O2_LR = ITensor(s2,s2p,a1_LR);
    svd(tempLR,O2_LR,SV_23_LR,O3_LR);
    /*
     *  s1'                    s2'                    s3' 
     *   |                      |                      |
     *  |H1|--a2 		   a1--|H2|--a3--<SV_23>--a4--|H3|
     *   |                      |                      |
     *  s1                     s2                     s3
     *
     */
    // PrintData(SV_23_LR);

    Index a3_LR = commonIndex(SV_23_LR,O2_LR);
    Index a4_LR = commonIndex(SV_23_LR,O3_LR);

	pw = 1.0/2.0;
	SV_23_LR.apply(pow_T);

	O2_LR = O2_LR * SV_23_LR; // --a4_LR
	O3_LR = O3_LR * SV_23_LR; // --a3_LR

	// double m1, m2, m3;
	// double sign1, sign2, sign3;
	// double m = 0.;
	// double sign = 0.;
 //    auto max_m = [&m, &sign](double d) {
 //        if(std::abs(d) > m) {
 //        	sign = (d > 0) ? 1 : ((d < 0) ? -1 : 0);
 //         	m = std::abs(d);
 //        }
 //    };

 //    mpo3s.H1.visit(max_m);
 //    sign1 = sign;
 //    m1 = m*sign;
 //    m = 0.;
 //    mpo3s.H2.visit(max_m);
 //    m2 = m*sign;
 //    m = 0.;
 //    mpo3s.H3.visit(max_m);
 //    m3 = m*sign;

 //    std::cout <<"m1: "<< m1 <<" m2: "<< m2 <<" m3: "<< m3 << std::endl;

	// if ( m1*m2*m3 > 0 ) {
	// 	mpo3s.H1 /= m1;
	// 	mpo3s.H2 /= m2;
	// 	mpo3s.H3 /= m3;	
	// }
	 
	PrintData(O1_LR);
	PrintData(O2_LR);
	PrintData(O3_LR);

	MPO_3site mpo3s;
	// Define physical indices
	mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
	mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
	mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

	// Define aux indices linking the on-site MPOs
	mpo3s.a12 = Index(TAG_MPO3S_12LINK,a1_LR.m(),MPOLINK);
	mpo3s.a23 = Index(TAG_MPO3S_23LINK,a3_LR.m(),MPOLINK);

	/*
     *  s1'                    s2'                    s3' 
     *   |                      |                      |
     *  |H1|--a2 		   a1--|H2|--a4           a3--|H3|
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

MPO_3site getMPO3s_Uj1j2_v2(double tau, double J1, double J2) {
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
	//PrintData(u123);

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

    // PrintData(SV1t);
    // PrintData(SV3t);

    pw = 0.5;
    SV1t.apply(pow_T);
	SV3t.apply(pow_T);

	O1 = O1*SV1t;
	O3 = O3*SV3t;
	O2 = SV1t * (O2Lt*delta(a1,a2)) * O2Rt * SV3t;

	MPO_3site mpo3s;
	// Define physical indices
	mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
	mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
	mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

	// Define aux indices linking the on-site MPOs
	mpo3s.a12 = Index(TAG_MPO3S_12LINK,a1L.m(),MPOLINK);
	mpo3s.a23 = Index(TAG_MPO3S_23LINK,a2R.m(),MPOLINK);

	mpo3s.H1 = ((O1*delta(s1,mpo3s.Is1)) *delta(s1p,prime(mpo3s.Is1)))
		*delta(a2L,mpo3s.a12);
	mpo3s.H2 = ( ((O2*delta(s2,mpo3s.Is2)) *delta(s2p,prime(mpo3s.Is2)))
		*delta(a1L,mpo3s.a12)) *delta(a2R,mpo3s.a23);
	mpo3s.H3 = ((O3*delta(s3,mpo3s.Is3)) *delta(s3p,prime(mpo3s.Is3)))
		*delta(mpo3s.a23,a1R);

	// Correct signs and normalize
	// double m1, m2, m3;
	// double sign1, sign2, sign3;
	// double m = 0.;
	// double sign = 0.;
 //    auto max_m = [&m, &sign](double d) {
 //        if(std::abs(d) > m) {
 //        	sign = (d > 0) ? 1 : ((d < 0) ? -1 : 0);
 //         	m = std::abs(d);
 //        }
 //    };

 //    mpo3s.H1.visit(max_m);
 //    sign1 = sign;
 //    m1 = m*sign;
 //    m = 0.;
 //    mpo3s.H2.visit(max_m);
 //    m2 = m*sign;
 //    m = 0.;
 //    mpo3s.H3.visit(max_m);
 //    m3 = m*sign;

    //std::cout <<"m1: "<< m1 <<" m2: "<< m2 <<" m3: "<< m3 << std::endl;

	// if ( m1*m2*m3 > 0 ) {
	// 	mpo3s.H1 /= m1;
	// 	mpo3s.H2 /= m2;
	// 	mpo3s.H3 /= m3;	
	// }

	PrintData(mpo3s.H1*mpo3s.H2*mpo3s.H3);
	// PrintData(mpo3s.H1);
    // PrintData(mpo3s.H2);
    // PrintData(mpo3s.H3);

	return mpo3s;
}

void initRT(ITensor& rt, std::string INIT_METHOD) {
	if(INIT_METHOD == "RANDOM") {
		randomize(rt);
	} else if (INIT_METHOD == "DELTA") {
		// expect 2 AUXLINK indices and single MPOLINK
		Index a1 = findtype(rt.inds(),AUXLINK);
		Index a2 = ( a1.primeLevel() < IOFFSET ) ? prime(a1,IOFFSET) : prime(a1,-IOFFSET);
		Index impo = findtype(rt.inds(),MPOLINK);

		for (int i=1; i<=a1.m(); i++) {
			// for (int j=1; j<=impo.m(); j++) {
			//  	rt.set(a1(i),a2(i),impo(j),1.0);
			// }
			rt.set(a1(i),a2(i),impo(1),1.0);
		}
	}
}

ITensor getT(ITensor const& s, std::array<Index, 4> const& plToEnv, 
	ITensor const& op, std::array<const ITensor *, 4> rt, bool dbg) {  

	Index aS(noprime(findtype(s, AUXLINK)));
	Index pS(noprime(findtype(s, PHYS)));
	Index pOp = (op) ? noprime(findtype(op, PHYS)) : Index();

	// build ket part
	ITensor res = s;
	if (rt[0]) res = res * (*rt[0]);
	if (rt[2]) res = res * (*rt[2]);
	if (dbg && (rt[0] || rt[2])) Print(res);
	if (op) res = (res * delta(pS,pOp)) * op;
	if (dbg) Print(res);
	// build bra part
	if (op) res = res * prime(conj(op), MPOLINK, 4);
	if (dbg) Print(res);
	if (rt[1]) res = res * prime(conj(*rt[1]), 4);
	if (rt[3]) res = res * prime(conj(*rt[3]), 4);
	if (dbg && (rt[1] || rt[3])) Print(res);
	if (op) 
		res = res * delta(pOp, pS) * prime(conj(s), AUXLINK, 4);
	else
		res = res * prime(conj(s), AUXLINK, 4);
	if (dbg) Print(res);
	for (int i=0; i<=7; i++) {
		res.mapprime(AUXLINK, IOFFSET+i, i);
	}
	if (dbg) Print(res);

	// contract given indices into ENV compatible indices
	auto cmb = combiner(aS,prime(aS,4));
	for (int i=0; i<=3; i++) {
		if(plToEnv[i]) {
			res = res * prime(cmb,i);
			res = res * delta(plToEnv[i],commonIndex(res,prime(cmb,i)));	
			if (dbg) {
				Print(plToEnv[i]);
				Print(res);
			}
		}
	}

	return res;
}

void fullUpdate(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl, int maxAltLstSqrIter,
	bool dbg) {

	//if(dbg) std::cout << ctmEnv;
	std::cout<< uJ1J2;
	PrintData(uJ1J2.H1);
	PrintData(uJ1J2.H2);
	PrintData(uJ1J2.H3);

	// map MPOs
	ITensor dummyMPO = ITensor();
	std::array<const ITensor *, 4> mpo({&uJ1J2.H1, &uJ1J2.H2, &uJ1J2.H3, &dummyMPO});

	// find index(int) of on-site tensors within CtmEnv
	std::vector<int> si;
	for (int i=0; i<=3; i++) {
		si.push_back(std::distance(ctmEnv.siteIds.begin(),
				std::find(std::begin(ctmEnv.siteIds), 
					std::end(ctmEnv.siteIds), tn[i])));
	}
	if(dbg) {
		std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
		for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] << std::endl; }
	}

	// read off auxiliary indices of the cluster sites
	std::array<Index, 4> aux({noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK))});

	// prepare map from on-site tensor aux-indices to half row/column T
	// environment tensors
	std::array<const std::vector<ITensor> * const, 4> iToT(
		{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

	// prepare map from on-site tensor aux-indices pair to half corner T-C-T
	// environment tensors
	const std::map<int, const std::vector<ITensor> * const > iToC(
		{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
		{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
		{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
		{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

	// precompute 4 (proto)corners of 2x2 environment
	std::vector<ITensor> pc(4);

	// for every on-site tensor point from primeLevel(index) to ENV index
	// eg. I_XH or I_XV (with appropriate prime level)
	std::array< std::array<Index, 4>, 4> iToE;

	std::vector<int> fullset({0,1,2,3});
	for (int s=0; s<=3; s++) {
		// Find of site 0 through 3 which are connected to environment
		std::vector<int> connected({pl[s*2], pl[s*2+1]});
		std::sort(connected.begin(), connected.end());
		std::vector<int> iToEnv;
		std::set_difference(fullset.begin(), fullset.end(), 
			connected.begin(), connected.end(), 
            std::inserter(iToEnv, iToEnv.begin()));
		iToEnv.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C env tensor
		if(dbg) { 
			std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
				<< tn[s] << std::endl;
			std::cout << iToEnv[0] <<" "<< iToEnv[1] << " iToC: " << iToEnv[2] << std::endl;
		}

		// Assign indices by which site is connected to ENV
		if( findtype( (*iToT.at(iToEnv[0]))[si[s]], HSLINK ) ) {
			iToE[s][iToEnv[0]] = findtype( (*iToT.at(iToEnv[0]))[si[s]], HSLINK );
			iToE[s][iToEnv[1]] = findtype( (*iToT.at(iToEnv[1]))[si[s]], VSLINK );
		} else {
			iToE[s][iToEnv[0]] = findtype( (*iToT.at(iToEnv[0]))[si[s]], VSLINK );
			iToE[s][iToEnv[1]] = findtype( (*iToT.at(iToEnv[1]))[si[s]], HSLINK );
		}

		pc[s] = (*iToT.at(iToEnv[0]))[si[s]]*(*iToC.at(iToEnv[2]))[si[s]]*
			(*iToT.at(iToEnv[1]))[si[s]];
		if(dbg) Print(pc[s]);
		// set primeLevel of ENV indices between T's to 0
		pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
	}

	if(dbg) {
		for(int i=0; i<=3; i++) {
			std::cout <<"Site: "<< tn[i] <<" ";
			for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
			std::cout << std::endl;
		}
	}

	std::vector<ITensor> rt(4); // reduction tensors
	rt[0] = ITensor(prime(aux[0],pl[1]), uJ1J2.a12, prime(aux[0],pl[1]+IOFFSET));
	rt[1] = ITensor(prime(aux[1],pl[2]), uJ1J2.a12, prime(aux[1],pl[2]+IOFFSET));
	rt[2] = ITensor(prime(aux[1],pl[3]), uJ1J2.a23, prime(aux[1],pl[3]+IOFFSET));
	rt[3] = ITensor(prime(aux[2],pl[4]), uJ1J2.a23, prime(aux[2],pl[4]+IOFFSET));
	for (int i=0; i<=3; i++) {
		initRT(rt[i],"DELTA");
		PrintData(rt[i]);
	}

	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};
	int altlstsquares_iter = 0;
	bool converged = false;
	std::vector<Cplx> overlaps;
	std::vector<double> max_niso;
	while (not converged) {
		
		for (int r=0; r<=3; r++) {
			// define M,K construction order (defined 
			// in *.h file in terms of indices of tn)
			std::array<int,4> ord = ORD[r];
			if(dbg) {
				std::cout <<"Order for rt["<< r <<"]: ("<< ORD_DIR[r] <<") ";
				for(int o=0; o<=3; o++) {std::cout << tn[ord[o]] <<" ";}
				std::cout << std::endl;
			}

			if(dbg) std::cout <<"COMPUTING Matrix M"<< std::endl;
			ITensor temp, deltaKet, deltaBra, M(1.0);
			std::array<const itensor::ITensor *, 4> rtp;
			for(int o=0; o<=3; o++) {

				for (int i=0; i<=3; i++) 
					rtp[i] = (RTPM[r][o][i] >= 0) ? &rt[RTPM[r][o][i]] : NULL;
			
				if(dbg) {
					std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
						<< tn[ord[o]] << std::endl;
					for(int i=0; i<=3; i++) {std::cout << RTPM[r][o][i] <<" ";}
					std::cout << std::endl;
				}

				temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		 			*mpo[ord[o]], rtp, false);
				if(dbg) Print(temp);
				if (o<3) {
					deltaKet = delta(prime(aux[ord[o]],pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[(o+1)%4]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					deltaBra = prime(deltaKet,4);
					temp = (temp * deltaBra) * deltaKet;
				} else {
					deltaKet = delta(prime(aux[ord[3]],IOFFSET+ pl[2*ord[3]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[0]],pl[2*ord[0]+(1-ORD_DIR[r])/2]));
					deltaBra = prime(deltaKet,4);
				}
				if(dbg) {
					Print(deltaKet);
					Print(deltaBra);
				}

				M *= temp;
				if (o==3) M = (M * deltaBra) * deltaKet;
			
				if(dbg) Print(M);
			}

			if(dbg) std::cout <<"COMPUTING Vector K"<< std::endl;
			ITensor K(1.0);
			for(int o=0; o<=3; o++) {

				for (int i=0; i<=3; i++) 
					rtp[i] = (RTPK[r][o][i] >= 0) ? &rt[RTPK[r][o][i]] : NULL;
			
				if(dbg) {
					std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
						<< tn[ord[o]] << std::endl;
					for(int i=0; i<=3; i++) {std::cout << RTPK[r][o][i] <<" ";}
					std::cout << std::endl;
				}

				temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		 			*mpo[ord[o]], rtp, false);
				if(dbg) Print(temp);
				if (o<3) {
					deltaKet = delta(prime(aux[ord[o]],pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[(o+1)%4]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					deltaBra = prime(deltaKet,4);
					temp = (temp * deltaBra) * deltaKet;
				} else {
					deltaKet = delta(prime(aux[ord[3]],pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[0]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					deltaBra = delta(prime(aux[ord[3]],IOFFSET+4+pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[0]],4+pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
				}
				if(dbg) {
					Print(deltaKet);
					Print(deltaBra);
				}

				K *= temp;
				if (o==3) K = (K * deltaBra) * deltaKet;
			
				if(dbg) Print(K);
			}

			if(dbg) std::cout <<"COMPUTING Vector Kp"<< std::endl;
			ITensor Kp(1.0);
			for(int o=0; o<=3; o++) {

				for (int i=0; i<=3; i++) 
					rtp[i] = (RTPK[r][o][i] >= 0) ? &rt[RTPK[r][o][i]] : NULL;
				//switch bra & ket reductions
				const itensor::ITensor * rtp_temp;
				rtp_temp = rtp[0];
				rtp[0] = rtp[1];
				rtp[1] = rtp_temp;
				rtp_temp = rtp[2];
				rtp[2] = rtp[3];
				rtp[3] = rtp_temp;

				if(dbg) {
					std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
						<< tn[ord[o]] << std::endl;
					for(int i=0; i<=3; i++) {std::cout << RTPK[r][o][i] <<" ";}
					std::cout << std::endl;
				}

				temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		 			*mpo[ord[o]], rtp, false);
				if(dbg) Print(temp);
				if (o<3) {
					deltaKet = delta(prime(aux[ord[o]],pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[(o+1)%4]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					deltaBra = prime(deltaKet,4);
					temp = (temp * deltaBra) * deltaKet;
				} else {
					deltaKet = delta(prime(aux[ord[3]],IOFFSET+pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[0]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					deltaBra = delta(prime(aux[ord[3]],4+pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[ord[0]],4+pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
				}
				if(dbg) {
					Print(deltaKet);
					Print(deltaBra);
				}

				Kp *= temp;
				if (o==3) Kp = (Kp * deltaBra) * deltaKet;
			
				if(dbg) Print(Kp);
			}

			PrintData(M);
			PrintData(K);
			PrintData(Kp);

			// Check Hermicity:
			//if (dbg) {
				std::cout <<"Check Hermicity of M"<< std::endl;
				std::cout <<"swapprime: "<< pl[2*ord[3]+(1+ORD_DIR[r])/2] <<" <-> "
					<< IOFFSET+pl[2*ord[3]+(1+ORD_DIR[r])/2] << std::endl;
				std::cout <<"swapprime: "<< 4+pl[2*ord[3]+(1+ORD_DIR[r])/2] <<" <-> "
					<< 4+IOFFSET+ pl[2*ord[3]+(1+ORD_DIR[r])/2] << std::endl;
				auto Mdag = mapprime(conj(M), MPOLINK,0,12, MPOLINK,4,8);
				Mdag = prime(swapPrime(swapPrime(Mdag, 
					pl[2*ord[3]+(1+ORD_DIR[r])/2],
					IOFFSET+pl[2*ord[3]+(1+ORD_DIR[r])/2]),
					4+pl[2*ord[3]+(1+ORD_DIR[r])/2],
					4+IOFFSET+ pl[2*ord[3]+(1+ORD_DIR[r])/2]),MPOLINK,-8);
				Print(Mdag);
				ITensor mherm = 0.5*(M + Mdag);
				ITensor mantiherm = 0.5*(M - Mdag);
				Print(mherm);

				m = 0.;
		        M.visit(max_m);
		        std::cout<<"M max element: "<< m <<std::endl;
				m = 0.;
		        mantiherm.visit(max_m);
		        std::cout<<"M-Mdag max element: "<< m <<std::endl;
			//}

		    auto cmbK  = combiner(K.inds()[0], K.inds()[1], K.inds()[2]);
		    auto cmbKp = combiner(Kp.inds()[0], Kp.inds()[1], Kp.inds()[2]);
		    Print(cmbK);
		    Print(cmbKp);

		    auto Mc = (cmbK * M) * cmbKp;
		    auto Kc = cmbK * K;
		    ITensor mU(combinedIndex(cmbK)),svM,mV;
			svd(Mc,mU,svM,mV,{"SVDThreshold",1.0e-1});

			Print(svM);
			for(int isv=1; isv<=svM.inds().front().m(); isv++) std::cout << 
				svM.real(svM.inds().front()(isv),svM.inds().back()(isv)) << std::endl;

		    // Set negative eigenvalues to 0
		    // define combiners:
		    // auto cmbM = combiner(Kp.inds()[0], Kp.inds()[1], Kp.inds()[2]);
		    // auto cmbI = combinedIndex(cmbM);
		    // Print(cmbI); 
		    // Print(cmbM);

		 //    auto RM = svM*mV;
			// for(int isv=1; isv<=RM.inds().front().m(); isv++) { 
			// for(int jsv=1; jsv<=RM.inds().front().m(); jsv++) { 
			// 	std::cout << isv <<" "<< jsv <<" "<< RM.real(RM.inds().front()(isv),RM.inds().back()(jsv)) << std::endl;
			// }}

		    Print((conj(mU)*Kc)*delta(svM.inds().front(),svM.inds().back())*conj(mV));
			auto RK = (conj(mU)*Kc); //*delta(svM.inds().front(),svM.inds().back())*conj(mV);
			RK = RK * delta(svM.inds().front(),svM.inds().back());
			
			Print(RK);
			for (int isv=1; isv<=RK.inds().front().m(); isv++) {
				std::cout<< isv <<" "<< RK.real(RK.inds().front()(isv)) << std::endl;
			}

			for (int isv=1; isv<=RK.inds().front().m(); isv++) {
				if (svM.real(svM.inds().front()(isv),svM.inds().back()(isv)) > 1.0e-14)  
					RK.set(RK.inds().front()(isv), 
						RK.real(RK.inds().front()(isv))/svM.real(
						svM.inds().front()(isv),svM.inds().back()(isv)) );
				else
					RK.set(RK.inds().front()(isv), 0);
			}
			
			Print(RK);
			for (int isv=1; isv<=RK.inds().front().m(); isv++) {
				std::cout<< isv <<" "<< RK.real(RK.inds().front()(isv)) << std::endl;
			}

			RK = RK * conj(mV);
			Print(RK);			
			for (int isv=1; isv<=RK.inds().front().m(); isv++) {
				std::cout<< isv <<" "<< RK.real(RK.inds().front()(isv)) << std::endl;
			}

		    // mherm = (cmbM * mherm) * prime(cmbM,4);
		    // mherm.prime(prime(cmbI,4),-3);

		    // mantiherm = (cmbM * mantiherm) * prime(cmbM,4);
		    // mantiherm.prime(prime(cmbI,4),-3);
		    
		    // Print(mherm);
		    // Print(mantiherm);

		 //    ITensor U,D;
			// diagHermitian(mherm,U,D);
			// auto indD = noprime(D.inds().front());
			// std::setprecision(std::numeric_limits<long double>::digits10 + 1);
			// std::vector<double> reg_D;
			// for (int iid=1; iid<=indD.m(); iid++) {
			// 	std::cout<< iid <<" : "<< D.cplx(indD(iid),prime(indD,1)(iid)) << std::endl;
			// 	if( D.cplx(indD(iid),prime(indD,1)(iid)).real() < 1.0e-14 ) {
			// 		break;
			// 	} else {
			// 		reg_D.push_back(D.cplx(indD(iid),prime(indD,1)(iid)).real());
			// 	}
			// }
			// PrintData(U);
			// Index irC = Index("irC",reg_D.size());
			// Index irD = Index("irD",reg_D.size());
			// D = diagTensor(reg_D, irD, prime(irD));
			// PrintData(D);

			// auto irCToCmbI = delta(irC,cmbI);
			// auto irdToIndD = delta(irD,indD);
			// U = irCToCmbI*U*irdToIndD;
			// PrintData(U);

			// mherm = dag(U)*D*prime(U);
			// mantiherm = irCToCmbI*mantiherm*prime(irCToCmbI);

			// solve M rt = K for rt
			// ITensor Mreg = mherm + mantiherm;
			// Print(mherm);
			// Print(mantiherm);
			// PrintData(Mreg);

			//Mreg.mapprime(MPOLINK,0,12, MPOLINK,4,8);
			// Mreg = prime(swapPrime(swapPrime(Mreg, 
			// 		pl[2*ord[3]+(1+ORD_DIR[r])/2],
			// 		IOFFSET+pl[2*ord[3]+(1+ORD_DIR[r])/2]),
			// 		4+pl[2*ord[3]+(1+ORD_DIR[r])/2],
			// 		4+IOFFSET+ pl[2*ord[3]+(1+ORD_DIR[r])/2]),MPOLINK,-8);
			
			// ITensor mregCheck1 = Mreg - M;
			// ITensor mregCheck2 = Mreg - Mdag;

			// m = 0.;
	  //       mregCheck1.visit(max_m);
	  //       std::cout<<"Mregcheck1 max element: "<< m <<std::endl;
			// m = 0.;
	  //       mregCheck2.visit(max_m);
	  //       std::cout<<"Mregcheck2 max element: "<< m <<std::endl;


	        // auto Kc = K*prime(prime(cmbM,4), prime(cmbI,4),-3);
	        // Kc = Kc*prime(irCToCmbI);
	        // PrintData(Kc);

			ITensor niso;
			niso = (RK * delta(combinedIndex(cmbK),combinedIndex(cmbKp)) ) * cmbK;
			// linsystem(M,K,niso,{"plDiff",4,"dbg",true});

			Print(niso);
			auto print_elem = [](double d) {
				std::setprecision(std::numeric_limits<long double>::digits10 + 1);
				std::cout<< d << std::endl;
			};
			niso.visit(print_elem);

			// niso = (niso*irCToCmbI)*cmbM;

			ITensor optCond = M*niso - prime(K,-4);
			m = 0.;
		    optCond.visit(max_m);
		    std::cout<<"optCond max element: "<< m <<std::endl;
		    
			optCond = Mdag*niso - prime(K,-4);
			m = 0.;
		    optCond.visit(max_m);
		    std::cout<<"optCond(Mdag) max element: "<< m <<std::endl;

			niso.prime(-4);
			optCond = niso*M - K;
			m = 0.;
		    optCond.visit(max_m);
		    std::cout<<"optCond max element: "<< m <<std::endl;

		    optCond = niso*Mdag - K;
			m = 0.;
		    optCond.visit(max_m);
		    std::cout<<"optCond(Mdag) max element: "<< m <<std::endl;



			m = 0.;
			niso.visit(max_m);
			max_niso.push_back(m);
			//niso = niso / m;
			//if (dbg) 
				PrintData(niso);
 			rt[r] = niso;

 		// 	if(dbg) std::cout <<"COMPUTING Vector Kp"<< std::endl;
			// ITensor Kp(1.0);
			// for(int o=0; o<=3; o++) {

			// 	for (int i=0; i<=3; i++) 
			// 		rtp[i] = (RTPK[r][o][i] >= 0) ? &rt[RTPK[r][o][i]] : NULL;
			// 	//switch bra & ket reductions
			// 	const itensor::ITensor * rtp_temp;
			// 	rtp_temp = rtp[0];
			// 	rtp[0] = rtp[1];
			// 	rtp[1] = rtp_temp;
			// 	rtp_temp = rtp[2];
			// 	rtp[2] = rtp[3];
			// 	rtp[3] = rtp_temp;

			// 	if(dbg) {
			// 		std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
			// 			<< tn[ord[o]] << std::endl;
			// 		for(int i=0; i<=3; i++) {std::cout << RTPK[r][o][i] <<" ";}
			// 		std::cout << std::endl;
			// 	}

			// 	temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		//  			*mpo[ord[o]], rtp, false);
			// 	if(dbg) Print(temp);
			// 	if (o<3) {
			// 		deltaKet = delta(prime(aux[ord[o]],pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
			// 			prime(aux[ord[(o+1)%4]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
			// 		deltaBra = prime(deltaKet,4);
			// 		temp = (temp * deltaBra) * deltaKet;
			// 	} else {
			// 		deltaKet = delta(prime(aux[ord[3]],IOFFSET+pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
			// 			prime(aux[ord[0]],pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
			// 		deltaBra = delta(prime(aux[ord[3]],4+pl[2*ord[o]+(1+ORD_DIR[r])/2]), 
			// 			prime(aux[ord[0]],4+pl[2*ord[(o+1)%4]+(1-ORD_DIR[r])/2]));
			// 	}
			// 	if(dbg) {
			// 		Print(deltaKet);
			// 		Print(deltaBra);
			// 	}

			// 	Kp *= temp;
			// 	if (o==3) Kp = (Kp * deltaBra) * deltaKet;
			
			// 	if(dbg) Print(Kp);
			// }

			// Check overlap
			ITensor tempOLP;
			tempOLP = niso*M*prime(niso,4);
			if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
			overlaps.push_back(sumelsC(tempOLP));
			tempOLP = K*prime(niso,4);
			if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
			overlaps.push_back(sumelsC(tempOLP));
			tempOLP = niso*Kp;
			if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
			overlaps.push_back(sumelsC(tempOLP));
		}

		altlstsquares_iter++;
		if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	}

	for(int i=0; i<(overlaps.size()/3); i++) {
		std::cout<<"M: "<<overlaps[i]<<" K: "<<overlaps[i+1]<<" Kp: "<<overlaps[i+2]
			<<" max_elem(niso): "<< max_niso[i] <<std::endl;
	}
}

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