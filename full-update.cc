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
	// definition of U_123 done
	//PrintData(u123);

	// double m = 0.;
	// auto max_m = [&m](double d) {
	// 	if(std::abs(d) > m) m = std::abs(d);
	// };
	// u123.visit(max_m);
	// u123 = u123 / m;

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

void initRT_basic(ITensor& rt, std::string INIT_METHOD, Args const& args) {
	if(INIT_METHOD == "RANDOM") {
		randomize(rt);
	} else if (INIT_METHOD == "DELTA") {
		// expect 2 AUXLINK indices and single MPOLINK
		Index a1 = findtype(rt.inds(),AUXLINK);
		Index a2 = ( a1.primeLevel() < IOFFSET ) ? prime(a1,IOFFSET) : prime(a1,-IOFFSET);
		Index impo = findtype(rt.inds(),MPOLINK);

		for (int i=1; i<=a1.m(); i++) {
			rt.set(a1(i),a2(i),impo(1),1.0);
		}
	} else if (INIT_METHOD == "NOISE") {
		auto fuIsoInitNoiseLevel = args.getReal("fuIsoInitNoiseLevel",1.0e-3);

		// expect 2 AUXLINK indices and single MPOLINK
		Index a1 = findtype(rt.inds(),AUXLINK);
		Index a2 = ( a1.primeLevel() < IOFFSET ) ? prime(a1,IOFFSET) : prime(a1,-IOFFSET);
		Index impo = findtype(rt.inds(),MPOLINK);

		randomize(rt);
		rt = rt * fuIsoInitNoiseLevel;
		for (int i=1; i<=a1.m(); i++) {
			rt.set(a1(i),a2(i),impo(1),1.0);
		}
	}
}

ITensor getT(ITensor const& s, std::array<Index, 4> const& plToEnv, bool dbg) {  

	return getT(s, plToEnv, ITensor(), {{NULL,NULL,NULL,NULL}}, true, dbg); 
}

ITensor getT(ITensor const& s, std::array<Index, 4> const& plToEnv, 
	ITensor const& op, std::array<const ITensor *, 4> rt, bool dbg) {  

	return getT(s, plToEnv, op, rt, true, dbg); 
}

ITensor getT(ITensor const& s, std::array<Index, 4> const& plToEnv, 
	ITensor const& op, std::array<const ITensor *, 4> rt, 
	bool pIcont, bool dbg) {  

	Index aS(noprime(findtype(s, AUXLINK)));
	Index pS(noprime(findtype(s, PHYS)));
	Index pOp = (op) ? noprime(findtype(op, PHYS)) : Index();

	// build |ket> part
	ITensor res = s;
	// apply reduction tensors if present to |ket>
	if (rt[0]) res = res * (*rt[0]);
	if (rt[2]) res = res * (*rt[2]);
	if (dbg && (rt[0] || rt[2])) Print(res);
	// apply physical operator - acting on physical index
	if (op) res = (res * delta(pS,pOp)) * op;
	if (dbg) Print(res);
	// build bra part
	// apply conjugate of physical operator
	if (!pIcont) res.prime(prime(pOp,1),1);
	if (op) res = res * prime(conj(op), MPOLINK, 4);
	if (dbg) Print(res);
	// apply reduction tensors if present to <bra| 
	if (rt[1]) res = res * prime(conj(*rt[1]), 4);
	if (rt[3]) res = res * prime(conj(*rt[3]), 4);
	if (dbg && (rt[1] || rt[3])) Print(res);
	// contract with <bra| on-site tensor
	if (op) 
		res = res * delta(pOp, pS) * prime(conj(s), AUXLINK, 4);
	else
		res = res * prime(conj(s), AUXLINK, 4);
	if (dbg) Print(res);
	// reset primeLevel of auxIndices from isometries back to primeLevel
	// of on-site indices
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

ITensor getketT(ITensor const& s, ITensor const& op, 
	std::array<const ITensor *, 2> rt, bool dbg) {  

	Index aS(noprime(findtype(s, AUXLINK)));
	Index pS(noprime(findtype(s, PHYS)));
	Index pOp = (op) ? noprime(findtype(op, PHYS)) : Index();

	// build |ket> part
	ITensor res = s;
	// apply reduction tensors if present to |ket>
	if (rt[0]) res = res * noprime(*rt[0],MPOLINK);
	if (rt[1]) res = res * noprime(*rt[1],MPOLINK);
	if (dbg && (rt[0] || rt[1])) Print(res);
	// apply physical operator - acting on physical index
	if (op) res = ((res * delta(pS,pOp)) * op) * delta(prime(pOp),pS);
	if (dbg) Print(res);
	// reset primeLevel of auxIndices from isometries back to primeLevel
	// of on-site indices
	for (int i=0; i<=3; i++) {
		res.mapprime(AUXLINK, IOFFSET+i, i);
	}
	if (dbg) Print(res);

	return res;
}

Args fullUpdate(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	std::vector< ITensor > & iso_store,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
	auto rtInitType = args.getString("fuIsoInit","DELTA");
    auto rtInitParam = args.getReal("fuIsoInitNoiseLevel",1.0e-3);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	std::cout<<"GATE: ";
	for(int i=0; i<=3; i++) {
		std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
	}
	std::cout<< std::endl;

	if(dbg && (dbgLvl >= 2)) {
		std::cout<< uJ1J2;
		PrintData(uJ1J2.H1);
		PrintData(uJ1J2.H2);
		PrintData(uJ1J2.H3);
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	// map MPOs
	ITensor dummyMPO = ITensor();
	std::array<const ITensor *, 4> mpo({&uJ1J2.H1, &uJ1J2.H2, &uJ1J2.H3, &dummyMPO});

	// find integer identifier of on-site tensors within CtmEnv
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

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux({
		noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

	std::array<Index, 4> auxRT({ aux[0], aux[1], aux[1], aux[2] });
	std::array<int, 4> plRT({ pl[1], pl[2], pl[3], pl[4] });

	std::array<Index, 4> phys({
		noprime(findtype(cls.sites.at(tn[0]), PHYS)),
		noprime(findtype(cls.sites.at(tn[1]), PHYS)),
		noprime(findtype(cls.sites.at(tn[2]), PHYS)),
		noprime(findtype(cls.sites.at(tn[3]), PHYS)) });

	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });

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
	// eg. I_XH or I_XV (with appropriate prime level). 
	std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

	// Find for site 0 through 3 which are connected to ENV
	std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of on-site tensor 
	for (int s=0; s<=3; s++) {
		// aux-indices connected to sites
		std::vector<int> connected({pl[s*2], pl[s*2+1]});
		// set_difference gives aux-indices connected to ENV
		std::sort(connected.begin(), connected.end());
		std::vector<int> tmp_iToE;
		std::set_difference(plOfSite.begin(), plOfSite.end(), 
			connected.begin(), connected.end(), 
            std::inserter(tmp_iToE, tmp_iToE.begin())); 
		tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
		if(dbg) { 
			std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
				<< tn[s] << std::endl;
			std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
		}

		// Assign indices by which site is connected to ENV
		if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
			iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
			iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
		} else {
			iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
			iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
		}

		pc[s] = (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
			(*iToT.at(tmp_iToE[1]))[si[s]];
		if(dbg) Print(pc[s]);
		// set primeLevel of ENV indices between T's to 0 to be ready for contraction
		pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
	}
	if(dbg) {
		for(int i=0; i<=3; i++) {
			std::cout <<"Site: "<< tn[i] <<" ";
			for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
			std::cout << std::endl;
		}
	}
	
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

	// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************

	// C  D
	//    |
	// A--B
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	// Decompose A tensor on which the gate is applied
	ITensor QA, tempSA, eA(prime(aux[0],pl[1]), phys[0]);
	svd(cls.sites.at(tn[0]), eA, tempSA, QA);
	Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
	eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
	QA *= delta(commonIndex(QA,tempSA), iQA);

	// Prepare corner of A
	ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
	if(dbg && (dbgLvl >=3)) Print(tempC);

	deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
	deltaBra = prime(deltaKet,4);
	tempC = (tempC * deltaBra) * deltaKet;
	if(dbg && (dbgLvl >=3)) Print(tempC);

	eRE = tempC;

	// Prepare corner of C
	tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >= 3)));
	if(dbg && (dbgLvl >=3)) Print(tempC);
	
	deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
	deltaBra = prime(deltaKet,4);
	tempC = (tempC * deltaBra) * deltaKet;
	if(dbg && (dbgLvl >=3)) Print(tempC);

	eRE = eRE * tempC;

	// Decompose D tensor on which the gate is applied
	ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
	svd(cls.sites.at(tn[2]), eD, tempSD, QD);
	Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
	eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
	QD *= delta(commonIndex(QD,tempSD), iQD);

	// Prepare corner of D
	tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
	if(dbg && (dbgLvl >=3)) Print(tempC);

	eRE = eRE * tempC;

	// Decompose B tensor on which the gate is applied
	ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	svd(cls.sites.at(tn[1]), eB, tempSB, QB);
	Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
	eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
	QB *= delta(commonIndex(QB,tempSB), iQB);

	tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
	if(dbg && (dbgLvl >=3)) Print(tempC);

	eRE = eRE * tempC;

	if(dbg && (dbgLvl >=3)) Print(eRE);
	// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
	
	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	if (symmProtoEnv) {
	// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
	auto cmbKet = combiner(iQA, iQB, iQD);
	auto cmbBra = prime(cmbKet,4);

	eRE = (eRE * cmbKet) * cmbBra;

	ITensor eRE_sym  = 0.5 * (eRE + swapPrime(eRE,0,4));
	ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE,0,4));

	m = 0.;
    eRE_sym.visit(max_m);
    diag_maxMsymLE = m;
    std::cout<<"eRE_sym max element: "<< m <<std::endl;
	m = 0.;
    eRE_asym.visit(max_m);
    diag_maxMasymLE = m;
    std::cout<<"eRE_asym max element: "<< m <<std::endl;

	diag_maxMsymFN  = norm(eRE_sym);
	diag_maxMasymFN = norm(eRE_asym);

	eRE_sym *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
	
	// ##### V3 ######################################################
	ITensor U_eRE, D_eRE;
	diagHermitian(eRE_sym, U_eRE, D_eRE);

	double msign = 1.0;
	double mval = 0.;
	std::vector<double> dM_elems;
	for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
		dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
		if (std::abs(dM_elems.back()) > mval) {
			mval = std::abs(dM_elems.back());
			msign = dM_elems.back()/mval;
		}
	}
	if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

	// Drop negative EV's
	if(dbg && (dbgLvl >= 1)) {
		std::cout<<"REFINED SPECTRUM"<< std::endl;
		std::cout<<"MAX EV: "<< mval << std::endl;
	}
	for (auto & elem : dM_elems) {
		if (elem < 0.0) {
			if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
			elem = 0.0;
		}
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
	// eRE *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet))); // 0--eRE--1
	
	// eRE_sym = conj(eRE); // 0--eRE*--1
	// eRE.mapprime(1,2);   // 0--eRE---2
	// eRE_sym = eRE_sym * eRE; // (0--eRE*--1) * (0--eRE--2) = (1--eRE^dag--0) * (0--eRE--2) 
	// eRE_sym.prime(-1);

	// ITensor U_eRE, D_eRE;
	// diagHermitian(eRE_sym, U_eRE, D_eRE);

	// std::vector<double> dM_elems;
	// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) dM_elems.push_back(
	// 		sqrt(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm))) );
	// D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
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
	// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
	// 		elem = 0.0;
	// 	} else if (elem < 0.0) {
	// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< std::fabs(elem) << std::endl;
	// 		elem = std::fabs(elem);
	// 	}
	// }
	// ##### END V6 ##################################################

	
	D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());

	eRE_sym = ((conj(U_eRE)*D_eRE)*prime(U_eRE))
		* delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));

	eRE = (eRE_sym * cmbKet) * cmbBra;
	// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
	}

	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K ***************************** 
	ITensor protoK = (eRE * eA) * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2]));
	protoK = (protoK * eB) * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4]));
	protoK = (protoK * eD);
	if(dbg && (dbgLvl >=3)) Print(protoK);

	protoK = ( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1;
	protoK = ( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2;
	protoK = ( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3;
	if(dbg && (dbgLvl >=3)) Print(protoK);

	// PROTOK - VARIANT 1
	protoK = protoK * conj(eA).prime(AUXLINK,4);
	protoK = ( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1).prime(uJ1J2.a12, 4+pl[1]);
	if(dbg && (dbgLvl >=3)) Print(protoK);

	protoK = protoK * conj(eD).prime(AUXLINK,4); 
	protoK = ( protoK * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3).prime(uJ1J2.a23, 4+pl[4]);
	if(dbg && (dbgLvl >=3)) Print(protoK);

	protoK = protoK * conj(eB).prime(AUXLINK,4);
	protoK = ( protoK * delta(opPI[1],phys[1]) ) 
		* (conj(uJ1J2.H2).prime(uJ1J2.a12, 4+pl[2])).prime(uJ1J2.a23, 4+pl[3]);
	if(dbg && (dbgLvl >=3)) Print(protoK);
	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************

	std::cout<<"eRE.scale(): "<< eRE.scale()<<" protoK.scale(): "<< protoK.scale() <<std::endl;

	// Create and initialize reduction tensors RT
	std::vector<ITensor> rt(4);
	rt[0] = ITensor(prime(aux[0],pl[1]), prime(uJ1J2.a12,pl[1]), prime(aux[0],pl[1]+IOFFSET));
	rt[1] = ITensor(prime(aux[1],pl[2]), prime(uJ1J2.a12,pl[2]), prime(aux[1],pl[2]+IOFFSET));
	rt[2] = ITensor(prime(aux[1],pl[3]), prime(uJ1J2.a23,pl[3]), prime(aux[1],pl[3]+IOFFSET));
	rt[3] = ITensor(prime(aux[2],pl[4]), prime(uJ1J2.a23,pl[4]), prime(aux[2],pl[4]+IOFFSET));
	// simple initialization routines
	if (rtInitType == "RANDOM" || rtInitType == "DELTA" || rtInitType == "NOISE") { 
		for (int i=0; i<=3; i++) {
			initRT_basic(rt[i],rtInitType,{"fuIsoInitNoiseLevel",rtInitParam});
			if(dbg && (dbgLvl >= 3)) PrintData(rt[i]);
		}
	}
	// self-consistent initialization - guess isometries between tn[0] and tn[1]
	else if (rtInitType == "LINKSVD") {
		// rt[0]--rt[1]
		ITensor ttemp = ( protoK * delta(prime(uJ1J2.a23,4+plRT[2]), prime(uJ1J2.a23,4+plRT[3])) )
			* delta(prime(auxRT[2],4+plRT[2]),prime(auxRT[3],4+plRT[3]));	
			
		auto cmb0 = combiner(prime(auxRT[0],4+plRT[0]), prime(uJ1J2.a12,4+plRT[0]));
		auto cmb1 = combiner(prime(auxRT[1],4+plRT[1]), prime(uJ1J2.a12,4+plRT[1]));

		if(dbg && (dbgLvl >= 3)) { Print(cmb0); Print(cmb1); } 

		ttemp = (ttemp * cmb0) * cmb1;

		ITensor tU(combinedIndex(cmb0)),tS,tV;
		svd(ttemp,tU,tS,tV,{"Maxm",aux[0].m()});

		if(dbg && (dbgLvl >= 2)) PrintData(tS);
			
		rt[0] = (tU*delta(commonIndex(tS,tU), 
				prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
		rt[1] = (tV*delta(commonIndex(tS,tV), 
				prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1;
		rt[0].prime(-4);
		rt[1].prime(-4);

		Print(rt[0]);
		Print(rt[1]);

		// rt[2]--rt[3]
		ttemp = ( protoK * delta(prime(uJ1J2.a12,4+plRT[0]), prime(uJ1J2.a12,4+plRT[1])) )
			* delta(prime(auxRT[0],4+plRT[0]),prime(auxRT[1],4+plRT[1]));	
			
		cmb0 = combiner(prime(auxRT[2],4+plRT[2]), prime(uJ1J2.a23,4+plRT[2]));
		cmb1 = combiner(prime(auxRT[3],4+plRT[3]), prime(uJ1J2.a23,4+plRT[3]));

		if(dbg && (dbgLvl >= 3)) { Print(cmb0); Print(cmb1); } 

		ttemp = (ttemp * cmb0) * cmb1;

		tU = ITensor(combinedIndex(cmb0));
		svd(ttemp,tU,tS,tV,{"Maxm",aux[0].m()});

		if(dbg && (dbgLvl >= 2)) PrintData(tS);

		rt[2] = (tU*delta(commonIndex(tS,tU), 
				prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
		rt[3] = (tV*delta(commonIndex(tS,tV), 
				prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1;
		rt[2].prime(-4);
		rt[3].prime(-4);
	
		Print(rt[2]);
		Print(rt[3]);
	}
	else if (rtInitType == "BIDIAG") {
		ITensor ttemp, tdelKet, tdelBra, tRT(1.0);
		std::array<const itensor::ITensor *, 4> trtp = {{NULL,NULL,NULL,NULL}};
		for(int r=0; r<=3; r+=3){
			std::array<int,4> tord = ORD[r];
			if(dbg) std::cout <<"GUESS RT "<< tn[tord[0]] <<"-"<< tn[tord[3]] << std::endl;
			for(int o=0; o<=3; o++) {
				// construct o'th corner of tmpRT
				if (o == 0 || o == 3) {
					// leave physical index uncontracted
					ttemp = pc[tord[o]] * getT(cls.sites.at(tn[tord[o]]), iToE[tord[o]], 
				 			*mpo[tord[o]], trtp, false, (dbg && (dbgLvl >= 3)) );
				} else {
					ttemp = pc[tord[o]] * getT(cls.sites.at(tn[tord[o]]), iToE[tord[o]], 
			 			*mpo[tord[o]], trtp, (dbg && (dbgLvl >= 3)) );
				}

				if (dbg && (dbgLvl >= 3)) Print(ttemp);
				if (o<3) {
					tdelKet = delta(prime(aux[tord[o]],pl[2*tord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[tord[(o+1)%4]],pl[2*tord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					tdelBra = prime(tdelKet,4);
					ttemp = (ttemp * tdelBra) * tdelKet;
				} else {
					tdelKet = ITensor();
					tdelBra = delta(prime(aux[tord[3]],4+pl[2*tord[o]+(1+ORD_DIR[r])/2]), 
						prime(aux[tord[0]],4+pl[2*tord[(o+1)%4]+(1-ORD_DIR[r])/2]));
					ttemp.mapprime(MPOLINK,0,IOFFSET);
				}
				if(dbg && (dbgLvl >= 3)) {
					Print(tdelKet);
					Print(tdelBra);
				}

				tRT *= ttemp;
				if (o==3) tRT = tRT * tdelBra;
			
				if(dbg && (dbgLvl >= 3)) Print(tRT);
			}
			if(dbg && (dbgLvl >= 3)) Print(tRT);
			
			// physical bra indices have primeLevel 1
			auto pI0 = noprime(findtype(*mpo[tord[0]], PHYS));
			auto pI3 = noprime(findtype(*mpo[tord[3]], PHYS));
			ITensor ob_contract(pI0,pI3);
			ob_contract.fill(1.0);

			tRT = tRT * prime(ob_contract,1);
			tRT.noprime(PHYS);
			if(dbg && (dbgLvl >= 3)) Print(tRT);

			// combiners for LEFT, RIGHT and PHYS
			auto cmbL = combiner(prime(aux[tord[0]],pl[2*tord[0]+(1-ORD_DIR[r])/2]),
				noprime(findtype(tRT, MPOLINK)));
			auto cmbR = combiner(prime(aux[tord[3]],pl[2*tord[3]+(1+ORD_DIR[r])/2]),
				prime(noprime(findtype(tRT, MPOLINK)),IOFFSET));
			auto cmbP = combiner(pI0,pI3);
			auto cmbIL = combinedIndex(cmbL);
			auto cmbIR = combinedIndex(cmbR);
			auto cmbIP = combinedIndex(cmbP);
			if(dbg && (dbgLvl >= 2)) { Print(cmbL); Print(cmbR); Print(cmbP); } 

			tRT = ((tRT * cmbL) * cmbR) *cmbP;

			// Perform QL
			ITensor QL, tSL, tL(cmbIL);
			svd(tRT, tL,tSL,QL);
			tL = tL * tSL;
			auto iQL = commonIndex(QL,tSL);

			// Perform QR
			ITensor QR, tSR, tR(cmbIR);
			svd(tRT, tR,tSR,QR);
			tR = tR * tSR;
			auto iQR = commonIndex(QR,tSR);

			ITensor tRL = tR * delta(cmbIL,cmbIR) * tL;
			Print(iQL);
			Print(iQR);
			Print(tRL);

			ITensor tU(combinedIndex(iQL)),tS,tV;
			svd(tRL,tU,tS,tV,{"Maxm",aux[0].m()});

			if(dbg && (dbgLvl >= 2)) PrintData(tS);

			// Regularize and invert singular value matrix tS
			std::vector<double> elems_regInvSvM;
			for (int isv=1; isv<=tS.inds().front().m(); isv++) {
				if ( tS.real(tS.inds().front()(isv),tS.inds().back()(isv))/
					 tS.real(tS.inds().front()(1),tS.inds().back()(1))  > svd_cutoff) {  
					elems_regInvSvM.push_back(1.0/sqrt(tS.real(tS.inds().front()(isv),
						tS.inds().back()(isv))) );
				} else
					elems_regInvSvM.push_back(0.0);
			}
			auto regInvSvM = diagTensor(elems_regInvSvM, tS.inds().front(),tS.inds().back());
			PrintData(regInvSvM);

			rt[r+(1+ORD_DIR[r])/2-r/3] = ( (tL * conj(tU) * regInvSvM) * 
				delta( commonIndex(tS,tV), prime(findtype(cmbL, AUXLINK),IOFFSET) ) 	
				) * cmbL;

			rt[r+(1-ORD_DIR[r])/2-r/3] = ( (tR * conj(tV) * regInvSvM) * 
				delta( commonIndex(tS,tU), prime(findtype(cmbR, AUXLINK),IOFFSET) )
				) * cmbR;
			rt[r+(1-ORD_DIR[r])/2-r/3].prime(MPOLINK,-IOFFSET);

			// rt[r+(1+ORD_DIR[r])/2-r/3] = ((tU*tS)*delta(commonIndex(tS,tV), 
			// 	prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
			// rt[r+(1-ORD_DIR[r])/2-r/3] = (((tV*tS)*delta(commonIndex(tS,tU), 
			// 	prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1).prime(MPOLINK,-IOFFSET);
			
			// rt[r+(1+ORD_DIR[r])/2-r/3] = (tU*delta(commonIndex(tS,tU), 
			// 	prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
			// rt[r+(1-ORD_DIR[r])/2-r/3] = ((tV*delta(commonIndex(tS,tV), 
			// 	prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1).prime(MPOLINK,-IOFFSET);
			
			if(dbg && (dbgLvl >= 1)) {
				std::cout<< r+(1-ORD_DIR[r])/2-r/3 <<" ";
				PrintData(rt[r+(1-ORD_DIR[r])/2-r/3]);
				std::cout<< r+(1+ORD_DIR[r])/2-r/3 <<" ";
				PrintData(rt[r+(1+ORD_DIR[r])/2-r/3]);
			}



			tRT = ITensor(1.0);
		}
	}
	else if (rtInitType == "REUSE") {
            std::cout << "REUSING ISO: ";
            if (iso_store[0]) {
                    rt[0] = iso_store[0];
                    std::cout << " 0 ";
            } else {
                    rt[0] = ITensor(prime(aux[0],pl[1]), prime(uJ1J2.a12,pl[1]), prime(aux[0],pl[1]+IOFFSET));
                    initRT_basic(rt[0],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[1]) {
                    rt[1] = iso_store[1];
                    std::cout << " 1 ";
            } else {
                    rt[1] = ITensor(prime(aux[1],pl[2]), prime(uJ1J2.a12,pl[2]), prime(aux[1],pl[2]+IOFFSET));
                    initRT_basic(rt[1],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[2]) {
                    rt[2] = iso_store[2];
                    std::cout << " 2 ";
            } else {
                    rt[2] = ITensor(prime(aux[1],pl[3]), prime(uJ1J2.a23,pl[3]), prime(aux[1],pl[3]+IOFFSET));
                    initRT_basic(rt[2],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[3]) {
                    rt[3] = iso_store[3];
                    std::cout << " 3" << std::endl;
            } else {
                    rt[3] = ITensor(prime(aux[2],pl[4]), prime(uJ1J2.a23,pl[4]), prime(aux[2],pl[4]+IOFFSET));
                    initRT_basic(rt[3],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
                    std::cout << std::endl;
            }
    }
	else {
		std::cout<<"Unsupported fu-isometry initialization: "<< rtInitType << std::endl;
		exit(EXIT_FAILURE);
	}
	// exit(EXIT_FAILURE);

	// Prepare Alternating Least Squares to maximize the overlap
	auto print_elem = [](double d) {
		std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		std::cout<< d << std::endl;
	};

	int r, altlstsquares_iter = 0;
	bool converged = false;
	std::vector<double> overlaps;
	std::vector<double> rt_diffs;
	//int min_indexCutoff = cls.auxBondDim*cls.auxBondDim*uJ1J2.a12.m();
	double minGapDisc = 100.0; // in logscale
	double minEvKept  = svd_cutoff;
	//double maxEvDisc  = 0.0;

	ITensor dbg_D, dbg_svM;
	while (not converged) {
		
		for (int i_rt=0; i_rt<=3; i_rt++) {
			//r = ((altlstsquares_iter % 2) == 0) ? i_rt : 3-i_rt;
			r = i_rt;

			// define M,K construction order (defined 
			// in *.h file in terms of indices of tn)
			std::array<int,3> ord = ORD_R[i_rt];
			if(dbg && (dbgLvl >= 3)) {
				std::cout <<"Order for rt["<< i_rt <<"]: ("<< ORD_DIR[i_rt] <<") ";
				for(int o=0; o<=2; o++) {std::cout << tn[ord[o]] <<" ";}
				std::cout << std::endl;
			}

			// construct matrix M, which is defined as <psi~|psi~> = rt[r]^dag * M * rt[r]	
		    if(dbg && (dbgLvl >= 3)) std::cout <<"COMPUTING Matrix M"<< std::endl;
			std::array<const itensor::ITensor *, 2> rtp; // holds reduction tensor pointers => rtp
			std::array<const itensor::ITensor *, 3> QS({ &eA, &eB, &eD });

			// BRA
			int shift = i_rt / 2;

			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][0][i] >= 0) ? &rt[RTPM_R[i_rt][0][i]] : NULL;
			ITensor M = eRE * getketT(*QS[ord[0]],*mpo[ord[0]], rtp, (dbg && (dbgLvl >= 3)));
			M *= delta(	prime(aux[ord[0]],pl[2*ord[0]+(1-ORD_DIR[i_rt])/2]),
			  			prime(auxRT[i_rt],IOFFSET+plRT[i_rt]) );		
			// Print(delta(prime(aux[ord[0]],pl[2*ord[0]+(1-ORD_DIR[i_rt])/2]),
			//   			prime(auxRT[i_rt],IOFFSET+plRT[i_rt]) ));
			if(dbg && (dbgLvl >= 3)) Print(M);

			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][1][i] >= 0) ? &rt[RTPM_R[i_rt][1][i]] : NULL;
			M = M * (getketT(*QS[ord[1]],*mpo[ord[1]], rtp, (dbg && (dbgLvl >= 3)))
				* delta(prime(aux[A_R[shift][0]],pl[PL_R[shift][0]]),
						prime(aux[A_R[shift][1]],pl[PL_R[shift][1]]) ));
			// Print(delta(prime(aux[A_R[shift][0]],pl[PL_R[shift][0]]),
			// 			prime(aux[A_R[shift][1]],pl[PL_R[shift][1]])) );
			if(dbg && (dbgLvl >= 3)) Print(M);

			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][2][i] >= 0) ? &rt[RTPM_R[i_rt][2][i]] : NULL;
			M = M * getketT(*QS[ord[2]],*mpo[ord[2]], rtp, (dbg && (dbgLvl >= 3)));
			M.prime(MPOLINK,plRT[i_rt]+IOFFSET);
			if(dbg && (dbgLvl >= 3)) Print(M);

			// KET
			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][0][i] >= 0) ? &rt[RTPM_R[i_rt][0][i]] : NULL;
			M = M * prime(conj(getketT(*QS[ord[0]],*mpo[ord[0]], rtp, (dbg && (dbgLvl >= 3)))), AUXLINK, 4);
			M *= delta( prime(aux[ord[0]],4+pl[2*ord[0]+(1-ORD_DIR[i_rt])/2]),
						prime(auxRT[i_rt],4+IOFFSET+plRT[i_rt]));
			// Print(delta(prime(aux[ord[0]],4+pl[2*ord[0]+(1-ORD_DIR[i_rt])/2]),
			// 			prime(auxRT[i_rt],4+IOFFSET+plRT[i_rt])));
			if(dbg && (dbgLvl >= 3)) Print(M);

			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][1][i] >= 0) ? &rt[RTPM_R[i_rt][1][i]] : NULL;
			M = M * (prime(conj(getketT(*QS[ord[1]],*mpo[ord[1]], rtp, (dbg && (dbgLvl >= 3)))), AUXLINK, 4)
				* delta(prime(aux[A_R[shift][0]],4+pl[PL_R[shift][0]]), 
			 	       	prime(aux[A_R[shift][1]],4+pl[PL_R[shift][1]]) ));
			// Print(delta(prime(aux[A_R[shift][0]],4+pl[PL_R[shift][0]]), 
			//  	       	prime(aux[A_R[shift][1]],4+pl[PL_R[shift][1]]))  );
			if(dbg && (dbgLvl >= 3)) Print(M);

			for (int i=0; i<2; i++) rtp[i] = (RTPM_R[i_rt][2][i] >= 0) ? &rt[RTPM_R[i_rt][2][i]] : NULL;
			M = M * prime(conj(getketT(*QS[ord[2]],*mpo[ord[2]], rtp, (dbg && (dbgLvl >= 3)))), AUXLINK, 4);
			
			M.mapprime(MPOLINK,0,4+plRT[i_rt], MPOLINK,plRT[i_rt]+IOFFSET,plRT[i_rt]);
			if(dbg && (dbgLvl >= 3)) Print(M);

			// construct vector K, which is defined as <psi~|psi'> = rt[r]^dag * K
			if(dbg && (dbgLvl >= 3)) std::cout <<"COMPUTING Vector K"<< std::endl;
			
			int dir        = ORD_DIR[r];
			int current_rt = (i_rt+dir+4)%4;
			ITensor K = protoK * conj(rt[current_rt]).prime(4);
			K *= delta(prime(auxRT[current_rt],100+4+plRT[current_rt]), prime(auxRT[i_rt],100+4+plRT[i_rt]));
			if(dbg && (dbgLvl >= 3)) Print(K);

			current_rt = (current_rt+dir+4)%4;
			int next_rt    = (current_rt+dir+4)%4;
			K = K * conj(rt[current_rt]).prime(4);
			K *= delta(prime(auxRT[current_rt],100+4+plRT[current_rt]), 
				prime(auxRT[next_rt],100+4+plRT[next_rt]));
			if(dbg && (dbgLvl >= 3)) Print(K);

			current_rt = (current_rt+dir+4)%4;
			K = K * conj(rt[current_rt]).prime(4);
			if(dbg && (dbgLvl >= 3)) Print(K);


			// ***** SOLVE LINEAR SYSTEM M*rt = K ******************************
			if(dbg && (dbgLvl >= 3)) { 
				PrintData(M);
				PrintData(K);
				//PrintData(Kp);
			}

			// Check Hermicity of M
			if (dbg && (dbgLvl >= 1)) {
				std::cout <<"Check Hermicity of M"<< std::endl;
				std::cout <<"swapprime: "<< pl[2*ord[2]+(1+ORD_DIR[i_rt])/2] <<" <-> "
					<< 4+pl[2*ord[2]+(1+ORD_DIR[i_rt])/2] << std::endl;
				std::cout <<"swapprime: "<< IOFFSET+pl[2*ord[2]+(1+ORD_DIR[i_rt])/2] <<" <-> "
					<< 4+IOFFSET+ pl[2*ord[2]+(1+ORD_DIR[i_rt])/2] << std::endl;
				auto Mdag = conj(M);
				Mdag = swapPrime(swapPrime(Mdag, 
					    pl[2*ord[2]+(1+ORD_DIR[i_rt])/2],
					4 + pl[2*ord[2]+(1+ORD_DIR[i_rt])/2]),
					  IOFFSET + pl[2*ord[2]+(1+ORD_DIR[i_rt])/2],
					4+IOFFSET + pl[2*ord[2]+(1+ORD_DIR[i_rt])/2]);
				
				ITensor mherm = 0.5*(M + Mdag);
				ITensor mantiherm = 0.5*(M - Mdag);

				m = 0.;
		        M.visit(max_m);
		        std::cout<<"M max element: "<< m <<std::endl;
				m = 0.;
		        mantiherm.visit(max_m);
		        std::cout<<"M-Mdag max element: "<< m <<std::endl;
			}

		    auto cmbK  = combiner(K.inds()[0], K.inds()[1], K.inds()[2]);
		    auto cmbKp = combiner(prime(K.inds()[0],-4), prime(K.inds()[1],-4), prime(K.inds()[2],-4));
			// auto cmbKp = combiner(Kp.inds()[0], Kp.inds()[1], Kp.inds()[2]);

		    if(dbg && (dbgLvl >= 3)) {
		    	for (int i=0; i<3; i++) Print(K.inds()[i]);
		    	//for (int i=0; i<3; i++) Print(Kp.inds()[i]);	
		    	Print(cmbK);
		    	Print(cmbKp);
		    }

		    // Solve linear system M*rt[r] = K for rt[r] by constructing the pseudoinverse M^dag
		    // of M via SVD. Then the solution, which minimizes norm distance |M rt[r] - K|
		    // is given by M^dag K = rt[r]
		    M = (cmbK * M) * cmbKp;
		    K = cmbK * K;

		    // symmetrize
		    auto Msym = 0.5*(M + ( (conj(M) * delta(combinedIndex(cmbK),prime(combinedIndex(cmbKp),1)))
		    	*delta(prime(combinedIndex(cmbK),1),combinedIndex(cmbKp)) ).prime(-1));

		    // check small negative eigenvalues
		    Msym = Msym*delta(prime(combinedIndex(cmbK),1),combinedIndex(cmbKp));
		    ITensor uM, dM;
		    auto spec = diagHermitian(Msym, uM, dM);
		    dbg_D = dM;
		    
		    if(dbg && (dbgLvl >= 1)) {
				std::cout<<"ORIGINAL SPECTRUM ";
				Print(dbg_D);
				std::setprecision(std::numeric_limits<long double>::digits10 + 1);
				for(int idm=1; idm<=dbg_D.inds().front().m(); idm++) 
					std::cout << dbg_D.real(dbg_D.inds().front()(idm),dbg_D.inds().back()(idm)) 
					<< std::endl;
			}

			// Find largest (absolute) EV and in the case of msign * mval < 0, multiply
			// dM by -1 as to set the sign of largest (and consecutive) EV to +1
			// SYM SOLUTION
			double msign = 1.0;
			double mval = 0.;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=dM.inds().front().m(); idm++) {  
				dM_elems.push_back(dM.real(dM.inds().front()(idm),dM.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// TODO in the case of msign < 0.0, the loop order has to be reversed
			// Drop small (and negative) EV's
			int index_cutoff;
			std::vector<double> log_dM_e, log_diffs;
			for (int idm=0; idm<dM_elems.size(); idm++) {
				if ( dM_elems[idm] > mval*machine_eps ) {
					log_dM_e.push_back(std::log(dM_elems[idm])); 
					log_diffs.push_back(log_dM_e[std::max(idm-1,0)]-log_dM_e[idm]);
				
					// candidate for cutoff
					if ((dM_elems[idm]/mval < svd_cutoff) && 
						(std::fabs(log_diffs.back()) > svd_maxLogGap) ) {
						index_cutoff = idm;

						// log diagnostics
						if ( minGapDisc > std::fabs(log_diffs.back()) ) {
							minGapDisc = std::fabs(log_diffs.back());
							//min_indexCutoff = std::min(min_indexCutoff, index_cutoff);
							minEvKept = dM_elems[std::max(idm-1,0)];
							//maxEvDisc  = dM_elems[idm];
						}
						
						for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

						//Dynamic setting of iso_eps
						iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);

						break;
					}
				} else {
					index_cutoff = idm;
					for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

					// log diagnostics
					minEvKept  = dM_elems[std::max(idm-1,0)];

					//Dynamic setting of iso_eps
					iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);
					
					break;
				}
				if (idm == dM_elems.size()-1) {
					index_cutoff = -1;

					// log diagnostics
					minEvKept  = dM_elems[idm];

					//Dynamic setting of iso_eps
					iso_eps = std::min(iso_eps, dM_elems[idm]);
				}
			}

			dM = diagTensor(dM_elems,dM.inds().front(),dM.inds().back());
			
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM ";
				Print(dM);
				for (int idm=1; idm<=dM.inds().front().m(); idm++) 
					std::cout << dM.real(dM.inds().front()(idm),dM.inds().back()(idm)) 
						<< std::endl;
			}

			// Invert Hermitian matrix Msym
			std::vector<double> elems_regInvDM;
			for (int idm=1; idm<=dM.inds().front().m(); idm++) {
				if (dM.real(dM.inds().front()(idm),dM.inds().back()(idm))/
						dM.real(dM.inds().front()(1),dM.inds().back()(1))  > svd_cutoff) {  
					elems_regInvDM.push_back(msign*1.0/dM.real(dM.inds().front()(idm),
						dM.inds().back()(idm)) );
				} else
					elems_regInvDM.push_back(0.0);
			}
			auto regInvDM = diagTensor(elems_regInvDM, dM.inds().front(),dM.inds().back());
			
			if(dbg && (dbgLvl >= 1)) { std::cout<<"regInvDM.scale(): "<< regInvDM.scale() << std::endl; }
			// END SYM SOLUTION

			//Msym = (uM*dM)*prime(uM);	
			//Msym = Msym*delta(prime(combinedIndex(cmbK),1),combinedIndex(cmbKp));

			// if(dbg && (dbgLvl >= 1)) {
			// 	Print(svM);
			// 	std::setprecision(std::numeric_limits<long double>::digits10 + 1);
			// 	for(int isv=1; isv<=svM.inds().front().m(); isv++) 
			// 		std::cout << svM.real(svM.inds().front()(isv),svM.inds().back()(isv)) 
			// 		<< std::endl;
			// }
			// if(dbg && (dbgLvl >= 1)) {
			// 	Print(K);
			// 	std::setprecision(std::numeric_limits<long double>::digits10 + 1);
			// 	for(int isv=1; isv<=K.inds().front().m(); isv++) 
			// 		std::cout << K.real(K.inds().front()(isv)) << std::endl;
			// }
		    
			// if(dbg && (dbgLvl >= 1)) {
			// 	Print(regInvSvM);
			// 	std::setprecision(std::numeric_limits<long double>::digits10 + 1);
			// 	for(int isv=1; isv<=regInvSvM.inds().front().m(); isv++) 
			// 		std::cout << regInvSvM.real(regInvSvM.inds().front()(isv),regInvSvM.inds().back()(isv)) 
			// 		<< std::endl;
			// }

		    ITensor niso;

		    // Msym = U^dag D U ==> Msym^-1 = U^-1 D^-1 (U^dag)^-1 = U^dag D^-1 U
			// SYM SOLUTION
		 	Msym = (conj(uM)*regInvDM)*prime(uM);
			Msym = Msym*delta(prime(combinedIndex(cmbK),1),combinedIndex(cmbKp));
			niso = (Msym*K)*cmbKp;
			// END SYM SOLUTION

			if(dbg && (dbgLvl >= 2)) {
				Print(niso);
				niso.visit(print_elem);
			}

			M = (cmbK * M) * cmbKp;
			K = K*cmbK;
			
			if(dbg && (dbgLvl >= 1)) {
				ITensor optCond = M*niso - K;
				m = 0.;
			    optCond.visit(max_m);
			    std::cout<<"optCond(M*niso - K) max element: "<< m <<std::endl;
			}

			// Largest elements of isometries
			// for (int i=0; i<=3; i++) {
		 	// m = 0.;
			// rt[i].visit(max_m);
			// std::cout<<" iso["<< i <<"] max_elem: "<< m;
			// }
			// std::cout << std::endl;
			// for (int i=0; i<=3; i++) std::cout<<" iso["<< i <<"].scale(): "<< rt[i].scale();
			// std::cout << std::endl;

		    //rt_diffs.push_back(norm(rt[r]-niso));
			rt_diffs.push_back(norm(rt[r]));
 			
 			// scale logScale of isometry tensor to 1.0
 			niso.scaleTo(1.0);
 			rt[r] = niso;

			// Check overlap
			if (i_rt==3) {
				std::complex<double> ovrlp_val;
				ITensor tempOLP;
				tempOLP = (prime(conj(niso),4)*M)*niso;
				if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				ovrlp_val = sumelsC(tempOLP);
				if (isComplex(tempOLP)) {
					std::cout<<"NORM is Complex: imag(ovrlp_val)="<< ovrlp_val.imag() << std::endl;
				}
				overlaps.push_back(ovrlp_val.real());

				tempOLP = K*prime(conj(niso),4);
				if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				ovrlp_val = sumelsC(tempOLP);
				if (isComplex(tempOLP)) {
					std::cout<<"OVERLAP is Complex: imag(ovrlp_val)="<< ovrlp_val.imag() << std::endl;
				}
				overlaps.push_back(ovrlp_val.real());

				// tempOLP = niso*Kp;
				// if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				overlaps.push_back(ovrlp_val.real());
			}
		}

		altlstsquares_iter++;
		// check convergence
		if (altlstsquares_iter > 1) {
			auto dist_init = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
				- overlaps[overlaps.size()-4];
			auto dist_curr = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
				- overlaps[overlaps.size()-1];
			if (std::abs((dist_curr-dist_init)/overlaps[overlaps.size()-6]) < iso_eps)
				converged = true;
		}
		
		if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	}

	for(int i=0; i<(overlaps.size()/3); i++) {
		std::cout<<"M: "<< overlaps[3*i] <<" K: "<< overlaps[3*i+1]
			<<" Kp: "<< overlaps[3*i+2] <<std::endl;
	}
	//std::cout<<"rt_diffs.size() = "<< rt_diffs.size() << std::endl;
	for(int i=0; i<(rt_diffs.size()/4); i++) {
		std::cout<<"rt_diffs: "<<rt_diffs[4*i]<<" "<<rt_diffs[4*i+1]
			<<" "<<rt_diffs[4*i+2]<<" "<<rt_diffs[4*i+3]<<std::endl;
	}

	// BALANCE ISOMETRIES
	std::cout << "BALANCING ISOMETRIES" << std::endl;
	double iso_tot_mag = 1.0;
    for (int i=0; i<=3; i++) {
    	m = 0.;
		rt[i].visit(max_m);
    	rt[i] = rt[i] / m;
    	iso_tot_mag = iso_tot_mag * m;
    }
    rt[0] = rt[0] * std::pow(iso_tot_mag,(1.0/3.0));
    rt[1] = rt[1] * std::pow(iso_tot_mag,(1.0/6.0));
    rt[2] = rt[2] * std::pow(iso_tot_mag,(1.0/6.0));
    rt[3] = rt[3] * std::pow(iso_tot_mag,(1.0/3.0));

	// STORE ISOMETRIES
    if (rtInitType == "REUSE") {
        std::cout << "STORING ISOMETRIES" << std::endl;
        for (int i=0; i<=3; i++) iso_store[i] = rt[i];
    }

	// update on-site tensors of cluster
	auto newT = getketT(cls.sites.at(tn[0]), uJ1J2.H1, {&rt[0],NULL}, (dbg && (dbgLvl >=3)) );
	cls.sites.at(tn[0]) = newT;
	newT = getketT(cls.sites.at(tn[1]), uJ1J2.H2, {&rt[1],&rt[2]}, (dbg && (dbgLvl >=3)) );
	cls.sites.at(tn[1]) = newT;
	newT = getketT(cls.sites.at(tn[2]), uJ1J2.H3, {&rt[3],NULL}, (dbg && (dbgLvl >=3)) );
	cls.sites.at(tn[2]) = newT;

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
		diag_maxElem = diag_maxElem + tn[i] +" : "+ std::to_string(m) +" ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "PTN3") {
		double nn = std::pow(std::abs(overlaps[overlaps.size()-3]), (1.0/6.0));
		for (int i=0; i<3; i++) cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / nn;
	} else if (otNormType == "PTN4") {
		double nn = std::sqrt(std::abs(overlaps[overlaps.size()-3]));
		double ot_norms_tot = 0.0;
		std::vector<double> ot_norms;
		for (int i=0; i<4; i++) 
			{ ot_norms.push_back(norm(cls.sites.at(tn[i]))); ot_norms_tot += ot_norms.back(); } 
		for (int i=0; i<4; i++) cls.sites.at(tn[i]) = 
			cls.sites.at(tn[i]) / std::pow(nn, (ot_norms[i]/ot_norms_tot));
	} else if (otNormType == "BLE") {
		for (int i=0; i<3; i++) {
			m = 0.;
			cls.sites.at(tn[i]).visit(max_m);
			cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / sqrt(m);
		}
	} else if (otNormType == "NONE") {
	} else {
		std::cout<<"Unsupported on-site tensor normalisation after full update: "
			<< otNormType << std::endl;
		exit(EXIT_FAILURE);
	}

	// max element of on-site tensors after normalization
    for (int i=0; i<4; i++) {
            m = 0.;
            cls.sites.at(tn[i]).visit(max_m);
        std::cout << tn[i] <<" : "<< std::to_string(m) <<" ";
    }
    std::cout << std::endl;

	for (int r=0; r<4; r++) { PrintData(rt[r]); }

	// prepare and return diagnostic data
	diag_data.add("alsSweep",altlstsquares_iter);
	diag_data.add("siteMaxElem",diag_maxElem);
	if(dbg && (dbgLvl >= 1)) {
		diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
		diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	}
	auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
		- overlaps[overlaps.size()-4];
	auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
		- overlaps[overlaps.size()-1];
	diag_data.add("finalDist0",dist0);
	diag_data.add("finalDist1",dist1);

	minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	diag_data.add("minGapDisc",minGapDisc);
	diag_data.add("minEvKept",minEvKept);
	//diag_data.add("maxEvDisc",maxEvDisc);

	return diag_data;
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