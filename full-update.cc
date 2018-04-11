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

ITensor getT(ITensor const& s, std::array<Index, 4> const& plToEnv, 
	ITensor const& op, std::array<const ITensor *, 4> rt, bool dbg) {  

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
	if (rt[0]) res = res * (*rt[0]);
	if (rt[1]) res = res * (*rt[1]);
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
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto rtInitType = args.getString("fuIsoInit","DELTA");
    auto rtInitParam = args.getReal("fuIsoInitNoiseLevel",1.0e-3);
    auto otNormType = args.getString("otNormType");

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

	// read off auxiliary indices of the cluster sites
	std::array<Index, 4> aux({
		noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

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
	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

	// Create and initialize reduction tensors RT
	std::vector<ITensor> rt(4);
	rt[0] = ITensor(prime(aux[0],pl[1]), uJ1J2.a12, prime(aux[0],pl[1]+IOFFSET));
	rt[1] = ITensor(prime(aux[1],pl[2]), uJ1J2.a12, prime(aux[1],pl[2]+IOFFSET));
	rt[2] = ITensor(prime(aux[1],pl[3]), uJ1J2.a23, prime(aux[1],pl[3]+IOFFSET));
	rt[3] = ITensor(prime(aux[2],pl[4]), uJ1J2.a23, prime(aux[2],pl[4]+IOFFSET));
	// simple initialization routines
	if (rtInitType == "RANDOM" || rtInitType == "DELTA" || rtInitType == "NOISE") { 
		for (int i=0; i<=3; i++) {
			initRT_basic(rt[i],rtInitType,{"fuIsoInitNoiseLevel",rtInitParam});
			if(dbg && (dbgLvl >= 3)) PrintData(rt[i]);
		}
	}
	// self-consistent initialization - guess isometries between tn[0] and tn[1]
	else if (rtInitType == "LINKSVD") {
		ITensor ttemp, tdelKet, tdelBra, tRT(1.0);
		std::array<const itensor::ITensor *, 4> trtp = {{NULL,NULL,NULL,NULL}};
		for(int r=0; r<=3; r+=3){
			std::array<int,4> tord = ORD[r];
			if(dbg) std::cout <<"GUESS RT "<< tn[tord[0]] <<"-"<< tn[tord[3]] << std::endl;
			for(int o=0; o<=3; o++) {
				// construct o'th corner of tmpRT
				ttemp = pc[tord[o]] * getT(cls.sites.at(tn[tord[o]]), iToE[tord[o]], 
			 			*mpo[tord[o]], trtp, (dbg && (dbgLvl >= 3)) );
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
			
			auto cmb0 = combiner(prime(aux[tord[0]],pl[2*tord[0]+(1-ORD_DIR[r])/2]),
				noprime(findtype(tRT, MPOLINK)));
			auto cmb1 = combiner(prime(aux[tord[3]],pl[2*tord[3]+(1+ORD_DIR[r])/2]),
				prime(noprime(findtype(tRT, MPOLINK)),IOFFSET));
			if(dbg && (dbgLvl >= 3)) { Print(cmb0); Print(cmb1); } 

			tRT = (tRT * cmb0) * cmb1;

			ITensor tU(combinedIndex(cmb0)),tS,tV;
			svd(tRT,tU,tS,tV,{"Maxm",aux[0].m()});

			if(dbg && (dbgLvl >= 2)) PrintData(tS);
			
			// rt[r+(1+ORD_DIR[r])/2-r/3] = ((tU*tS)*delta(commonIndex(tS,tV), 
			// 	prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
			// rt[r+(1-ORD_DIR[r])/2-r/3] = (((tV*tS)*delta(commonIndex(tS,tU), 
			// 	prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1).prime(MPOLINK,-IOFFSET);
			
			rt[r+(1+ORD_DIR[r])/2-r/3] = (tU*delta(commonIndex(tS,tU), 
				prime(findtype(cmb0, AUXLINK),IOFFSET)))*cmb0;
			rt[r+(1-ORD_DIR[r])/2-r/3] = ((tV*delta(commonIndex(tS,tV), 
				prime(findtype(cmb1, AUXLINK),IOFFSET)))*cmb1).prime(MPOLINK,-IOFFSET);
			
			//if(dbg && (dbgLvl >= 3)) {
				// std::cout<< r+(1-ORD_DIR[r])/2-r/3 <<" ";
				// Print(rt[r+(1-ORD_DIR[r])/2-r/3]);
				// std::cout<< r+(1+ORD_DIR[r])/2-r/3 <<" ";
				// Print(rt[r+(1+ORD_DIR[r])/2-r/3]);
			//}

			tRT = ITensor(1.0);
		}
	} 
	else if (rtInitType == "REUSE") {
            std::cout << "REUSING ISO: ";
            if (iso_store[0]) {
                    rt[0] = iso_store[0];
                    std::cout << " 0 ";
            } else {
                    rt[0] = ITensor(prime(aux[0],pl[1]), uJ1J2.a12, prime(aux[0],pl[1]+IOFFSET));
                    initRT_basic(rt[0],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[1]) {
                    rt[1] = iso_store[1];
                    std::cout << " 1 ";
            } else {
                    rt[1] = ITensor(prime(aux[1],pl[2]), uJ1J2.a12, prime(aux[1],pl[2]+IOFFSET));
                    initRT_basic(rt[1],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[2]) {
                    rt[2] = iso_store[2];
                    std::cout << " 2 ";
            } else {
                    rt[2] = ITensor(prime(aux[1],pl[3]), uJ1J2.a23, prime(aux[1],pl[3]+IOFFSET));
                    initRT_basic(rt[2],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
            }

            if (iso_store[3]) {
                    rt[3] = iso_store[3];
                    std::cout << " 3" << std::endl;
            } else {
                    rt[3] = ITensor(prime(aux[2],pl[4]), uJ1J2.a23, prime(aux[2],pl[4]+IOFFSET));
                    initRT_basic(rt[3],"DELTA",{"fuIsoInitNoiseLevel",rtInitParam});
                    std::cout << std::endl;
            }
    }
	else {
		std::cout<<"Unsupported fu-isometry initialization: "<< rtInitType << std::endl;
		exit(EXIT_FAILURE);
	}

	// Prepare Alternating Least Squares to maximize the overlap
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	auto print_elem = [](double d) {
		std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		std::cout<< d << std::endl;
	};
	
	int r, altlstsquares_iter = 0;
	bool converged = false;
	std::vector<double> overlaps;
	std::vector<double> rt_diffs; 
	
	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	ITensor dbg_D, dbg_svM;
	while (not converged) {
		
		for (int i_rt=0; i_rt<=3; i_rt++) {
			//r = ((altlstsquares_iter % 2) == 0) ? i_rt : 3-i_rt;
			r = i_rt;

			// define M,K construction order (defined 
			// in *.h file in terms of indices of tn)
			std::array<int,4> ord = ORD[r];
			if(dbg && (dbgLvl >= 3)) {
				std::cout <<"Order for rt["<< r <<"]: ("<< ORD_DIR[r] <<") ";
				for(int o=0; o<=3; o++) {std::cout << tn[ord[o]] <<" ";}
				std::cout << std::endl;
			}

			// construct matrix M, which is defined as <psi~|psi~> = rt[r]^dag * M * rt[r]	
			if(dbg && (dbgLvl >= 3)) std::cout <<"COMPUTING Matrix M"<< std::endl;
			ITensor temp, deltaKet, deltaBra, M(1.0);
			std::array<const itensor::ITensor *, 4> rtp; // holds reduction tensor pointers => rtp
			for(int o=0; o<=3; o++) {

				// depending on choosen order (r) set up rtp for current site o
				for (int i=0; i<=3; i++) 
					rtp[i] = (RTPM[r][o][i] >= 0) ? &rt[RTPM[r][o][i]] : NULL;
			
				if(dbg && (dbgLvl >= 3)) {
					std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
						<< tn[ord[o]] << std::endl;
					for(int i=0; i<=3; i++) {std::cout << RTPM[r][o][i] <<" ";}
					std::cout << std::endl;
				}

				// construct o'th corner of M
				temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		 			*mpo[ord[o]], rtp, (dbg && (dbgLvl >= 3)) );
				if(dbg && (dbgLvl >=3)) Print(temp);
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
				if(dbg && (dbgLvl >= 3)) {
					Print(deltaKet);
					Print(deltaBra);
				}

				M *= temp;
				if (o==3) M = (M * deltaBra) * deltaKet;
			
				if(dbg && (dbgLvl >= 3)) Print(M);
			}

			// construct vector K, which is defined as <psi~|psi'> = rt[r]^dag * K
			if(dbg && (dbgLvl >= 3)) std::cout <<"COMPUTING Vector K"<< std::endl;
			ITensor K(1.0);
			for(int o=0; o<=3; o++) {

				// depending on choosen order (r) set up rtp for current site o
				for (int i=0; i<=3; i++) 
					rtp[i] = (RTPK[r][o][i] >= 0) ? &rt[RTPK[r][o][i]] : NULL;
			
				if(dbg && (dbgLvl >= 3)) {
					std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
						<< tn[ord[o]] << std::endl;
					for(int i=0; i<=3; i++) {std::cout << RTPK[r][o][i] <<" ";}
					std::cout << std::endl;
				}

				// construct o'th corner of K
				temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		 			*mpo[ord[o]], rtp, (dbg && (dbgLvl >= 3)) );
				if(dbg && (dbgLvl >= 3)) Print(temp);
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
				if(dbg && (dbgLvl >= 3)) {
					Print(deltaKet);
					Print(deltaBra);
				}

				K *= temp;
				if (o==3) K = (K * deltaBra) * deltaKet;
			
				if(dbg && (dbgLvl >= 3)) Print(K);
			}

			// construct vector Kp, which is defined as <psi'|~psi> = Kp * rt[r]
			// if(dbg && (dbgLvl >= 3)) std::cout <<"COMPUTING Vector Kp"<< std::endl;
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

			// 	if(dbg && (dbgLvl >= 3)) {
			// 		std::cout <<"Optimizing rt["<< r <<"] - RedTens site "
			// 			<< tn[ord[o]] << std::endl;
			// 		for(int i=0; i<=3; i++) {std::cout << RTPK[r][o][i] <<" ";}
			// 		std::cout << std::endl;
			// 	}

			// 	// construct o'th corner of K
			// 	temp = pc[ord[o]] * getT(cls.sites.at(tn[ord[o]]), iToE[ord[o]], 
 		//  			*mpo[ord[o]], rtp, (dbg && (dbgLvl >= 3)) );
			// 	if(dbg && (dbgLvl >= 3)) Print(temp);
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
			// 	if(dbg && (dbgLvl >= 3)) {
			// 		Print(deltaKet);
			// 		Print(deltaBra);
			// 	}

			// 	Kp *= temp;
			// 	if (o==3) Kp = (Kp * deltaBra) * deltaKet;
			
			// 	if(dbg && (dbgLvl >= 2)) Print(Kp);
			// }

			if(dbg && (dbgLvl >= 3)) { 
				PrintData(M);
				PrintData(K);
				//PrintData(Kp);
			}

			// Check Hermicity of M
			if (dbg && (dbgLvl >= 1)) {
				std::cout <<"Check Hermicity of M"<< std::endl;
				std::cout <<"swapprime: "<< pl[2*ord[3]+(1+ORD_DIR[r])/2] <<" <-> "
					<< 4+pl[2*ord[3]+(1+ORD_DIR[r])/2] << std::endl;
				std::cout <<"swapprime: "<< IOFFSET+pl[2*ord[3]+(1+ORD_DIR[r])/2] <<" <-> "
					<< 4+IOFFSET+ pl[2*ord[3]+(1+ORD_DIR[r])/2] << std::endl;
				auto Mdag = mapprime(conj(M), MPOLINK,0,12, MPOLINK,4,8);
				Mdag = prime(swapPrime(swapPrime(Mdag, 
					    pl[2*ord[3]+(1+ORD_DIR[r])/2],
					4 + pl[2*ord[3]+(1+ORD_DIR[r])/2]),
					  IOFFSET + pl[2*ord[3]+(1+ORD_DIR[r])/2],
					4+IOFFSET + pl[2*ord[3]+(1+ORD_DIR[r])/2]),MPOLINK,-8);
				
				ITensor mherm = 0.5*(M + Mdag);
				ITensor mantiherm = 0.5*(M - Mdag);

				m = 0.;
		        M.visit(max_m);
		        diag_maxMsymLE = m;
		        std::cout<<"M max element: "<< m <<std::endl;
				m = 0.;
		        mantiherm.visit(max_m);
		        diag_maxMasymLE = m;
		        std::cout<<"M-Mdag max element: "<< m <<std::endl;
			
				diag_maxMsymFN  = norm(mherm);
				diag_maxMasymFN = norm(mantiherm);
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

			// Drop small (and negative) EV's
			for (auto & elem : dM_elems) elem = (elem > svd_cutoff) ? elem : 0.0; 
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
// END SYM SOLUTION

			//Msym = (uM*dM)*prime(uM);	
			//Msym = Msym*delta(prime(combinedIndex(cmbK),1),combinedIndex(cmbKp));

// ASYM SOLUTION
		 // 	ITensor mU(combinedIndex(cmbK)),svM,mV;
			// svd(M,mU,svM,mV);
			// dbg_svM = svM;
// END ASYM SOLUTION

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
		    
// ASYM SOLUTION
			// // Regularize and invert singular value matrix svM
			// std::vector<double> elems_regInvSvM;
			// for (int isv=1; isv<=svM.inds().front().m(); isv++) {
			// 	if ( svM.real(svM.inds().front()(isv),svM.inds().back()(isv))/
			// 		 svM.real(svM.inds().front()(1),svM.inds().back()(1))  > svd_cutoff) {  
			// 		elems_regInvSvM.push_back(1.0/svM.real(svM.inds().front()(isv),
			// 			svM.inds().back()(isv)) );
			// 	} else
			// 		elems_regInvSvM.push_back(0.0);
			// }

			// auto regInvSvM = diagTensor(elems_regInvSvM, svM.inds().front(),svM.inds().back());
// END ASYM SOLUTION
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

// ASYM SOLUTION
			// niso = (conj(mV)*(regInvSvM*(conj(mU)*K))) * cmbKp; 
// END ASYM SOLUTION
			// linsystem(M,K,niso,{"plDiff",4,"dbg",true});

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
			    
				// optCond = prime(conj(niso),4)*M - Kp;
				// m = 0.;
			 //    optCond.visit(max_m);
			 //    std::cout<<"optCond(niso^dag*M - Kp) max element: "<< m <<std::endl;
			}

			//rt_diffs.push_back(norm(rt[r]-niso));
			rt_diffs.push_back(norm(rt[r]));
 			
 			rt[r] = niso;

			// Check overlap
			if (i_rt==3) {
				ITensor tempOLP;
				tempOLP = (prime(conj(niso),4)*M)*niso;
				if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				overlaps.push_back(sumels(tempOLP));
				tempOLP = K*prime(conj(niso),4);
				if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				overlaps.push_back(sumels(tempOLP));
				// tempOLP = niso*Kp;
				// if (rank(tempOLP) > 0) std::cout<<"ERROR - tempOLP not a scalar"<<std::endl;
				overlaps.push_back(sumels(tempOLP));
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