#include "models.h"

using namespace itensor;

// ----- Trotter gates (2Site, 3site, ...) MPOs -----------------------
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

MPO_3site getMPO3s_NNHLadder_2site(double tau, double J, double alpha) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP1 define exact U_123 = exp(-J(Sz_1.Sz_2 + Sz_2.Sz_3) - h(Sx_1+Sx_2+Sx_3))
    ITensor h123 = ITensor(s1,s2,s1p,s2p);
    h123 += alpha*J*( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) );

    auto cmbI = combiner(s1,s2);
    h123 = (cmbI * h123 ) * prime(cmbI);
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return ltorMPO2StoMPO3Sdecomp(u123, s1, s2);
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

MPO_3site getMPO3s_Ising3Body(double tau, double J1, double J2, double h) {
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
    h123 += -J1*( 2.0*SU2_getSpinOp(SU2_S_Z, s1) * 2.0*SU2_getSpinOp(SU2_S_Z, s2))* delta(s3,s3p);
    h123 += -J1*delta(s1,s1p)*( 2.0*SU2_getSpinOp(SU2_S_Z, s2) * 2.0*SU2_getSpinOp(SU2_S_Z, s3) );
    h123 += -h*( ((SU2_getSpinOp(SU2_S_P, s1) + SU2_getSpinOp(SU2_S_M, s1))*delta(s2,s2p))*delta(s3,s3p)
        + (delta(s1,s1p)*(SU2_getSpinOp(SU2_S_P, s2)+SU2_getSpinOp(SU2_S_M, s2)))*delta(s3,s3p)
        + delta(s1,s1p)*(delta(s2,s2p)*(SU2_getSpinOp(SU2_S_P, s3)+SU2_getSpinOp(SU2_S_M, s3))) );
    h123 += -J2*( 2.0*SU2_getSpinOp(SU2_S_Z, s1) * 2.0*SU2_getSpinOp(SU2_S_Z, s2) 
        * 2.0*SU2_getSpinOp(SU2_S_Z, s3) );

    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI); 
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

MPO_2site getMPO2s_NNH_2site(double tau, double J, double h) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP1 define exact U_12 = exp(-J(S_1.S_2) - h(Sz_1+Sz_2))
    ITensor h12 = ITensor(s1,s2,s1p,s2p);
    h12 += J*( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) );
    h12 += -h*(SU2_getSpinOp(SU2_S_Z, s1)*delta(s2,s2p) + delta(s1,s1p)*SU2_getSpinOp(SU2_S_Z, s2));

    auto cmbI = combiner(s1,s2);
    h12 = (cmbI * h12 ) * prime(cmbI);
    ITensor u12 = expHermitian(h12, {-tau, 0.0});
    u12 = (cmbI * u12 ) * prime(cmbI);
    // definition of U_12 done

    return symmMPO2Sdecomp(u12, s1, s2);
}
// ----- END Trotter gates (2Site, 3site, ...) MPOs -------------------

// ----- Definition of model base class and its particular instances --
J1J2Model::J1J2Model(double arg_J1, double arg_J2)
    : J1(arg_J1), J2(arg_J2) {}

void J1J2Model::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg SS_NN, "<<"Avg SS_NNN, "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void J1J2Model::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> evNNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    bool compute_SS_NNN = ( std::abs(J2) > 1.0e-15 );

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    // compute energies NNN links
    if ( compute_SS_NNN ) {
        evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
        evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
        evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
        evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(2,1)) );

        evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
        evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
        evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
        evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,1)) );
    }

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

    
    output << lineNo <<" "; 
    // write individual NN SS terms and average over all non-eq links
    double avgSS_NN = 0.;
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
        avgSS_NN += evNN[j];
    }
    avgSS_NN = avgSS_NN / 8.0; 
    output <<" "<< avgSS_NN;
    
    // write average NNN SS term over all non-eq NNN
    double avgSS_NNN = 0.;
    if (compute_SS_NNN) {
        for ( unsigned int j=0; j<evNNN.size(); j++ ) avgSS_NNN += evNNN[j];
        avgSS_NNN = avgSS_NNN / 8.0;
    }    
    output <<" "<< avgSS_NNN;

    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
        + sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )
        + sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = 2.0 * avgSS_NN * J1 + 2.0 * avgSS_NNN * J2; 
    output <<" "<< energy;

    output << std::endl;
}

NNHLadderModel::NNHLadderModel(double arg_J1, double arg_alpha)
    : J1(arg_J1), alpha(arg_alpha) {}

void NNHLadderModel::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg SS CA+DB, "<<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void NNHLadderModel::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

    // write energy
    double avgE_CAplusDB = 0.;
    output << lineNo <<" "; 
    for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    avgE_CAplusDB = (evNN[5] + evNN[6])/2.0;
    output <<" "<< avgE_CAplusDB;
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
        + sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )
        + sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = (evNN[0]+evNN[1]+evNN[2]+evNN[3]+evNN[4]+evNN[7]) * J1
         + (evNN[5]+evNN[6]) * (alpha*J1); 
    output <<" "<< energy; 

    output << std::endl;
}

IsingModel::IsingModel(double arg_J1, double arg_h)
    : J1(arg_J1), h(arg_h) {}

void IsingModel::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SzSz AB (0,0)(1,0), "<<"SzSz AC (0,0)(0,1), "
        <<"SzSz BD (1,0)(1,1), "<<"SzSz CD (0,1)(1,1), "
        <<"SzSz BA (1,0)(2,0), "<<"SzSz CA (0,1)(0,2), "
        <<"SzSz DB (1,1)(1,2), "<<"SzSz DC (1,1)(2,1), "
        <<"Avg SzSz, "<<"Avg Sz, "<<"Avg Sx, "<<"Energy"
        <<std::endl;
}

void IsingModel::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,0), std::make_pair(1,0)) );    //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,0), std::make_pair(0,1)) );    //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,0), std::make_pair(1,1)) );    //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,1), std::make_pair(1,1)) );    //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

    // write energy
    double avgE_8links = 0.;
    output << lineNo <<" "; 
    for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
        avgE_8links += evNN[j];
        output<<" "<< evNN[j];
    }
    avgE_8links = avgE_8links/8.0;
    output <<" "<< avgE_8links;
    
    // write Z magnetization
    double evMagZ_avg = 0.;
    double evMagX_avg = 0.;
    evMagZ_avg = 0.25*(ev_sA[0] + ev_sB[0] + ev_sC[0] + ev_sD[0]);
    output <<" "<< evMagZ_avg;
    evMagX_avg = 0.25*(ev_sA[1] + ev_sB[1] + ev_sC[1] + ev_sD[1]);
    output <<" "<< evMagX_avg;

    // write Energy 
    // * working with spin DoFs instead of Ising DoFs hence factor of 2
    double energy = -4.0*(8.0*avgE_8links) * J1 - 4.0 * 2.0 * h * evMagX_avg;
    output <<" "<< energy;

    output << std::endl;
}

Ising3BodyModel::Ising3BodyModel(double arg_J1, double arg_J2, double arg_h)
    : J1(arg_J1), J2(arg_J2), h(arg_h) {}

void Ising3BodyModel::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SzSz AB (0,0)(1,0), "<<"SzSz AC (0,0)(0,1), "
        <<"SzSz BD (1,0)(1,1), "<<"SzSz CD (0,1)(1,1), "
        <<"SzSz BA (1,0)(2,0), "<<"SzSz CA (0,1)(0,2), "
        <<"SzSz DB (1,1)(1,2), "<<"SzSz DC (1,1)(2,1), "
        <<"Avg SzSz, "<<"Avg SzSzSz, "<<"Avg Sz, "<<"Avg Sx, "<<"Energy"
        <<std::endl;
}

void Ising3BodyModel::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev3SZ;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,0), std::make_pair(1,0)) );    //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,0), std::make_pair(0,1)) );    //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,0), std::make_pair(1,1)) );    //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,1), std::make_pair(1,1)) );    //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    // compute "3-site" terms Sz_i Sz_j Sz_k
    // 4 triangles centered on site [0,0]
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,-1), std::make_pair(0,0), std::make_pair(1,0), std::make_pair(1,-1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,-1), std::make_pair(0,0), std::make_pair(-1,0), std::make_pair(-1,-1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,1), std::make_pair(0,0), std::make_pair(1,0), std::make_pair(1,1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,1), std::make_pair(0,0), std::make_pair(-1,0), std::make_pair(-1,1)}) );
    // 4 triangles centered on site [1,0]
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,-1), std::make_pair(1,0), std::make_pair(2,0), std::make_pair(2,-1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,-1), std::make_pair(1,0), std::make_pair(0,0), std::make_pair(0,-1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,1), std::make_pair(1,0), std::make_pair(2,0), std::make_pair(2,1)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,1), std::make_pair(1,0), std::make_pair(0,0), std::make_pair(0,1)}) );
    // 4 triangles centered on site [0,1]
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,0), std::make_pair(0,1), std::make_pair(1,1), std::make_pair(1,0)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,0), std::make_pair(0,1), std::make_pair(-1,1), std::make_pair(-1,0)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,2), std::make_pair(0,1), std::make_pair(1,1), std::make_pair(1,2)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(0,2), std::make_pair(0,1), std::make_pair(-1,1), std::make_pair(-1,2)}) );
    // 4 triangles centered on site [1,1]
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,0), std::make_pair(1,1), std::make_pair(2,1), std::make_pair(2,0)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,0), std::make_pair(1,1), std::make_pair(0,1), std::make_pair(0,0)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,2), std::make_pair(1,1), std::make_pair(2,1), std::make_pair(2,2)}) );
    ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
        {std::make_pair(1,2), std::make_pair(1,1), std::make_pair(0,1), std::make_pair(0,2)}) );
    // end computing "3-site" terms Sz_i Sz_j Sz_k

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

    // write energy
    double avgE_8links = 0.;
    output << lineNo <<" "; 
    for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
        avgE_8links += evNN[j];
        output<<" "<< evNN[j];
    }
    avgE_8links = avgE_8links/8.0;
    output <<" "<< avgE_8links;
    
    // write energy
    double avgE_3sz = 0.;
    for ( unsigned int j=0; j<ev3SZ.size(); j++ ) {
        avgE_3sz += ev3SZ[j];
    }
    avgE_3sz = avgE_3sz/16.0;
    output <<" "<< avgE_3sz;

    // write Z magnetization
    double evMagZ_avg = 0.;
    double evMagX_avg = 0.;
    evMagZ_avg = 0.25*(
        sqrt(ev_sA[0]*ev_sA[0])
        + sqrt(ev_sB[0]*ev_sB[0])
        + sqrt(ev_sC[0]*ev_sC[0])
        + sqrt(ev_sD[0]*ev_sD[0])
        );
    output <<" "<< evMagZ_avg;
    evMagX_avg = 0.25*(
        sqrt(ev_sA[1]*ev_sA[1])
        + sqrt(ev_sB[1]*ev_sB[1])
        + sqrt(ev_sC[1]*ev_sC[1])
        + sqrt(ev_sD[1]*ev_sD[1])
        );
    output <<" "<< evMagX_avg;

    // write Energy 
    // * working with spin DoFs instead of Ising DoFs hence factor of 2
    double energy = -4.0*(8.0*avgE_8links) * J1 - 4.0 * 2.0 * h * evMagX_avg
        -8.0*(16.0*avgE_3sz) * J2;
    output <<" "<< energy; 

    output << std::endl;
}

NNHModel_2x2Cell_AB::NNHModel_2x2Cell_AB(double arg_J, double arg_h)
    : J1(arg_J), h(arg_h) {}

void NNHModel_2x2Cell_AB::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AB (0,0)(0,1), "
        <<"SS BA (1,0)(2,0), "<<"SS BA (0,1)(0,2), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void NNHModel_2x2Cell_AB::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) ); //AB horizontal
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) ); //AB vertical

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA horizontal
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //BA vertical

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    // write energy
    output << lineNo <<" "; 
    for ( unsigned int j=evNN.size()-4; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.5*(
        sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = ( (evNN[0]+evNN[1]+evNN[2]+evNN[3]) * J1
         + (ev_sA[0]+ev_sB[0]) * h)/2 ; 
    output <<" "<< energy; 

    output << std::endl;
}

NNHModel_2x2Cell_ABCD::NNHModel_2x2Cell_ABCD(double arg_J, double arg_h)
    : J1(arg_J), h(arg_h) {}

void NNHModel_2x2Cell_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void NNHModel_2x2Cell_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args const& metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

    // write energy
    output << lineNo <<" "; 
    for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
        + sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )
        + sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = ( (evNN[0]+evNN[1]+evNN[2]+evNN[3]+evNN[4]+evNN[5]+evNN[6]+evNN[7]) * J1
         + (ev_sA[0] + ev_sB[0] + ev_sC[0] + ev_sD[0]) * h)/4.0; 
    output <<" "<< energy; 

    output << std::endl;
}

// ----- END Definition of model class --------------------------------

// ----- Model Definitions --------------------------------------------
void getModel_J1J2(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_J2 = json_model["J2"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new J1J2Model(arg_J1, arg_J2));

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
    std::unique_ptr<Model> & ptr_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_alpha = json_model["alpha"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new NNHLadderModel(arg_J1, arg_alpha));

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

        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_J1) );
        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_alpha*arg_J1) );

        ptr_gateMPO = {
            &(gateMPO[0]), &(gateMPO[1]), &(gateMPO[1]), &(gateMPO[0]),
            &(gateMPO[0]), &(gateMPO[1]), &(gateMPO[1]), &(gateMPO[0]),
            &(gateMPO[0]), &(gateMPO[1]), &(gateMPO[1]), &(gateMPO[0]),
            &(gateMPO[0]), &(gateMPO[1]), &(gateMPO[1]), &(gateMPO[0])
        };
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

        gateMPO.push_back( getMPO3s_NNHLadder_2site(arg_tau, arg_J1, 1.0) );
        gateMPO.push_back( getMPO3s_NNHLadder_2site(arg_tau, arg_J1, arg_alpha) );

        for (int i=0; i<6; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
        for (int i=0; i<2; i++) ptr_gateMPO.push_back( &(gateMPO[1]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_NNH_2x2Cell_AB(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_2site > & gateMPO,
    std::vector< MPO_2site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1 = json_model["J1"].get<double>();
    double arg_h = json_model["h"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new NNHModel_2x2Cell_AB(arg_J1, arg_h));

    // time step
    double arg_tau = json_model["tau"].get<double>();
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    if (arg_fuGateSeq == "2SITE") {
        gates = {
            {"A", "B"}, {"B", "A"},
            {"A", "B"}, {"B", "A"}
        };

        gate_auxInds = {
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
        };

        gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, arg_h) );

        for (int i=0; i<4; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 2-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_NNH_2x2Cell_ABCD(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_2site > & gateMPO,
    std::vector< MPO_2site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1 = json_model["J1"].get<double>();
    double arg_h = json_model["h"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new NNHModel_2x2Cell_ABCD(arg_J1, arg_h));

    // time step
    double arg_tau = json_model["tau"].get<double>();
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    if (arg_fuGateSeq == "2SITE") {
        gates = {
            {"A", "B"}, {"B", "A"}, 
            {"C", "D"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };

        gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, arg_h) );

        for (int i=0; i<8; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_Ising(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new IsingModel(arg_J1, arg_h));

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

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
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

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
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
        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
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

        gateMPO.push_back( getMPO3s_Ising_2site(arg_tau, arg_J1, arg_h/4.0) );
        ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_Ising3Body(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_J2     = json_model["J2"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    
    ptr_model = std::unique_ptr<Model>(new Ising3BodyModel(arg_J1, arg_J2, arg_h));

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

        gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1/4.0, arg_J2, arg_h/12.0) );
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

        gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1/4.0, arg_J2, arg_h/12.0) );
        ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_3site(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    std::string arg_modelType = json_model["type"].get<std::string>(); 

    if(arg_modelType == "J1J2") {
        getModel_J1J2(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "NNHLadder") {
        getModel_NNHLadder(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "Ising") {
        getModel_Ising(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "Ising3Body") {
        getModel_Ising3Body(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_2site(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_2site > & gateMPO,
    std::vector< MPO_2site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    std::string arg_modelType = json_model["type"].get<std::string>(); 

    if (arg_modelType == "NNH_2x2Cell_AB") {
        getModel_NNH_2x2Cell_AB(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "NNH_2x2Cell_ABCD") {
        getModel_NNH_2x2Cell_ABCD(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }
}
// ----- END Model Definitions ----------------------------------------