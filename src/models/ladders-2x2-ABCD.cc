#include "p-ipeps/config.h"
#include "p-ipeps/models/ladders-2x2-ABCD.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_Ladders(double tau, double J) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // del:  XY limit -1 <-- 0 (HB)-->--> >>1 Ising limit
    // STEP 1 define exact h_12 = J*(S_1.S_2) + del*(Sz_1Sz_2) - h*(Sz_1+Sz_2)
    ITensor h12 = ITensor(s1,s2,s1p,s2p);
    h12 += J*( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) );

    auto cmbI = combiner(s1,s2);
    h12 = (cmbI * h12 ) * prime(cmbI);
    ITensor u12 = expHermitian(h12, {-tau, 0.0});
    u12 = (cmbI * u12 ) * prime(cmbI);
    // definition of U_12 done

    return symmMPO2Sdecomp(u12, s1, s2);
}

MPO_3site getMPO3s_Ladders(double tau, double J, double alpha) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

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
            + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) )*delta(s3,s3p);
    h123 += alpha*J*( SU2_getSpinOp(SU2_S_Z, s2) * SU2_getSpinOp(SU2_S_Z, s3)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s2) * SU2_getSpinOp(SU2_S_M, s3)
            + SU2_getSpinOp(SU2_S_M, s2) * SU2_getSpinOp(SU2_S_P, s3) ) )*delta(s1,s1p);
    
    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI); 
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    // definition of U_123 done

    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

OpNS getOP4s_Ladders(double tau, double J, double alpha) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s4 = Index("S4", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);
    Index s4p = prime(s4);

    ITensor h4 = ITensor(s1,s2,s3,s4,s1p,s2p,s3p,s4p);
   
    auto nnS1S2 = [](Index s1, Index s2) {
        return ( SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
            + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
            + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) );
    };

    // s1--J--s2
    //  |     |<-- alpha*J
    // s4--J--s3
    h4 += J * nnS1S2(s1,s2) * delta(s3,s3p) * delta(s4,s4p);        // S1S2id3id4
    h4 += alpha*J * nnS1S2(s2,s3) * delta(s1,s1p) * delta(s4,s4p);  // id1S2S3id4
    h4 += alpha*J * nnS1S2(s1,s4) * delta(s2,s2p) * delta(s3,s3p);  // S1id2id3S4
    h4 += J * nnS1S2(s3,s4) * delta(s1,s1p) * delta(s2,s2p);        // id1id2S3S4

    auto cmbI = combiner(s1,s2,s3,s4);
    h4 = (cmbI * h4 ) * prime(cmbI);
    ITensor u4 = expHermitian(h4, {-tau, 0.0});
    u4 = (cmbI * u4 ) * prime(cmbI);

    auto op4s = OpNS(4);

    op4s.op = u4;
    op4s.pi = {s1,s2,s3,s4};

    return op4s;
}
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
LaddersModel_2x2_ABCD::LaddersModel_2x2_ABCD(double arg_J, double arg_alpha)
    : J(arg_J), alpha(arg_alpha) {}

void LaddersModel_2x2_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg SS CA+DB, "<<"Avg mag=|S|, "<<"Energy, "
        <<"m_z(0,0), "<<"m_x(0,0), "<<"Re(m_y(0,0)), "
        <<"m_z(1,0), "<<"m_x(1,0), "<<"Re(m_y(1,0)), "
        <<"m_z(0,1), "<<"m_x(0,1), "<<"Re(m_y(0,1)), "
        <<"m_z(1,1), "<<"m_x(1,1), "<<"Re(m_y(1,1))"
        <<std::endl;
}

void LaddersModel_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,0), Vertex(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,0), Vertex(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,0), Vertex(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,1), Vertex(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,0), Vertex(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,1), Vertex(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,1), Vertex(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,1), Vertex(2,1))); //DC

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,1));

    // write energy
    double avgE_CAplusDB = 0.;
    output << lineNo; 
    for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    avgE_CAplusDB = (evNN[5] + evNN[6])/2.0;
    output <<" "<< avgE_CAplusDB;
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        std::sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[2] )
        + std::sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[2] )
        + std::sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[2] )
        + std::sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[2] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = ((evNN[0]+evNN[1]+evNN[2]+evNN[3]+evNN[4]+evNN[7]) * J
         + (evNN[5]+evNN[6]) * (alpha*J))/4.0; 
    output <<" "<< energy;

    // write components of magentization
    output <<" "<< ev_sA[0] <<" "<< 0.5*(ev_sA[1]+ev_sA[2]) <<" "<< 0.5*(ev_sA[1]-ev_sA[2]);
    output <<" "<< ev_sB[0] <<" "<< 0.5*(ev_sB[1]+ev_sB[2]) <<" "<< 0.5*(ev_sB[1]-ev_sB[2]);
    output <<" "<< ev_sC[0] <<" "<< 0.5*(ev_sC[1]+ev_sC[2]) <<" "<< 0.5*(ev_sC[1]-ev_sC[2]);
    output <<" "<< ev_sD[0] <<" "<< 0.5*(ev_sD[1]+ev_sD[2]) <<" "<< 0.5*(ev_sD[1]-ev_sD[2]);

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> LaddersModel_2x2_ABCD::create(nlohmann::json & json_model) {
    
    double arg_J     = json_model["J1"].get<double>();
    double arg_alpha = json_model["alpha"].get<double>();
    
    return std::unique_ptr<Model>(new LaddersModel_2x2_ABCD(arg_J, arg_alpha));
}

std::unique_ptr<Engine> LaddersModel_2x2_ABCD::buildEngine(nlohmann::json & json_model) {
   
    double arg_J     = json_model["J1"].get<double>();
    double arg_alpha = json_model["alpha"].get<double>();
    double arg_tau   = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_Ladders(arg_tau, arg_J) );
        pe->td.gateMPO.push_back( getMPO2s_Ladders(arg_tau, arg_alpha*arg_J) );
        pe->td.gateMPO[0].uuid = "STRONG";
        pe->td.gateMPO[1].uuid = "WEAK";
        
        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(0,1)}, &pe->td.gateMPO[1])
        };

        std::cout<<"LaddersModel_2x2_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Ladders(arg_tau, arg_J, 1.0) );
        pe->td.gateMPO.push_back( getMPO3s_Ladders(arg_tau, arg_J, arg_alpha) );
        
        pe->td.tgates = {
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[1])
        };

        std::cout<<"LaddersModel_2x2_ABCD SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    else if (arg_fuGateSeq == "4SITE") {
        TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_Ladders(arg_tau, arg_J, 1.0) );
        pe->td.gateMPO.push_back( getOP4s_Ladders(arg_tau, arg_J, arg_alpha) );
        
        pe->td.tgates = {
            TrotterGate<OpNS>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<OpNS>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[1])
        };

        std::cout<<"NNH_2x2Cell_Ladder 4SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}
// ----- END Definition of model class --------------------------------

// ----- Definition of model base class and its particular instances --
LaddersModel_4x2_ABCD::LaddersModel_4x2_ABCD(double arg_J, double arg_alpha)
    : J(arg_J), alpha(arg_alpha) {}

void LaddersModel_4x2_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS A1A2 (0,0)(1,0), "<<"SS A2A3 (1,0)(2,0), "
        <<"SS A3A4 (2,0)(3,0), "<<"SS A4A1 (3,0)(0,0), "
        <<"SS B1B2 (0,1)(1,1), "<<"SS B2B3 (1,1)(2,1), "
        <<"SS B3B4 (2,1)(3,1), "<<"SS B4B1 (3,1)(0,1), "
        <<"SS A1B1 (0,0)(0,1), "<<"SS A2B2 (1,0)(1,1), "
        <<"SS A3B3 (2,0)(2,1), "<<"SS A3B3 (3,0)(3,1), "
        <<"SS B1A1 (0,1)(0,2), "<<"SS B2A2 (1,1)(1,2), "
        <<"SS B3A3 (2,1)(2,2), "<<"SS B3A3 (3,1)(3,2), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void LaddersModel_4x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector< std::vector<double> > szspsm(8,{0.0,0.0,0.0});

    // horizontal
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,0), Vertex(1,0)) ); //A1A2
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,0), Vertex(2,0)) ); //A2A3
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(2,0), Vertex(3,0)) ); //A3A4
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(3,0), Vertex(4,0), true) ); //A4A1
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,1), Vertex(1,1)) ); //B1B2
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,1), Vertex(2,1)) ); //B2B3
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(2,1), Vertex(3,1)) ); //B3B4
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(3,1), Vertex(4,1)) ); //B4B1

    // vertical
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,0), Vertex(0,1)) ); //A1B1
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,0), Vertex(1,1)) ); //A2B2
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(2,0), Vertex(2,1)) ); //A3B3
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(3,0), Vertex(3,1)) ); //A4B4
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(0,1), Vertex(0,2)) ); //B1A1
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(1,1), Vertex(1,2)) ); //B2A2
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(2,1), Vertex(2,2)) ); //B3A3
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        Vertex(3,1), Vertex(3,2)) ); //B4A4
    
    // magnetization
    for(int y=0; y<2; y++) {
        for(int x=0; x<4; x++) {
            auto v = Vertex(x,y);
            szspsm[x+y*4][0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, v);
            szspsm[x+y*4][1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, v);
            szspsm[x+y*4][2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, v);
        }
    }

    // write energy
    output << lineNo; 
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    
    // write magnetization
    double evMag_avg = 0.;
    for (int j = 0; j<8; j++) evMag_avg += (1.0/8.0) * 
        std::sqrt(szspsm[j][0]*szspsm[j][0] + szspsm[j][1]*szspsm[j][1]);
    output <<" "<< evMag_avg;

    // write Energy
    double energy = 0.0;
    for (int j = 0; j<12; j++) energy += (1.0/8.0) * evNN[j] * J;
    for (int j = 12; j<16; j++) energy += (1.0/8.0) * evNN[j] * (J * alpha);
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> LaddersModel_4x2_ABCD::create(nlohmann::json & json_model) {
    
    double arg_J     = json_model["J1"].get<double>();
    double arg_alpha = json_model["alpha"].get<double>();
    
    return std::unique_ptr<Model>(new LaddersModel_4x2_ABCD(arg_J, arg_alpha));
}

std::unique_ptr<Engine> LaddersModel_4x2_ABCD::buildEngine(nlohmann::json & json_model) {
   
    double arg_J     = json_model["J1"].get<double>();
    double arg_alpha = json_model["alpha"].get<double>();
    double arg_tau   = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_Ladders(arg_tau, arg_J) );
        pe->td.gateMPO.push_back( getMPO2s_Ladders(arg_tau, arg_alpha*arg_J) );
        pe->td.gateMPO[0].uuid = "STRONG";
        pe->td.gateMPO[1].uuid = "WEAK";
        
        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(2,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(3,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(2,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(3,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(2,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(3,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_2site>(Vertex(2,1), {Shift(0,1)}, &pe->td.gateMPO[1]),
            TrotterGate<MPO_2site>(Vertex(3,1), {Shift(0,1)}, &pe->td.gateMPO[1])
        };

        std::cout<<"LaddersModel_4x2_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}
// ----- END Definition of model class --------------------------------

} //namespace itensor
