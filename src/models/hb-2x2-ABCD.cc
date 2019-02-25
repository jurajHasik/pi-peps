#include "pi-peps/config.h"
#include "pi-peps/models/hb-2x2-ABCD.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
// TODO implement more generic external field
MPO_2site getMPO2s_HB(double tau, double J, double h, double del) {
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
    h12 += del*(SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2));
    h12 += -h*(SU2_getSpinOp(SU2_S_Z, s1)*delta(s2,s2p) 
    	+ delta(s1,s1p)*SU2_getSpinOp(SU2_S_Z, s2));

    auto cmbI = combiner(s1,s2);
    h12 = (cmbI * h12 ) * prime(cmbI);
    ITensor u12 = expHermitian(h12, {-tau, 0.0});
    u12 = (cmbI * u12 ) * prime(cmbI);
    // definition of U_12 done

    return symmMPO2Sdecomp(u12, s1, s2);
}

// TODO implement more generic external field
MPO_3site getMPO3s_HB(double tau, double J, double h, double del) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    ITensor h123 = ITensor(s1,s2,s3,s1p,s2p,s3p);
   
   	// del:  XY limit -1 <-- 0 (HB)-->--> >>1 Ising limit
    // STEP 1 define exact h_123 = J*(S_1.S_2 + S_2.S_3) + del*(Sz_1Sz_2 + Sz_2Sz_3)
    // - h*(Sz_1+Sz_2+Sz_3)
    ITensor nnS1S2 = (J+del)* SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2) 
        + J*0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) );

    // Nearest-neighbour terms S_1.S_2 and S_2.S_3
    h123 += nnS1S2 * delta(s3,s3p);
    h123 += (nnS1S2 * delta(s1,s3) * delta(s1p,s3p)) * delta(s1,s1p);

    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI);
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);
    
    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

// TODO implement more generic external field
OpNS getOP4s_HB(double tau, double J, double h, double del) {
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
   
   	// del:  XY limit -1 <-- 0 (HB)-->--> >>1 Ising limit
    // STEP 1 define exact h4 = J*(S_1.S_2 + S_2.S_3 + S_3.S_4 + S_4.S_1) 
    // + del*(Sz_1Sz_2 + Sz_2Sz_3 + Sz_3Sz_4 + Sz_4Sz_1)
    // - h*(Sz_1+Sz_2+Sz_3+Sz_4)
    ITensor nnS1S2 = (J+del) * SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + J*0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) );

    h4 += nnS1S2 * delta(s3,s3p) * delta(s4,s4p);                                       // S1S2id3id4
    h4 += (nnS1S2 * delta(s1,s3) * delta(s1p,s3p)) * delta(s1,s1p) * delta(s4,s4p);     // id1S2S3id4
    h4 += (nnS1S2 * delta(s2,s4) * delta(s2p,s4p)) * delta(s2,s2p) * delta(s3,s3p);     // S1id2id3S4
    h4 += (nnS1S2 * delta(s2,s4) * delta(s2p,s4p) * delta(s1,s3) * delta(s1p,s3p)) *
        delta(s1,s1p) * delta(s2,s2p);                                                  // id1id2S3S4

    h4 += h*SU2_getSpinOp(SU2_S_Z, s1)*delta(s2,s2p)*delta(s3,s3p)*delta(s4,s4p); 
    h4 += h*delta(s1,s1p)*SU2_getSpinOp(SU2_S_Z, s2)*delta(s3,s3p)*delta(s4,s4p);
    h4 += h*delta(s1,s1p)*delta(s2,s2p)*SU2_getSpinOp(SU2_S_Z, s3)*delta(s4,s4p);
    h4 += h*delta(s1,s1p)*delta(s2,s2p)*delta(s3,s3p)*SU2_getSpinOp(SU2_S_Z, s4);

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
HeisenbergModel_2x2_ABCD::HeisenbergModel_2x2_ABCD(double arg_J, double arg_h,
    double arg_del) : J(arg_J), h(arg_h), del(arg_del) {}

void HeisenbergModel_2x2_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void HeisenbergModel_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    // construct nn S.S operator
    Index s1 = Index("S1", 2, PHYS);
    Index s2 = Index("S2", 2, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    ITensor h12 = ITensor(s1,s2,s1p,s2p);
    h12 += (J+del)* SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + J*0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) );

    // Perform SVD to split in half
    auto nnSS = symmMPO2Sdecomp(h12, s1, s2);

    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,0), Vertex(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,0), Vertex(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(1,0), Vertex(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,1), Vertex(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(1,0), Vertex(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,1), Vertex(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(1,1), Vertex(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
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
    output << lineNo <<" "; 
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        std::sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + std::sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
        + std::sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )
        + std::sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = 0.25*( (evNN[0]+evNN[1]+evNN[2]+evNN[3]+evNN[4]+evNN[5]+evNN[6]+evNN[7]) 
         + (ev_sA[0] + ev_sB[0] + ev_sC[0] + ev_sD[0]) * h);
    output <<" "<< energy; 

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> HeisenbergModel_2x2_ABCD::create(nlohmann::json & json_model) {

    double arg_J   = json_model["J1"].get<double>();
    double arg_h   = json_model["h"].get<double>();
    double arg_del = json_model["del"].get<double>();
    
    return std::unique_ptr<Model>(new HeisenbergModel_2x2_ABCD(arg_J, arg_h, arg_del));
}

std::unique_ptr<Engine> HeisenbergModel_2x2_ABCD::buildEngine(nlohmann::json & json_model) {
    
    double arg_J   = json_model["J1"].get<double>();
    double arg_h   = json_model["h"].get<double>();
    double arg_del = json_model["del"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    
    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_HB(arg_tau, arg_J, arg_h/4.0, arg_del) );

        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"HeisenbergModel_2x2_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_HB(arg_tau, arg_J/4.0, arg_h/12.0, arg_del/4.0) );

        pe->td.tgates = {
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(-1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(1,0), Shift(0,-1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,0), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,0), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(0,1), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_3site>(Vertex(1,1), {Shift(-1,0), Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"HeisenbergModel_2x2_ABCD SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe ); 
    }
    else if (arg_fuGateSeq == "4SITE") {
        TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_HB(arg_tau, arg_J/2.0, arg_h/4.0, arg_del/2.0) );

        pe->td.tgates = {
            TrotterGate<OpNS>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"HeisenbergModel_2x2_ABCD 4SITE ENGINE constructed"<<std::endl;
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
HeisenbergModel_2x2_AB::HeisenbergModel_2x2_AB(double arg_J, double arg_h,
    double arg_del) : J(arg_J), h(arg_h), del(arg_del) {}

void HeisenbergModel_2x2_AB::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS BA (1,0)(2,0), "
        <<"SS AB (0,0)(0,1), "<<"SS BA (1,0)(1,1), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void HeisenbergModel_2x2_AB::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);

    // construct nn S.S operator
    Index s1 = Index("S1", 2, PHYS);
    Index s2 = Index("S2", 2, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    ITensor h12 = ITensor(s1,s2,s1p,s2p);
    h12 += (J+del)* SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + J*0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) );

    // Perform SVD to split in half
    auto nnSS = symmMPO2Sdecomp(h12, s1, s2);

    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,0), Vertex(1,0)) ); //A-2--0-B
    evNN.push_back( ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(1,0), Vertex(2,0)) ); //B-2--0-A

    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(0,0), Vertex(0,1))); //A-3--1-B
    evNN.push_back(ev.eval2Smpo(std::make_pair(nnSS.H1,nnSS.H2),
        Vertex(1,0), Vertex(1,1))); //B-3--1-A

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

    // write energy
    output << lineNo <<" "; 
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        output<<" "<< evNN[j];
    }
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.5*(
        std::sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + std::sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = 0.5*( (evNN[0]+evNN[1]+evNN[2]+evNN[3]) 
         + (ev_sA[0] + ev_sB[0]) * h);
    output <<" "<< energy; 

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> HeisenbergModel_2x2_AB::create(nlohmann::json & json_model) {

    double arg_J   = json_model["J1"].get<double>();
    double arg_h   = json_model["h"].get<double>();
    double arg_del = json_model["del"].get<double>();
    
    return std::unique_ptr<Model>(new HeisenbergModel_2x2_AB(arg_J, arg_h, arg_del));
}

std::unique_ptr<Engine> HeisenbergModel_2x2_AB::buildEngine(nlohmann::json & json_model) {
    
    double arg_J   = json_model["J1"].get<double>();
    double arg_h   = json_model["h"].get<double>();
    double arg_del = json_model["del"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    
    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_HB(arg_tau, arg_J, arg_h/4.0, arg_del) );
        
        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"HeisenbergModel_2x2_AB 2SITE ENGINE constructed"<<std::endl;
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
