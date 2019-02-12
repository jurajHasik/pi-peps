#include "ising-2x2-ABCD.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_Ising_2site(double tau, double J, double h) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    // STEP 1 define exact h_12 = -J(Sz_1.Sz_2) - h(Sx_1+Sx_2)
    ITensor h12 = ITensor(s1,s2,s1p,s2p);
    h12 += -J*( 2*SU2_getSpinOp(SU2_S_Z, s1) * 2*SU2_getSpinOp(SU2_S_Z, s2) );
    h12 += -h*( (SU2_getSpinOp(SU2_S_P, s1) + SU2_getSpinOp(SU2_S_M, s1))*delta(s2,s2p)
        + delta(s1,s1p)*(SU2_getSpinOp(SU2_S_P, s2)+SU2_getSpinOp(SU2_S_M, s2)) );

    // STEP 2 compute exponential u_12 = exp(-tau h_12)
    auto cmbI = combiner(s1,s2);
    h12 = (cmbI * h12 ) * prime(cmbI);
    ITensor u12 = expHermitian(h12, {-tau, 0.0});
    u12 = (cmbI * u12 ) * prime(cmbI);
    // definition of U_12 done

    return symmMPO2Sdecomp(u12, s1, s2);
}

MPO_3site getMPO3s_Ising_3site(double tau, double J, double h) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

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

// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
IsingModel_2x2_ABCD::IsingModel_2x2_ABCD(double arg_J1, double arg_h)
    : J1(arg_J1), h(arg_h) {}

void IsingModel_2x2_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SzSz AB (0,0)(1,0), "<<"SzSz AC (0,0)(0,1), "
        <<"SzSz BD (1,0)(1,1), "<<"SzSz CD (0,1)(1,1), "
        <<"SzSz BA (1,0)(2,0), "<<"SzSz CA (0,1)(0,2), "
        <<"SzSz DB (1,1)(1,2), "<<"SzSz DC (1,1)(2,1), "
        <<"Avg SzSz, "<<"Avg Sz, "<<"Avg Sx, "<<"Energy"
        <<std::endl;
}

void IsingModel_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,0), Vertex(1,0)) );    //AB
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,0), Vertex(0,1)) );    //AC
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,0), Vertex(1,1)) );    //BD
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,1), Vertex(1,1)) );    //CD

    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,0), Vertex(2,0))); //BA
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,1), Vertex(0,2))); //CA
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,1), Vertex(1,2))); //DB
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,1), Vertex(2,1))); //DC

    ev_sA[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
    ev_sA[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
    ev_sA[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

    ev_sB[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
    ev_sB[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
    ev_sB[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

    ev_sC[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,1));
    ev_sC[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,1));
    ev_sC[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,1));

    ev_sD[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,1));
    ev_sD[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,1));
    ev_sD[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,1));

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
    double evMagZ_avg = 0.25*(ev_sA[0] + ev_sB[0] + ev_sC[0] + ev_sD[0]);
    double evMagX_avg = 0.25*(ev_sA[1] + ev_sB[1] + ev_sC[1] + ev_sD[1]);
    output <<" "<< evMagZ_avg;
    output <<" "<< evMagX_avg;

    // write Energy per site
    // * working with spin DoFs instead of Ising DoFs hence factor of 2
    double energy = -J1*(2.0*avgE_8links) - h * evMagX_avg;
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> IsingModel_2x2_ABCD::create(nlohmann::json & json_model) {

    double arg_J1    = json_model["J1"].get<double>();
    double arg_h = json_model["h"].get<double>();

    return std::unique_ptr<Model>(new IsingModel_2x2_ABCD(arg_J1, arg_h));
}

std::unique_ptr<Engine> IsingModel_2x2_ABCD::buildEngine(nlohmann::json & json_model) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_tau    = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_Ising_2site(arg_tau, arg_J1, arg_h/4.0) );

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

        std::cout<<"IsingModel_2x2_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        // complete sequence of gates acts 4 times on each NN link J->J/4.0
        // complete sequence of gates acts applies Sx term 12 times on each site h->h/12.0 

        pe->td.gateMPO.push_back( getMPO3s_Ising_3site(arg_tau, arg_J1/4.0, arg_h/12.0) );
        
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

        std::cout<<"IsingModel_2x2_ABCD SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    else {
        std::cout<<"[IsingModel_2x2_ABCD] Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}
// ----- END Definition of model class --------------------------------

// ----- Model Definitions --------------------------------------------
IsingModel_2x2_AB::IsingModel_2x2_AB(double arg_J1, double arg_h)
    : J1(arg_J1), h(arg_h) {}

void IsingModel_2x2_AB::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SzSz AB (0,0)(1,0), "<<"SzSz BA (1,0)(2,0), "
        <<"SzSz AB (0,0)(0,1), "<<"SzSz BA (1,0)(1,1), "
        <<"Avg SzSz, "<<"Avg Sz, "<<"Avg Sx, "<<"Energy"
        <<std::endl;
}

void IsingModel_2x2_AB::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);

    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,0), Vertex(1,0)) );    //A-2--0-B
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,0), Vertex(2,0)) );    //B-2--0-A
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(0,0), Vertex(0,1)) );    //A-3--1-B
    evNN.push_back( 4.0*ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
        Vertex(1,0), Vertex(1,1)) );    //B-3--1-A

    ev_sA[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
    ev_sA[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
    ev_sA[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

    ev_sB[0] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
    ev_sB[1] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
    ev_sB[2] = 2.0*ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

    // write energy
    double avgE_4links = 0.;
    output << lineNo <<" "; 
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        avgE_4links += evNN[j];
        output<<" "<< evNN[j];
    }
    avgE_4links = avgE_4links/4.0;
    output <<" "<< avgE_4links;
    
    // write Z magnetization
    double evMagZ_avg = 0.5*(ev_sA[0] + ev_sB[0]);
    double evMagX_avg = 0.5*(ev_sA[1] + ev_sB[1]);
    output <<" "<< evMagZ_avg;
    output <<" "<< evMagX_avg;

    // write Energy per site
    // * working with spin DoFs instead of Ising DoFs hence factor of 2
    double energy = -J1*(2.0*avgE_4links) - h * evMagX_avg;
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> IsingModel_2x2_AB::create(nlohmann::json & json_model) {

    double arg_J1    = json_model["J1"].get<double>();
    double arg_h = json_model["h"].get<double>();

    return std::unique_ptr<Model>(new IsingModel_2x2_AB(arg_J1, arg_h));
}

std::unique_ptr<Engine> IsingModel_2x2_AB::buildEngine(nlohmann::json & json_model) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_tau    = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_Ising_2site(arg_tau, arg_J1, arg_h/4.0) );

        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"IsingModel_2x2_AB 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"[IsingModel_2x2_AB] Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}
// ----- END Definition of model class --------------------------------

} //namespace itensor