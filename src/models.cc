#include "p-ipeps/config.h"
#include "p-ipeps/models.h"

using namespace itensor;

// ----- Trotter gates (3site, ...) MPOs ------------------------------
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
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Trotter gates (3site, ...) MPOs ------------------------------
OpNS getOP4s_J1Q(double tau, double J1, double Q) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF
    std::cout.precision(10);

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s4 = Index("S4", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);
    Index s4p = prime(s4);

    ITensor h4 = ITensor(s1,s2,s3,s4,s1p,s2p,s3p,s4p);
   
    ITensor nnS1S2 = J1*(SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2)
        + 0.5*( SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2)
        + SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2) ) );

    h4 += nnS1S2 * delta(s3,s3p) * delta(s4,s4p);                                       // S1S2id3id4
    h4 += (nnS1S2 * delta(s1,s3) * delta(s1p,s3p)) * delta(s1,s1p) * delta(s4,s4p);     // id1S2S3id4
    h4 += (nnS1S2 * delta(s2,s4) * delta(s2p,s4p)) * delta(s2,s2p) * delta(s3,s3p);     // S1id2id3S4
    h4 += (nnS1S2 * delta(s2,s4) * delta(s2p,s4p) * delta(s1,s3) * delta(s1p,s3p)) *
        delta(s1,s1p) * delta(s2,s2p);                                                  // id1id2S3S4

    // TODO Q-part of the Hamiltonian

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
J1QModel::J1QModel(double arg_J1, double arg_Q)
    : J1(arg_J1), Q(arg_Q) {}

void J1QModel::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg SS_NN, "<<"Avg SSSS, "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void J1QModel::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> evSSSS;
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

    // compute energies of plaquettes
    evSSSS.push_back( ev.eval2x2op4s(EVBuilder::OP4S_Q, Vertex(0,0)) );
    evSSSS.push_back( ev.eval2x2op4s(EVBuilder::OP4S_Q, Vertex(1,0)) );
    evSSSS.push_back( ev.eval2x2op4s(EVBuilder::OP4S_Q, Vertex(0,1)) );
    evSSSS.push_back( ev.eval2x2op4s(EVBuilder::OP4S_Q, Vertex(1,1)) );

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
    double avgSSSS = 0.;
    for ( unsigned int j=0; j<evSSSS.size(); j++ ) avgSSSS += evSSSS[j];
    avgSSSS = avgSSSS / 4.0;   
    output <<" "<< avgSSSS;

    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25*(
        sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
        + sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )
        + sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )
    );
    output <<" "<< evMag_avg;

    // TODO write Energy
    double energy = 2.0 * avgSS_NN * J1 + 2.0 * avgSSSS * Q; 
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

// Ising3BodyModel::Ising3BodyModel(double arg_J1, double arg_J2, double arg_h)
//     : J1(arg_J1), J2(arg_J2), h(arg_h) {}

// void Ising3BodyModel::setObservablesHeader(std::ofstream & output) {
//     output <<"STEP, " 
//         <<"SzSz AB (0,0)(1,0), "<<"SzSz AC (0,0)(0,1), "
//         <<"SzSz BD (1,0)(1,1), "<<"SzSz CD (0,1)(1,1), "
//         <<"SzSz BA (1,0)(2,0), "<<"SzSz CA (0,1)(0,2), "
//         <<"SzSz DB (1,1)(1,2), "<<"SzSz DC (1,1)(2,1), "
//         <<"Avg SzSz, "<<"Avg SzSzSz, "<<"Avg Sz, "<<"Avg Sx, "<<"Energy"
//         <<std::endl;
// }

// void Ising3BodyModel::computeAndWriteObservables(EVBuilder const& ev, 
//     std::ofstream & output, Args & metaInf) {

//     auto lineNo = metaInf.getInt("lineNo",0);

//     std::vector<double> evNN;
//     std::vector<double> ev3SZ;
//     std::vector<double> ev_sA(3,0.0);
//     std::vector<double> ev_sB(3,0.0);
//     std::vector<double> ev_sC(3,0.0);
//     std::vector<double> ev_sD(3,0.0);

//     evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(0,0), Vertex(1,0)) );    //AB
//     evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(0,0), Vertex(0,1)) );    //AC
//     evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(1,0), Vertex(1,1)) );    //BD
//     evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(0,1), Vertex(1,1)) );    //CD

//     evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(1,0), Vertex(2,0))); //BA
//     evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(0,1), Vertex(0,2))); //CA
//     evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(1,1), Vertex(1,2))); //DB
//     evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
//         Vertex(1,1), Vertex(2,1))); //DC

//     // compute "3-site" terms Sz_i Sz_j Sz_k
//     // 4 triangles centered on site [0,0]
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,-1), Vertex(0,0), Vertex(1,0), Vertex(1,-1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,-1), Vertex(0,0), Vertex(-1,0), Vertex(-1,-1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,1), Vertex(0,0), Vertex(1,0), Vertex(1,1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,1), Vertex(0,0), Vertex(-1,0), Vertex(-1,1)}) );
//     // 4 triangles centered on site [1,0]
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,-1), Vertex(1,0), Vertex(2,0), Vertex(2,-1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,-1), Vertex(1,0), Vertex(0,0), Vertex(0,-1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,1), Vertex(1,0), Vertex(2,0), Vertex(2,1)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,1), Vertex(1,0), Vertex(0,0), Vertex(0,1)}) );
//     // 4 triangles centered on site [0,1]
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,0), Vertex(0,1), Vertex(1,1), Vertex(1,0)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,0), Vertex(0,1), Vertex(-1,1), Vertex(-1,0)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,2), Vertex(0,1), Vertex(1,1), Vertex(1,2)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(0,2), Vertex(0,1), Vertex(-1,1), Vertex(-1,2)}) );
//     // 4 triangles centered on site [1,1]
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,0), Vertex(1,1), Vertex(2,1), Vertex(2,0)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,0), Vertex(1,1), Vertex(0,1), Vertex(0,0)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,2), Vertex(1,1), Vertex(2,1), Vertex(2,2)}) );
//     ev3SZ.push_back(ev.contract3Smpo2x2( ev.get3Smpo("3SZ"),
//         {Vertex(1,2), Vertex(1,1), Vertex(0,1), Vertex(0,2)}) );
//     // end computing "3-site" terms Sz_i Sz_j Sz_k

//     ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
//     ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
//     ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

//     ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
//     ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
//     ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

//     ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,1));
//     ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,1));
//     ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,1));

//     ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,1));
//     ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,1));
//     ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,1));

//     // write energy
//     double avgE_8links = 0.;
//     output << lineNo <<" "; 
//     for ( unsigned int j=evNN.size()-8; j<evNN.size(); j++ ) {
//         avgE_8links += evNN[j];
//         output<<" "<< evNN[j];
//     }
//     avgE_8links = avgE_8links/8.0;
//     output <<" "<< avgE_8links;
    
//     // write energy
//     double avgE_3sz = 0.;
//     for ( unsigned int j=0; j<ev3SZ.size(); j++ ) {
//         avgE_3sz += ev3SZ[j];
//     }
//     avgE_3sz = avgE_3sz/16.0;
//     output <<" "<< avgE_3sz;

//     // write Z magnetization
//     double evMagZ_avg = 0.;
//     double evMagX_avg = 0.;
//     evMagZ_avg = 0.25*(
//         sqrt(ev_sA[0]*ev_sA[0])
//         + sqrt(ev_sB[0]*ev_sB[0])
//         + sqrt(ev_sC[0]*ev_sC[0])
//         + sqrt(ev_sD[0]*ev_sD[0])
//         );
//     output <<" "<< evMagZ_avg;
//     evMagX_avg = 0.25*(
//         sqrt(ev_sA[1]*ev_sA[1])
//         + sqrt(ev_sB[1]*ev_sB[1])
//         + sqrt(ev_sC[1]*ev_sC[1])
//         + sqrt(ev_sD[1]*ev_sD[1])
//         );
//     output <<" "<< evMagX_avg;

//     // write Energy 
//     // * working with spin DoFs instead of Ising DoFs hence factor of 2
//     double energy = -4.0*(8.0*avgE_8links) * J1 - 4.0 * 2.0 * h * evMagX_avg
//         -8.0*(16.0*avgE_3sz) * J2;
//     output <<" "<< energy; 

//     output << std::endl;
// }

// ----- END Definition of model class --------------------------------

// ----- Model Definitions -------------------------------------------

std::unique_ptr<Model> getModel_J1Q(nlohmann::json & json_model) {
    double arg_J1 = json_model["J1"].get<double>();
    double arg_Q = json_model["Q"].get<double>();
    
    return std::unique_ptr<Model>(new J1QModel(arg_J1, arg_Q));
}

// std::unique_ptr<Model> getModel_Ising3Body(nlohmann::json & json_model) {

//     double arg_J1     = json_model["J1"].get<double>();
//     double arg_J2     = json_model["J2"].get<double>();
//     double arg_h      = json_model["h"].get<double>();
    
//     return std::unique_ptr<Model>(new Ising3BodyModel(arg_J1, arg_J2, arg_h));
// }

// void getModel_Ising3Body(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_3site > & gateMPO,
//     std::vector< MPO_3site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds) {

//     double arg_J1     = json_model["J1"].get<double>();
//     double arg_J2     = json_model["J2"].get<double>();
//     double arg_h      = json_model["h"].get<double>();
//     double arg_lambda = json_model["LAMBDA"].get<double>();
    
//     ptr_model = std::unique_ptr<Model>(new Ising3BodyModel(arg_J1, arg_J2, arg_h));

//     // time step
//     double arg_tau    = json_model["tau"].get<double>();
//     // gate sequence
//     std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

//     if (arg_fuGateSeq == "SYM1") {
//         gates = {
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
//         };

//         gate_auxInds = {
//             {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
//             {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

//             {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
//             {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

//             {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
//             {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
//             {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
//         };

//         gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1/4.0, arg_J2, arg_h/12.0) );
//         ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
//     } else if (arg_fuGateSeq == "SYM3") {
//         gates = {
//             {"A", "B", "D", "C"},
//             {"C", "D", "B", "A"},
//             {"D", "C", "A", "B"},
//             {"B", "A", "C", "D"},

//             {"B", "A", "C", "D"},
//             {"D", "C", "A", "B"},
//             {"C", "D", "B", "A"},
//             {"A", "B", "D", "C"},

//             {"D", "C", "A", "B"},
//             {"B", "A", "C", "D"},
//             {"A", "B", "D", "C"},
//             {"C", "D", "B", "A"},

//             {"C", "D", "B", "A"}, 
//             {"A", "B", "D", "C"},
//             {"B", "A", "C", "D"},
//             {"D", "C", "A", "B"}
//         };

//         gate_auxInds = {
//             {3,2, 0,3, 1,0, 2,1},
//             {3,0, 2,3, 1,2, 0,1},
//             {3,0, 2,3, 1,2, 0,1},
//             {3,2, 0,3, 1,0, 2,1},

//             {3,0, 2,3, 1,2, 0,1},
//             {3,2, 0,3, 1,0, 2,1},
//             {3,2, 0,3, 1,0, 2,1},
//             {3,0, 2,3, 1,2, 0,1},

//             {1,0, 2,1, 3,2, 0,3},
//             {1,2, 0,1, 3,0, 2,3},
//             {1,2, 0,1, 3,0, 2,3}, 
//             {1,0, 2,1, 3,2, 0,3},

//             {1,2, 0,1, 3,0, 2,3},
//             {1,0, 2,1, 3,2, 0,3},
//             {1,0, 2,1, 3,2, 0,3},
//             {1,2, 0,1, 3,0, 2,3}
//         };

//         gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1/4.0, arg_J2, arg_h/12.0) );
//         ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
//     } else {
//         std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// void getModel_3site(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_3site > & gateMPO,
//     std::vector< MPO_3site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds) {

//     std::string arg_modelType = json_model["type"].get<std::string>(); 

//     if(arg_modelType == "J1J2") {
//         getModel_J1J2(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else if (arg_modelType == "NNHLadder") {
//         getModel_NNHLadder(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else if (arg_modelType == "Ising") {
//         getModel_Ising(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else if (arg_modelType == "Ising3Body") {
//         getModel_Ising3Body(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else {
//         std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }


std::unique_ptr<Model> getModel(nlohmann::json & json_model) {

    std::string arg_modelType = json_model["type"].get<std::string>(); 

    if(arg_modelType == "J1Q") {
        return getModel_J1Q(json_model);
    // } else if (arg_modelType == "Ising3Body") {
    //     return getModel_Ising3Body(json_model);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}
// ----- END Model Definitions ----------------------------------------
