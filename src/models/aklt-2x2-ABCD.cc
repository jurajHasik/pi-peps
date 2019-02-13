#include "aklt-2x2-ABCD.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
// TODO implement more generic external field
ITensor Projector_S2tpS2_S1(Index & s1, Index & s2) {
    int physDim = 5; // dimension of Hilbert space of spin s=2 DoF
    
    s1 = Index("S1", physDim, PHYS);
    s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    auto p = ITensor(s1, s2, s1p, s2p);
    // we use Dyknin notation to label representation
    // Loop over <bra| indices
    int rS = physDim-1; // Label of SU(2) irrep 
    int mbA, mbB, mkA, mkB;
    double hVal;
    for(int bA=1;bA<=physDim;bA++) {
    for(int bB=1;bB<=physDim;bB++) {
        // Loop over |ket> indices
        for(int kA=1;kA<=physDim;kA++) {
        for(int kB=1;kB<=physDim;kB++) {
            // Use Dynkin notation to specify irreps
            mbA = -(rS) + 2*(bA-1);
            mbB = -(rS) + 2*(bB-1);
            mkA = -(rS) + 2*(kA-1);
            mkB = -(rS) + 2*(kB-1);
            // Loop over possible values of m given by tensor product
            // of 2 spin (physDim-1) irreps
            hVal = 0.0;
            for(int m=-2*(rS);m<=2*(rS);m=m+2) {
                if ((mbA+mbB == m) && (mkA+mkB == m)) {
                    
                //DEBUG
                // if(dbg) std::cout <<"<"<< mbA <<","<< mbB <<"|"<< m 
                //     <<"> x <"<< m <<"|"<< mkA <<","<< mkB 
                //     <<"> = "<< SU2_getCG(rS, rS, 2*rS, mbA, mbB, m)
                //     <<" x "<< SU2_getCG(rS, rS, 2*rS, mkA, mkB, m)
                //     << std::endl;

                hVal += SU2_getCG(rS, rS, 2*rS, mbA, mbB, m) 
                    *SU2_getCG(rS, rS, 2*rS, mkA, mkB, m);
                }
            }
            if((bA == kA) && (bB == kB)) {
                // add 2*Id(bA,kA;bB,kB) == 
                //    sqrt(2)*Id(bA,kA)(x)sqrt(2)*Id(bB,kB)
                p.set(s1(kA),s2(kB),s1p(bA),s2p(bB),hVal);
            } else {
                p.set(s1(kA),s2(kB),s1p(bA),s2p(bB),hVal);
            }
        }}
    }}

    return p;
}

MPO_2site getMPO2s_AKLT(double tau) {
    Index s1,s2;

    auto p12 = Projector_S2tpS2_S1(s1,s2);
    
    auto cmbI = combiner(s1,s2);
    p12 = (cmbI * p12 ) * prime(cmbI);
    ITensor u12 = expHermitian(p12, {-tau, 0.0});
    u12 = (cmbI * u12 ) * prime(cmbI);
    // definition of U_12 done

    return symmMPO2Sdecomp(u12, s1, s2);
}

MPO_3site getMPO3s_AKLT(double tau) {
    Index s1,s2;
    auto p12 = Projector_S2tpS2_S1(s1,s2);

    Index s3 = Index("S3", s1.m(), PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    auto h123 = ITensor(s1, s2, s3, s1p, s2p, s3p);

    h123 = p12 * delta(s3,s3p);
    h123 += (p12 * delta(s1,s3) * delta(s1p,s3p)) * delta(s1,s1p);

    auto cmbI = combiner(s1,s2,s3);
    h123 = (cmbI * h123 ) * prime(cmbI);
    ITensor u123 = expHermitian(h123, {-tau, 0.0});
    u123 = (cmbI * u123 ) * prime(cmbI);

    return symmMPO3Sdecomp(u123, s1, s2, s3);
}
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
AKLTModel_2x2_ABCD::AKLTModel_2x2_ABCD() {}

void AKLTModel_2x2_ABCD::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS AC (0,0)(0,1), "
        <<"SS BD (1,0)(1,1), "<<"SS CD (0,1)(1,1), "
        <<"SS BA (1,0)(2,0), "<<"SS CA (0,1)(0,2), "
        <<"SS DB (1,1)(1,2), "<<"SS DC (1,1)(2,1), "
        <<"Avg mag=|S|, "<<"Energy"<<std::endl;
}

void AKLTModel_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,0), Vertex(1,0)) ); //AB
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,0), Vertex(0,1)) ); //AC
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(1,0), Vertex(1,1)) ); //BD
    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,1), Vertex(1,1)) ); //CD

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(1,0), Vertex(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,1), Vertex(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(1,1), Vertex(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
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
    output << lineNo; 
    double evNN_avg =0.0;
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        evNN_avg += evNN[j];
        output<<" "<< evNN[j];
    }
    evNN_avg = evNN_avg / evNN.size();
    
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
    double energy = 2.0*evNN_avg;
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> AKLTModel_2x2_ABCD::create(nlohmann::json & json_model) {
    return std::unique_ptr<Model>(new AKLTModel_2x2_ABCD());
}

std::unique_ptr<Engine> AKLTModel_2x2_ABCD::buildEngine(nlohmann::json & json_model) {
    
    double arg_tau = json_model["tau"].get<double>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_AKLT(arg_tau) );
        
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

        std::cout<<"AKLT 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_AKLT(arg_tau) );
        
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

        std::cout<<"AKLT SYM3 ENGINE constructed"<<std::endl;
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
AKLTModel_2x2_AB::AKLTModel_2x2_AB() {}

void AKLTModel_2x2_AB::setObservablesHeader(std::ofstream & output) {
    output <<"STEP, " 
        <<"SS AB (0,0)(1,0), "<<"SS BA (1,0)(2,0), "
        <<"SS AB (0,0)(0,1), "<<"SS BA (1,0)(1,1), "
        <<"Avg mag=|S|, "<<"Energy"
        <<std::endl;
}

void AKLTModel_2x2_AB::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {

    auto lineNo = metaInf.getInt("lineNo",0);

    std::vector<double> evNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);

    evNN.push_back( ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,0), Vertex(1,0))); //A-2--0-B
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(1,0), Vertex(2,0))); //B-2--0-A
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(0,0), Vertex(0,1))); //A-3--1-B
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_AKLT_S2_H,
        Vertex(1,0), Vertex(1,1))); //B-3--1-A

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0,0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0,0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0,0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1,0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1,0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1,0));

    // write energy
    output << lineNo; 
    double evNN_avg =0.0;
    for ( unsigned int j=0; j<evNN.size(); j++ ) {
        evNN_avg += evNN[j];
        output<<" "<< evNN[j];
    }
    evNN_avg = evNN_avg / evNN.size();
    
    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.5*(
        std::sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )
        + std::sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )
    );
    output <<" "<< evMag_avg;

    // write Energy
    double energy = 2.0*evNN_avg;
    output <<" "<< energy;

    // return energy in metaInf
    metaInf.add("energy",energy);

    output << std::endl;
}

std::unique_ptr<Model> AKLTModel_2x2_AB::create(nlohmann::json & json_model) {
    return std::unique_ptr<Model>(new AKLTModel_2x2_AB());
}

std::unique_ptr<Engine> AKLTModel_2x2_AB::buildEngine(nlohmann::json & json_model) {
    
    double arg_tau = json_model["tau"].get<double>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_AKLT(arg_tau) );
        
        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"AKLT 2SITE ENGINE constructed"<<std::endl;
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