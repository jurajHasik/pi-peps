#include "id-2x2-ABCD.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
// TODO implement more generic external field
MPO_2site getMPO2s_ID(int pd) {
    int physDim = 5; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);

    auto idTs1 = ITensor(s1,s1p);
    for (int i=1; i<=physDim; i++) idTs1.set(s1(i),s1p(i),1.0);

    ITensor u12 = idTs1 * delta(s2,s2p);

    return symmMPO2Sdecomp(u12, s1, s2);
}

// TODO implement more generic external field
MPO_3site getMPO3s_ID(int pd) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    auto idTs1 = ITensor(s1,s1p);
    for (int i=1; i<=physDim; i++) idTs1.set(s1(i),s1p(i),1.0);

    ITensor u123 = (idTs1 * delta(s2,s2p)) * delta(s3,s3p);
    
    return symmMPO3Sdecomp(u123, s1, s2, s3);
}

// TODO implement more generic external field
OpNS getOP4s_ID(int pd) {
    int physDim = 2; // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s4 = Index("S4", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);
    Index s4p = prime(s4);

    auto idTs1 = ITensor(s1,s1p);
    for (int i=1; i<=physDim; i++) idTs1.set(s1(i),s1p(i),1.0);
   
    ITensor u4 = ((idTs1 * delta(s2,s2p)) * delta(s3,s3p)) * delta(s4,s4p);

    auto op4s = OpNS(4);

    op4s.op = u4;
    op4s.pi = {s1,s2,s3,s4};

    return op4s;
}
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
IdentityModel_2x2_ABCD::IdentityModel_2x2_ABCD(int pd) { physDim = pd; }

void IdentityModel_2x2_ABCD::setObservablesHeader(std::ofstream & output) {}

void IdentityModel_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {}

std::unique_ptr<Model> IdentityModel_2x2_ABCD::create(nlohmann::json & json_model) {
    
    auto arg_physDim = json_model["physDim"].get<int>();

    return std::unique_ptr<Model>(new IdentityModel_2x2_ABCD(arg_physDim));
}

std::unique_ptr<Engine> IdentityModel_2x2_ABCD::buildEngine(nlohmann::json & json_model) {
    
    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);
    auto arg_physDim = json_model["physDim"].get<int>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    
    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_ID(arg_physDim) );
        
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

        std::cout<<"IdentityModel_2x2_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_ID(arg_physDim) );
        
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

        std::cout<<"IdentityModel_2x2_ABCD SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe ); 
    }
    else if (arg_fuGateSeq == "4SITE") {
        TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_ID(arg_physDim) );
        
        pe->td.tgates = {
            TrotterGate<OpNS>(Vertex(0,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(1,0), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(0,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<OpNS>(Vertex(1,1), {Shift(1,0), Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"IdentityModel_2x2_ABCD 4SITE ENGINE constructed"<<std::endl;
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
IdentityModel_2x2_AB::IdentityModel_2x2_AB(int pd) { physDim = pd; }

void IdentityModel_2x2_AB::setObservablesHeader(std::ofstream & output) {}

void IdentityModel_2x2_AB::computeAndWriteObservables(EVBuilder const& ev, 
    std::ofstream & output, Args & metaInf) {}

std::unique_ptr<Model> IdentityModel_2x2_AB::create(nlohmann::json & json_model) {
    
    auto arg_physDim = json_model["physDim"].get<int>();

    return std::unique_ptr<Model>(new IdentityModel_2x2_AB(arg_physDim));
}

std::unique_ptr<Engine> IdentityModel_2x2_AB::buildEngine(nlohmann::json & json_model) {
    
    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);
    auto arg_physDim = json_model["physDim"].get<int>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    
    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_ID(arg_physDim) );
        
        pe->td.tgates = {
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(1,0)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(0,0), {Shift(0,1)}, &pe->td.gateMPO[0]),
            TrotterGate<MPO_2site>(Vertex(1,0), {Shift(0,1)}, &pe->td.gateMPO[0])
        };

        std::cout<<"IdentityModel_2x2_AB 2SITE ENGINE constructed"<<std::endl;
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