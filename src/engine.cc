#include "engine.h"

using namespace itensor;

template <class T>
void TrotterDecomposition<T>::symmetrize() {
    // For symmetric Trotter decomposition
    if ( !symmetrized ) {
        int init_gate_size = gates.size();
        for (int i=0; i<init_gate_size; i++) {
            ptr_gateMPO.push_back(ptr_gateMPO[init_gate_size-1-i]);
            gates.push_back(gates[init_gate_size-1-i]);
            gate_auxInds.push_back(gate_auxInds[init_gate_size-1-i]);
        }
    }
    // can't symmetrize twice
    symmetrized = true;

    std::cout<<"TrotterDecomposition symmetrized"<<std::endl;
}

template <class T>
int TrotterDecomposition<T>::nextCyclicIndex() {
    currentPosition = (currentPosition + 1) % gates.size();
    return currentPosition;
}

std::unique_ptr<Engine> buildEngine_NNH_2x2Cell_Ladder(nlohmann::json & json_model) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_alpha = json_model["alpha"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    // if (arg_fuGateSeq == "SYM1") {
    //     gates = {
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
    //     };

    //     gate_auxInds = {
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
    //     };

    //     gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_J1) );
    //     gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_alpha*arg_J1) );

    //     for (int i=0; i<8; i++) ptr_gateMPO.push_back( &(gateMPO[0]) ); 
    //     for (int i=0; i<8; i++) ptr_gateMPO.push_back( &(gateMPO[1]) );
    // } 
    if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_J1) );
        pe->td.gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, arg_alpha*arg_J1) );
        

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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


        pe->td.ptr_gateMPO = {
            &(pe->td.gateMPO[0]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[0]),
            &(pe->td.gateMPO[0]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[0]),
            &(pe->td.gateMPO[0]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[0]),
            &(pe->td.gateMPO[0]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[1]), &(pe->td.gateMPO[0])
        };

        std::cout<<"NNH_2x2Cell_Ladder SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    else if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, 0.0) );
        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_alpha*arg_J1, 0.0) );
        pe->td.gateMPO[0].uuid = "STRONG";
        pe->td.gateMPO[1].uuid = "WEAK";
        
        pe->td.gates = {
            {"A", "B"}, {"B", "A"}, 
            {"C", "D"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };
        
        for (int i=0; i<6; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );
        for (int i=0; i<2; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[1]) );

        std::cout<<"NNH_2x2Cell_Ladder 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "2SITE-ALT") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, 0.0) );
        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_alpha*arg_J1, 0.0) );
        pe->td.gateMPO[0].uuid = "STRONG";
        pe->td.gateMPO[1].uuid = "WEAK";
        
        pe->td.gates = {
            {"A", "B"}, {"C", "D"}, 
            {"B", "A"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };
        
        for (int i=0; i<6; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );
        for (int i=0; i<2; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[1]) );

        std::cout<<"NNH_2x2Cell_Ladder 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "4SITE") {
         TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_Uladder(arg_tau, arg_J1, arg_J1) );
        pe->td.gateMPO.push_back( getOP4s_Uladder(arg_tau, arg_J1, arg_alpha*arg_J1) );
        
        pe->td.gates = {
            {"B", "A", "C", "D"},
            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},
            {"D", "C", "A", "B"}
        };

        pe->td.gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1}
        };
        
        for (int i=0; i<2; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );
        for (int i=0; i<2; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[1]) );

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

std::unique_ptr<Engine> buildEngine_NNH_4x2Cell_Ladder(nlohmann::json & json_model) {

    double arg_J1 = json_model["J1"].get<double>();
    double arg_alpha = json_model["alpha"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, 0.0) );
        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_alpha*arg_J1, 0.0) );
        pe->td.gateMPO[0].uuid = "STRONG";
        pe->td.gateMPO[1].uuid = "WEAK";
        
        pe->td.gates = {
            {"A1", "A2"}, {"A2", "A3"}, {"A3", "A4"}, {"A4", "A1"},
            {"B1", "B2"}, {"B2", "B3"}, {"B3", "B4"}, {"B4", "B1"}, 
            {"A1", "B1"}, {"A2", "B2"}, {"A3", "B3"}, {"A4", "B4"},
            {"B1", "A1"}, {"B2", "A2"}, {"B3", "A3"}, {"B4", "A4"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0}, {2, 0}, {2, 0},
            {2, 0}, {2, 0}, {2, 0}, {2, 0},
            {3, 1}, {3, 1}, {3, 1}, {3, 1},
            {3, 1}, {3, 1}, {3, 1}, {3, 1}
        };
        
        for (int i=0; i<12; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );
        for (int i=0; i<4; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[1]) );

        std::cout<<"NNH_4x2Cell_Ladder 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

std::unique_ptr<Engine> buildEngine_AKLT(nlohmann::json & json_model) {

    double arg_tau = json_model["tau"].get<double>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_AKLT(arg_tau) );
        
        pe->td.gates = {
            {"A", "B"}, {"B", "A"}, 
            {"C", "D"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };
        
        for (int i=0; i<8; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );

        std::cout<<"AKLT 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_AKLT(arg_tau) );
        

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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


        for (int i=0; i<16; i++) pe->td.ptr_gateMPO.push_back( &(pe->td.gateMPO[0]) );

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

// void getModel_NNH_2x2Cell_AB(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_2site > & gateMPO,
//     std::vector< MPO_2site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds) {

//     double arg_J1 = json_model["J1"].get<double>();
//     double arg_h = json_model["h"].get<double>();
    
//     ptr_model = std::unique_ptr<Model>(new NNHModel_2x2Cell_AB(arg_J1, arg_h));

//     // time step
//     double arg_tau = json_model["tau"].get<double>();
//     // gate sequence
//     std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

//     if (arg_fuGateSeq == "2SITE") {
//         gates = {
//             {"A", "B"}, {"B", "A"},
//             {"A", "B"}, {"B", "A"}
//         };

//         gate_auxInds = {
//             {2, 0}, {2, 0},
//             {3, 1}, {3, 1},
//         };

//         gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, arg_h) );

//         for (int i=0; i<4; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
//     } else {
//         std::cout<<"Unsupported 2-site gate sequence: "<< arg_fuGateSeq << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// void getModel_Ising(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_3site > & gateMPO,
//     std::vector< MPO_3site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds) {

//     double arg_J1     = json_model["J1"].get<double>();
//     double arg_h      = json_model["h"].get<double>();
//     double arg_lambda = json_model["LAMBDA"].get<double>();
    
//     ptr_model = std::unique_ptr<Model>(new IsingModel(arg_J1, arg_h));

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

//         gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
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

//         gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
//         ptr_gateMPO = std::vector< MPO_3site * >(16, &(gateMPO[0]) );
//     } else if (arg_fuGateSeq == "SYM4") {
//         gates = {
//             {"B", "A", "C", "D"},
//             {"D", "C", "A", "B"},

//             {"C", "D", "B", "A"},
//             {"A", "B", "D", "C"},

//             {"A", "B", "D", "C"},
//             {"C", "D", "B", "A"},

//             {"D", "C", "A", "B"},
//             {"B", "A", "C", "D"}
//         };

//         gate_auxInds = {
//             {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3},

//             {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3},

//             {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3},

//             {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3}
//         };
//         gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h/3.0) );
//         ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
//     } else if (arg_fuGateSeq == "2SITE") {
//         gates = {
//             {"A", "B", "D", "C"},
//             {"B", "A", "C", "D"}, 
            
//             {"C", "D", "B", "A"}, 
//             {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)

//             {"A", "C", "D", "B"}, {"B", "D", "C", "A"},

//             {"C", "A", "B", "D"}, {"D", "B", "A", "C"}
//         };

//         gate_auxInds = {
//             {3,2, 0,3, 1,0, 2,1},
//             {3,2, 0,3, 1,0, 2,1},

//             {1,2, 0,1, 3,0, 2,3},
//             {1,2, 0,1, 3,0, 2,3},

//             {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0},
            
//             {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0}
//         };

//         gateMPO.push_back( getMPO3s_Ising_2site(arg_tau, arg_J1, arg_h/4.0) );
//         ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
//     } else {
//         std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

std::unique_ptr<Engine> buildEngine_ISING3BODY(nlohmann::json & json_model) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_J2     = json_model["J2"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_tau    = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_gateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    if (arg_fuGateSeq == "SYM1") {
    	TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1, arg_J2, arg_h) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
        };

        pe->td.gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
        };

        std::cout<<"ISING3BODY SYM1 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } else if (arg_fuGateSeq == "SYM3") {
    	TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1, arg_J2, arg_h) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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

        std::cout<<"ISING3BODY SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

std::unique_ptr<Engine> buildEngine_NNH_2x2Cell_ABCD(nlohmann::json & json_model) {

    double arg_J1  = json_model["J1"].get<double>();
    double arg_h   = json_model["h"].get<double>();
    double arg_del = json_model["del"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_gateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    
    if (arg_fuGateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_NNH_2site(arg_tau, arg_J1, arg_h) );
        pe->td.ptr_gateMPO = std::vector< MPO_2site* >(8, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"A", "B"}, {"B", "A"}, 
            {"C", "D"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };

        std::cout<<"NNH_2x2Cell_ABCD 2SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_ANISJ1J2(arg_tau, arg_J1, 0.0, arg_del) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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

        std::cout<<"ANISOTROPIC NNH SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe ); 
    }
    else if (arg_fuGateSeq == "4SITE") {
        TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_NNH(arg_tau, arg_J1, arg_h, arg_del) );
        pe->td.ptr_gateMPO = std::vector< OpNS * >(4, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"B", "A", "C", "D"},
            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},
            {"D", "C", "A", "B"}
        };

        pe->td.gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1}
        };

        std::cout<<"NNH 4SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

// std::unique_ptr<Model> getModel_Ising_2x2Cell_ABCD(nlohmann::json & json_model) {

//     double arg_J1     = json_model["J1"].get<double>();
//     double arg_h      = json_model["h"].get<double>();
//     double arg_lambda = json_model["LAMBDA"].get<double>();
    
//     return ptr_model = std::unique_ptr<Model>(new IsingModel(arg_J1, arg_h));

//     // // time step
//     // double arg_tau    = json_model["tau"].get<double>();
//     // // gate sequence
//     // std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

   
//     // if (arg_fuGateSeq == "2SITE") {
//     //     gates = {
//     //         {"A", "B"}, {"B", "A"}, 
//     //         {"C", "D"}, {"D", "C"},
//     //         {"A", "C"}, {"B", "D"},
//     //         {"C", "A"}, {"D", "B"}
//     //     };

//     //     gate_auxInds = {
//     //         {2, 0}, {2, 0},
//     //         {2, 0}, {2, 0},
//     //         {3, 1}, {3, 1},
//     //         {3, 1}, {3, 1}
//     //     };

//     //     gateMPO.push_back( getMPO2s_Ising_2site(arg_tau, arg_J1, arg_h/4.0) );
//     //     ptr_gateMPO = std::vector< MPO_2site * >(8, &(gateMPO[0]) );
//     // } else {
//     //     std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
//     //     exit(EXIT_FAILURE);
//     // }
// }

// void getModel_2site(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_2site > & gateMPO,
//     std::vector< MPO_2site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds) {

//     std::string arg_modelType = json_model["type"].get<std::string>(); 

//     if (arg_modelType == "NNH_2x2Cell_AB") {
//         getModel_NNH_2x2Cell_AB(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else if (arg_modelType == "NNH_2x2Cell_ABCD") {
//         getModel_NNH_2x2Cell_ABCD(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else if (arg_modelType == "Ising_2x2Cell_ABCD") {
//         getModel_Ising_2x2Cell_ABCD(json_model, ptr_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
//     } else {
//         std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

std::unique_ptr<Engine> buildEngine_J1J2(nlohmann::json & json_model) {
    
    double arg_J1 = json_model["J1"].get<double>();
    double arg_J2 = json_model["J2"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
    double arg_tau = json_model["tau"].get<double>();
    
    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter",false);

    if (arg_fuGateSeq == "SYM1") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
        };

        pe->td.gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
        };
    
        std::cout<<"J1J2 SYM1 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    else if (arg_fuGateSeq == "SYM2") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (1 AD ABCD)
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (1 BC ABCD) 
            
            {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (2 BC BADC)
            {"D", "C", "A", "B"}, {"A", "B", "D", "C"}, // (2 AD BADC)
            
            {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (3 AD CDAB) 
            {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (3 BC CDAB)

            {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (4 BC DCBA)        
            {"D", "C", "A", "B"}, {"A", "B", "D", "C"}  // (4 AD DCBA)
        };

        pe->td.gate_auxInds = {
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            
            {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 

            {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},        
            {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3}
        };

        std::cout<<"J1J2 SYM2 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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

        std::cout<<"J1J2 SYM3 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    else if (arg_fuGateSeq == "SYM4") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"B", "A", "C", "D"},
            {"D", "C", "A", "B"},

            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},

            {"A", "B", "D", "C"},
            {"C", "D", "B", "A"},

            {"D", "C", "A", "B"}, 
            {"B", "A", "C", "D"}
        };

        pe->td.gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3},

            {3,0, 2,3, 1,2, 0,1},
            {1,0, 2,1, 3,2, 0,3}
        };

        std::cout<<"J1J2 SYM4 ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    }
    else if (arg_fuGateSeq == "4SITE") {
        TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

        pe->td.gateMPO.push_back( getOP4s_J1J2(arg_tau, arg_J1, arg_J2) );
        pe->td.ptr_gateMPO = std::vector< OpNS * >(4, &pe->td.gateMPO[0] );

        pe->td.gates = {
            {"B", "A", "C", "D"},
            {"C", "D", "B", "A"},
            {"A", "B", "D", "C"},
            {"D", "C", "A", "B"}
        };

        pe->td.gate_auxInds = {
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1},
            {3,0, 2,3, 1,2, 0,1}
        };

        std::cout<<"J1J2 4SITE ENGINE constructed"<<std::endl;
        if (arg_symmTrotter) pe->td.symmetrize();
        return std::unique_ptr<Engine>( pe );
    } 
    // else if (arg_fuGateSeq == "2SITE") {
    //     TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

    //     pe->td.gateMPO.push_back( new getMPO3s_NNH_2site(arg_tau, arg_J1, 0.0) );
    //     pe->td.ptr_gateMPO = std::vector< OpNS * >(8, &(gateMPO[0]) );

    //     gates = {
    //         {"A", "B", "D", "C"},
    //         {"B", "A", "C", "D"}, 
            
    //         {"C", "D", "B", "A"}, 
    //         {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)

    //         {"A", "C", "D", "B"}, {"B", "D", "C", "A"},

    //         {"C", "A", "B", "D"}, {"D", "B", "A", "C"}
    //     };

    //     gate_auxInds = {
    //         {3,2, 0,3, 1,0, 2,1},
    //         {3,2, 0,3, 1,0, 2,1},

    //         {1,2, 0,1, 3,0, 2,3},
    //         {1,2, 0,1, 3,0, 2,3},

    //         {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0},
            
    //         {2,3, 1,2, 0,1, 3,0}, {2,3, 1,2, 0,1, 3,0}
    //     };

    //     gateMPO.push_back( new getMPO3s_NNH_2site(arg_tau, arg_J1, 0.0) );
    //     ptr_gateMPO = std::vector< OpNS * >(8, &(gateMPO[0]) );
    // }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

std::unique_ptr<Engine> buildEngine_IDENTITY(nlohmann::json & json_model) {
    
    // gate sequence
    std::string arg_gateSeq = json_model["gateSeq"].get<std::string>();

    // if (arg_fuGateSeq == "SYM1") {
    //     TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_2site>();

    //     pe->td.gateMPO.push_back( getMPO3s_Id(2) );
    //     pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

    //     pe->td.gates = {
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
    //     };

    //     pe->td.gate_auxInds = {
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
    //     };
    
    //     std::cout<<"IDENTITY SYM1 ENGINE constructed"<<std::endl;
    //     return std::unique_ptr<Engine>( pe );
    // } 
    // else if (arg_fuGateSeq == "SYM2") {
    //     TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

    //     pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
    //     pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

    //     pe->td.gates = {
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (1 AD ABCD)
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (1 BC ABCD) 
            
    //         {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (2 BC BADC)
    //         {"D", "C", "A", "B"}, {"A", "B", "D", "C"}, // (2 AD BADC)
            
    //         {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, // (3 AD CDAB) 
    //         {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, // (3 BC CDAB)

    //         {"C", "D", "B", "A"}, {"B", "A", "C", "D"}, // (4 BC DCBA)        
    //         {"D", "C", "A", "B"}, {"A", "B", "D", "C"}  // (4 AD DCBA)
    //     };

    //     pe->td.gate_auxInds = {
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
            
    //         {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 

    //         {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},        
    //         {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3}
    //     };

    //     std::cout<<"J1J2 SYM2 ENGINE constructed"<<std::endl;
    //     return std::unique_ptr<Engine>( pe );
    // }
    if (arg_gateSeq == "SYM3") {
        TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

        pe->td.gateMPO.push_back( getMPO3s_Id_v2(2) );
        pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

        pe->td.gates = {
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

        pe->td.gate_auxInds = {
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

        std::cout<<"IDENTITY 3SITE ENGINE constructed"<<std::endl;
        return std::unique_ptr<Engine>( pe );
    } 
    // else if (arg_fuGateSeq == "SYM4") {
    //     TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

    //     pe->td.gateMPO.push_back( getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda) );
    //     pe->td.ptr_gateMPO = std::vector< MPO_3site * >(16, &pe->td.gateMPO[0] );

    //     pe->td.gates = {
    //         {"B", "A", "C", "D"},
    //         {"D", "C", "A", "B"},

    //         {"C", "D", "B", "A"},
    //         {"A", "B", "D", "C"},

    //         {"A", "B", "D", "C"},
    //         {"C", "D", "B", "A"},

    //         {"D", "C", "A", "B"}, 
    //         {"B", "A", "C", "D"}
    //     };

    //     pe->td.gate_auxInds = {
    //         {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3},

    //         {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3},

    //         {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3},

    //         {3,0, 2,3, 1,2, 0,1},
    //         {1,0, 2,1, 3,2, 0,3}
    //     };

    //     std::cout<<"J1J2 SYM4 ENGINE constructed"<<std::endl;
    //     return std::unique_ptr<Engine>( pe );
    // }
    // else if (arg_fuGateSeq == "4SITE") {
    //     TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

    //     pe->td.gateMPO.push_back( getOP4s_J1J2(arg_tau, arg_J1, arg_J2) );
    //     pe->td.ptr_gateMPO = std::vector< OpNS * >(4, &pe->td.gateMPO[0] );

    //     pe->td.gates = {
    //         {"B", "A", "C", "D"},

    //         {"C", "D", "B", "A"},

    //         {"A", "B", "D", "C"},

    //         {"D", "C", "A", "B"}
    //     };

    //     pe->td.gate_auxInds = {
    //         {3,0, 2,3, 1,2, 0,1},

    //         {3,0, 2,3, 1,2, 0,1},

    //         {3,0, 2,3, 1,2, 0,1},

    //         {3,0, 2,3, 1,2, 0,1}
    //     };

    //     std::cout<<"J1J2 4SITE ENGINE constructed"<<std::endl;
    //     return std::unique_ptr<Engine>( pe );
    // } 
    else if (arg_gateSeq == "2SITE") {
        TrotterEngine<MPO_2site>* pe = new TrotterEngine<MPO_2site>();

        pe->td.gateMPO.push_back( getMPO2s_Id(2) );
        pe->td.ptr_gateMPO = std::vector< MPO_2site* >(8, &pe->td.gateMPO[0]);

        pe->td.gates = {
            {"A", "B"}, {"B", "A"}, 
            {"C", "D"}, {"D", "C"},
            {"A", "C"}, {"B", "D"},
            {"C", "A"}, {"D", "B"}
        };

        pe->td.gate_auxInds = {
            {2, 0}, {2, 0},
            {2, 0}, {2, 0},
            {3, 1}, {3, 1},
            {3, 1}, {3, 1}
        };

        std::cout<<"IDENTITY 2SITE ENGINE constructed"<<std::endl;
        return std::unique_ptr<Engine>( pe );
    }
    else {
        std::cout<<"Unsupported gate sequence: "<< arg_gateSeq << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model) {
    
    std::string arg_modelType = json_model["type"].get<std::string>();

    if(arg_modelType == "J1J2") {
        return buildEngine_J1J2(json_model);
    } else if (arg_modelType == "NNH_2x2Cell_Ladder") {
        return buildEngine_NNH_2x2Cell_Ladder(json_model);
    } else if (arg_modelType == "NNH_4x2Cell_Ladder") {
        return buildEngine_NNH_4x2Cell_Ladder(json_model);
    } else if (arg_modelType == "AKLT") {
        return buildEngine_AKLT(json_model);
    // } else if (arg_modelType == "Ising") {
    //     return getModel_Ising(json_model);
    } else if (arg_modelType == "Ising3Body") {
        return buildEngine_ISING3BODY(json_model);
    // } else if (arg_modelType == "NNH_2x2Cell_AB") {
    //     return getModel_NNH_2x2Cell_AB(json_model);
    } else if (arg_modelType == "NNH_2x2Cell_ABCD") {
        return buildEngine_NNH_2x2Cell_ABCD(json_model);
    // } else if (arg_modelType == "Ising_2x2Cell_ABCD") {
    //     return getModel_Ising_2x2Cell_ABCD(json_model);
    } else if (arg_modelType == "IDENTITY") {
        return buildEngine_IDENTITY(json_model);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }

    return nullptr;
}

std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model, 
    LinSysSolver * solver) {

    std::string arg_modelType = json_model["type"].get<std::string>();

    std::unique_ptr<Engine> pE;
    if(arg_modelType == "J1J2") {
        pE = buildEngine_J1J2(json_model);
    } else if (arg_modelType == "NNH_2x2Cell_Ladder") {
        pE = buildEngine_NNH_2x2Cell_Ladder(json_model);
    } else if (arg_modelType == "NNH_4x2Cell_Ladder") {
        pE = buildEngine_NNH_4x2Cell_Ladder(json_model);
    } else if (arg_modelType == "AKLT") {
        pE = buildEngine_AKLT(json_model);
    // } else if (arg_modelType == "Ising") {
    //     return getModel_Ising(json_model);
    } else if (arg_modelType == "Ising3Body") {
        pE = buildEngine_ISING3BODY(json_model);
    // } else if (arg_modelType == "NNH_2x2Cell_AB") {
    //     return getModel_NNH_2x2Cell_AB(json_model);
    } else if (arg_modelType == "NNH_2x2Cell_ABCD") {
        pE = buildEngine_NNH_2x2Cell_ABCD(json_model);
    // } else if (arg_modelType == "Ising_2x2Cell_ABCD") {
    //     return getModel_Ising_2x2Cell_ABCD(json_model);
    } else if (arg_modelType == "IDENTITY") {
        pE = buildEngine_IDENTITY(json_model);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }

    pE->pSolver = solver;
    return pE;
}

// template<class T>
// Args TrotterEngine<T>::performSimpleUpdate(Cluster & cls, Args const& args) 
// {
//     auto gi = td.nextCyclicIndex();

//     return simpleUpdate(*td.ptr_gateMPO[gi], cls, td.gates[gi], td.gate_auxInds[gi], args);
// }

template<> Args TrotterEngine<MPO_2site>::performSimpleUpdate(
    Cluster & cls, Args const& args) {

    auto gi = td.nextCyclicIndex();

    return simpleUpdate(*td.ptr_gateMPO[gi], cls, td.gates[gi], td.gate_auxInds[gi], args);
}

template<> Args TrotterEngine<MPO_3site>::performSimpleUpdate(
    Cluster & cls, Args const& args) {

    auto gi = td.nextCyclicIndex();

    return simpleUpdate(*td.ptr_gateMPO[gi], cls, td.gates[gi], td.gate_auxInds[gi], args);
}

template<> Args TrotterEngine<OpNS>::performSimpleUpdate(
    Cluster & cls, Args const& args) {

    auto gi = td.nextCyclicIndex();

    return Args::global();
}


template<class T> Args TrotterEngine<T>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    return Args::global();
}

template<> Args TrotterEngine<MPO_2site>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    auto gi = td.nextCyclicIndex();

    // return fullUpdate_ALS2S_IT(*td.ptr_gateMPO[gi], cls, ctmEnv,
    //     td.gates[gi], td.gate_auxInds[gi], *(this->pSolver), args);
    return fullUpdate_2S(*td.ptr_gateMPO[gi], cls, ctmEnv,
        td.gates[gi], td.gate_auxInds[gi], *(this->pSolver), args);
}

template<> Args TrotterEngine<MPO_3site>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    auto gi = td.nextCyclicIndex();
    std::cout<<"Current index: "<< gi << std::endl;

    return fullUpdate_ALS3S_IT(*td.ptr_gateMPO[gi], cls, ctmEnv,
        td.gates[gi], td.gate_auxInds[gi], *(this->pSolver), args);
}

template<> Args TrotterEngine<OpNS>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    auto gi = td.nextCyclicIndex();
    std::cout<<"Current index: "<< gi << std::endl;

    if ( td.ptr_gateMPO[gi]->nSite != 4 ) {
        std::cout<<"Unsupported OpNS: nSite = "<< td.ptr_gateMPO[gi]->nSite << std::endl;
        exit(EXIT_FAILURE);
    }

    // return fullUpdate_ALS4S_LSCG_IT(*td.ptr_gateMPO[gi], cls, ctmEnv,
    //     td.gates[gi], td.gate_auxInds[gi], args);
    return fullUpdate_CG_full4S(*td.ptr_gateMPO[gi], cls, ctmEnv,
        td.gates[gi], td.gate_auxInds[gi], args);
}