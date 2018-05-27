#ifndef __MODELS_
#define __MODELS_

#include <string>
#include "json.hpp"
#include "full-update.h"

void getModel_J1J2(nlohmann::json & json_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_J2 = json_model["J2"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
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
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

	double arg_J1 = json_model["J1"].get<double>();
	double arg_alpha = json_model["alpha"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
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

        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_J1, 0.0) );
        gateMPO.push_back( getMPO3s_Uladder_v2(arg_tau, arg_alpha*arg_J1, 0.0) );

        for (int i=0; i<6; i++) ptr_gateMPO.push_back( &(gateMPO[0]) );
        for (int i=0; i<2; i++) ptr_gateMPO.push_back( &(gateMPO[1]) );
    } 
    else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getModel_Ising(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    double arg_J1     = json_model["J1"].get<double>();
    double arg_h      = json_model["h"].get<double>();
    double arg_lambda = json_model["LAMBDA"].get<double>();
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

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
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

        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
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
        gateMPO.push_back( getMPO3s_Ising_v2(arg_tau, arg_J1, arg_h) );
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

        gateMPO.push_back( getMPO3s_Ising_2site(arg_tau, arg_J1, arg_h) );
        ptr_gateMPO = std::vector< MPO_3site * >(8, &(gateMPO[0]) );
    } else {
        std::cout<<"Unsupported 3-site gate sequence: "<< arg_fuGateSeq << std::endl;
        exit(EXIT_FAILURE);
    }
}


void getModel(nlohmann::json & json_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds) {

    std::string arg_modelType = json_model["type"].get<std::string>(); 

    if(arg_modelType == "J1J2") {
        getModel_J1J2(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "NNHLadder") {
        getModel_NNHLadder(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else if (arg_modelType == "Ising") {
        getModel_Ising(json_model, gateMPO, ptr_gateMPO, gates, gate_auxInds);
    } else {
        std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif