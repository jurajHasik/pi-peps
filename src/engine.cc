#include "engine.h"

using namespace itensor;

// std::unique_ptr<Engine> buildEngine_ISING3BODY(nlohmann::json & json_model) {

//     double arg_J1     = json_model["J1"].get<double>();
//     double arg_J2     = json_model["J2"].get<double>();
//     double arg_h      = json_model["h"].get<double>();
//     double arg_tau    = json_model["tau"].get<double>();
    
//     // gate sequence
//     std::string arg_gateSeq = json_model["fuGateSeq"].get<std::string>();

//     // symmetrize Trotter Sequence
//     bool arg_symmTrotter = json_model.value("symmTrotter",false);

//     // gate sequence
//     std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

//     if (arg_fuGateSeq == "SYM1") {
//     	TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

//         pe->td.gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1, arg_J2, arg_h) );
//         pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

//         pe->td.gates = {
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (1 AD ABCD)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (1 BC ABCD) 
            
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (2 AD BADC)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (2 BC BADC)

//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (3 AD CDAB) 
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}, //{"B", "D", "C", "A"}, // (3 BC CDAB)
            
//             {"A", "B", "D", "C"}, {"D", "C", "A", "B"}, //{"A", "C", "D", "B"}, // (4 AD DCBA)
//             {"B", "A", "C", "D"}, {"C", "D", "B", "A"}  //{"B", "D", "C", "A"}  // (4 BC DCBA)
//         };

//         pe->td.gate_auxInds = {
//             {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},
//             {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},

//             {3,0, 2,3, 1,2, 0,1}, {1,2, 0,1, 3,0, 2,3},
//             {3,2, 0,3, 1,0, 2,1}, {1,0, 2,1, 3,2, 0,3},

//             {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1},
//             {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1}, 
           
//             {1,0, 2,1, 3,2, 0,3}, {3,2, 0,3, 1,0, 2,1},
//             {1,2, 0,1, 3,0, 2,3}, {3,0, 2,3, 1,2, 0,1}
//         };

//         std::cout<<"ISING3BODY SYM1 ENGINE constructed"<<std::endl;
//         if (arg_symmTrotter) pe->td.symmetrize();
//         return std::unique_ptr<Engine>( pe );
//     } else if (arg_fuGateSeq == "SYM3") {
//     	TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

//         pe->td.gateMPO.push_back( getMPO3s_Ising3Body(arg_tau, arg_J1, arg_J2, arg_h) );
//         pe->td.ptr_gateMPO = std::vector< MPO_3site* >(16, &pe->td.gateMPO[0] );

//         pe->td.gates = {
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

//         pe->td.gate_auxInds = {
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

//         std::cout<<"ISING3BODY SYM3 ENGINE constructed"<<std::endl;
//         if (arg_symmTrotter) pe->td.symmetrize();
//         return std::unique_ptr<Engine>( pe );
//     } else {
//         std::cout<<"Unsupported gate sequence: "<< arg_fuGateSeq << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     return nullptr;
// }

// std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model) {
    
//     std::string arg_modelType = json_model["type"].get<std::string>();

//     if (arg_modelType == "Ising3Body") {
//         return buildEngine_ISING3BODY(json_model);
//     } else {
//         std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     return nullptr;
// }

// std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model, 
//     LinSysSolver * solver) {

//     std::string arg_modelType = json_model["type"].get<std::string>();

//     std::unique_ptr<Engine> pE;
//     if (arg_modelType == "Ising3Body") {
//         pE = buildEngine_ISING3BODY(json_model);
//     } else {
//         std::cout<<"Unsupported model: "<< arg_modelType << std::endl;
//         exit(EXIT_FAILURE);
//     }

//     pE->pSolver = solver;
//     return pE;
// }

// template<class T>
// Args TrotterEngine<T>::performSimpleUpdate(Cluster & cls, Args const& args) 
// {
//     auto gi = td.nextCyclicIndex();

//     return simpleUpdate(*td.ptr_gateMPO[gi], cls, td.gates[gi], td.gate_auxInds[gi], args);
// }

template<> Args TrotterEngine<MPO_2site>::performSimpleUpdate(
    Cluster & cls, Args const& args) {

    auto gi = td.nextCyclicIndex();

    // TODO transitional
    std::vector<std::string> tmp_siteId_seq = {
        cls.vertexToId(td.tgates[gi].init_vertex), 
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0])
    };

    //     1                          1
    //     |                          |
    //  0--A=[0,0]--2--Shift(1,0)--0--B=[1,0]--2
    //     |                          |
    //     3                          3 
    std::vector<int> tmp_auxIndsDir_seq = {
        dirFromShift(td.tgates[gi].disp[0]),
        dirFromShift( -1 * td.tgates[gi].disp[0] )
    };

    return simpleUpdate(*td.tgates[gi].ptr_gate, cls, tmp_siteId_seq, tmp_auxIndsDir_seq, args);
    // NEW_INTERFACE return simpleUpdate(tgates[gi], args);
}

template<> Args TrotterEngine<MPO_3site>::performSimpleUpdate(
    Cluster & cls, Args const& args) {

    auto gi = td.nextCyclicIndex();

    // TODO transitional
    std::vector<std::string> tmp_siteId_seq = {
        cls.vertexToId(td.tgates[gi].init_vertex), 
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0] 
            + td.tgates[gi].disp[1]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[1])
    };

    //     1                          1
    //     |       (1)            (2) |
    //  0--A=[0,0]--2--Shift(1,0)--0--B=[1,0]--2
    //     |                          |
    //  (0)3                       (3)3
    //  +Shift(0,1)               +Shift(0,1)
    //  (7)1                       (4)1
    //     |       (6)            (7) |
    //  0--C=[0,1]--2--Shift(1,0)--0--D=[1,1]--2
    //     |                          |
    //     3                          3
    //
    std::vector<int> tmp_auxIndsDir_seq = {
        dirFromShift(-1 * td.tgates[gi].disp[1]),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[1] ),
        dirFromShift( td.tgates[gi].disp[1] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[1] )
    };

    return simpleUpdate(*td.tgates[gi].ptr_gate, cls, tmp_siteId_seq, tmp_auxIndsDir_seq, args);
    // NEW_INTERFACE return simpleUpdate(tgates[gi], args);
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

    // TODO transitional
    std::vector<std::string> tmp_siteId_seq = {
        cls.vertexToId(td.tgates[gi].init_vertex), 
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0])
    };

    //     1                          1
    //     |                          |
    //  0--A=[0,0]--2--Shift(1,0)--0--B=[1,0]--2
    //     |                          |
    //     3                          3 
    std::vector<int> tmp_auxIndsDir_seq = {
        dirFromShift(td.tgates[gi].disp[0]),
        dirFromShift( -1 * td.tgates[gi].disp[0] )
    };

    return fullUpdate_ALS2S_IT(*td.tgates[gi].ptr_gate, cls, ctmEnv, 
        tmp_siteId_seq, tmp_auxIndsDir_seq, 
        *(this->pSolver), args);
    // NEW_INTERFACE return fullUpdate_ALS2S_IT(tgates[gi], cls, ctmEnv, *(this->pSolver), args);
    
    // return fullUpdate_2S(*td.ptr_gateMPO[gi], cls, ctmEnv,
    //     td.gates[gi], td.gate_auxInds[gi], *(this->pSolver), args);
}

template<> Args TrotterEngine<MPO_3site>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    auto gi = td.nextCyclicIndex();

    // TODO transitional
    std::vector<std::string> tmp_siteId_seq = {
        cls.vertexToId(td.tgates[gi].init_vertex), 
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0] 
            + td.tgates[gi].disp[1]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[1])
    };

    //     1                          1
    //     |       (1)            (2) |
    //  0--A=[0,0]--2--Shift(1,0)--0--B=[1,0]--2
    //     |                          |
    //  (0)3                       (3)3
    //  +Shift(0,1)               +Shift(0,1)
    //  (7)1                       (4)1
    //     |       (6)            (7) |
    //  0--C=[0,1]--2--Shift(1,0)--0--D=[1,1]--2
    //     |                          |
    //     3                          3
    //
    std::vector<int> tmp_auxIndsDir_seq = {
        dirFromShift(-1 * td.tgates[gi].disp[1]),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[1] ),
        dirFromShift( td.tgates[gi].disp[1] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[1] )
    };

    return fullUpdate_ALS3S_IT(*td.tgates[gi].ptr_gate, cls, ctmEnv, 
        tmp_siteId_seq, tmp_auxIndsDir_seq, *(this->pSolver), args);
    // NEW_INTERFACE return fullUpdate_ALS3S_IT(tgates[gi], cls, ctmEnv, *(this->pSolver), args);
}

template<> Args TrotterEngine<OpNS>::performFullUpdate(
    Cluster & cls, CtmEnv const& ctmEnv, Args const& args) {

    auto gi = td.nextCyclicIndex();

    if ( td.tgates[gi].ptr_gate->nSite != 4 ) {
        std::cout<<"Unsupported OpNS: nSite = "<< td.tgates[gi].ptr_gate->nSite << std::endl;
        exit(EXIT_FAILURE);
    }

    // TODO transitional
    std::vector<std::string> tmp_siteId_seq = {
        cls.vertexToId(td.tgates[gi].init_vertex), 
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[0] 
            + td.tgates[gi].disp[1]),
        cls.vertexToId(td.tgates[gi].init_vertex + td.tgates[gi].disp[1])
    };

    //     1                          1
    //     |       (1)            (2) |
    //  0--A=[0,0]--2--Shift(1,0)--0--B=[1,0]--2
    //     |                          |
    //  (0)3                       (3)3
    //  +Shift(0,1)               +Shift(0,1)
    //  (7)1                       (4)1
    //     |       (6)            (7) |
    //  0--C=[0,1]--2--Shift(1,0)--0--D=[1,1]--2
    //     |                          |
    //     3                          3
    //
    std::vector<int> tmp_auxIndsDir_seq = {
        dirFromShift(-1 * td.tgates[gi].disp[1]),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift(-1 * td.tgates[gi].disp[1] ),
        dirFromShift( td.tgates[gi].disp[1] ),
        dirFromShift(-1 * td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[0] ),
        dirFromShift( td.tgates[gi].disp[1] )
    };
    
    return fullUpdate_CG_full4S(*td.tgates[gi].ptr_gate, cls, ctmEnv,
        tmp_siteId_seq, tmp_auxIndsDir_seq, args);
    // NEW_INTERFACE return fullUpdate_CG_full4S(tgates[gi], cls, ctmEnv, args);
    
    // LEGACY return fullUpdate_ALS4S_LSCG_IT(*td.ptr_gateMPO[gi], cls, ctmEnv,
    //     td.gates[gi], td.gate_auxInds[gi], args);
}