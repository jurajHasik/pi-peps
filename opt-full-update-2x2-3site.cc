#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "ctm-cluster-io.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "full-update.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

	std::string arg_initFile = std::string(argv[1]);
	std::ifstream infile(arg_initFile, std::ios::in);

	nlohmann::json jsonCls;
	infile >> jsonCls;

    // write simulation parameters to log file
    std::cout << jsonCls.dump(4) << std::endl;

	//read cluster infile OR initialize by one of the predefined
	//options FILE, RND, RND_AB, AFM, RVB, ...
	std::string initBy(jsonCls["initBy"].get<std::string>());

	int physDim, auxBondDim;
	std::string inClusterFile;
	if (initBy=="FILE") inClusterFile = jsonCls["inClusterFile"].get<std::string>();
	physDim = jsonCls["physDim"].get<int>();
	auxBondDim = jsonCls["auxBondDim"].get<int>();

	//read cluster outfile
	std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

	//define Hamiltonian
	double arg_J1 = jsonCls["J1"].get<double>();
	double arg_J2 = jsonCls["J2"].get<double>();
    double arg_lambda = jsonCls["LAMBDA"].get<double>();

	//read simulation parameters

	//total iterations of full update
	std::string arg_fuGateSeq = jsonCls["fuGateSeq"].get<std::string>();
    int arg_fuIter = jsonCls["fuIter"].get<int>();
    std::string arg_fuIsoInit = jsonCls["fuIsoInit"].get<std::string>();
    double arg_fuIsoInitNoiseLevel = jsonCls["fuIsoInitNoiseLevel"].get<double>();
    int arg_maxAltLstSqrIter = jsonCls["maxAltLstSqrIter"].get<int>();
    double pseudoInvCutoff = jsonCls["pseudoInvCutoff"].get<double>();
    double isoEpsilon = jsonCls["isoEpsilon"].get<double>();
    std::string arg_otNormType = jsonCls["otNormType"].get<std::string>();
    bool arg_fuDbg = jsonCls["fuDbg"].get<bool>();
    int arg_fuDbgLevel = jsonCls["fuDbgLevel"].get<int>();
	//time step
	double arg_tau = jsonCls["tau"].get<double>();

	//env parameters
	int auxEnvDim = jsonCls["auxEnvDim"].get<int>();
	std::string arg_ioEnvTag(jsonCls["ioEnvTag"].get<std::string>());
	CtmEnv::init_env_type arg_initEnvType(toINIT_ENV(jsonCls["initEnvType"].get<std::string>()));
	bool envIsComplex = jsonCls["envIsComplex"].get<bool>();
	CtmEnv::isometry_type iso_type(toISOMETRY(jsonCls["isoType"].get<std::string>()));
	double arg_isoPseudoInvCutoff = jsonCls["isoPseudoInvCutoff"].get<double>();
    CtmEnv::normalization_type norm_type(toNORMALIZATION(jsonCls["normType"].get<std::string>()));

	//max iterations for environment per full update step
	int arg_maxEnvIter = jsonCls["maxEnvIter"].get<int>();
    double arg_envEps  = jsonCls["envEpsilon"].get<double>();
    bool arg_reinitEnv = jsonCls["reinitEnv"].get<bool>();

	// INITIALIZE CLUSTER
	Cluster cls;
	if (initBy=="FILE") {
		Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
		ITensor A, B, C, D;

		cls = readCluster(inClusterFile);
		if (cls.auxBondDim > auxBondDim) std::cout <<"Warning: auxBondDim of the"
			<<" input cluster is higher then the desired one!" << std::endl;
		cls.auxBondDim = auxBondDim;
		if (cls.physDim != physDim) std::cout <<"Warning: physDim of the"
			<<" input cluster and simulation parameters are not in agreement!"
			<< std::endl;

		A = cls.sites[cls.cToS.at(std::make_pair(0,0))];
        B = cls.sites[cls.cToS.at(std::make_pair(1,0))];
        C = cls.sites[cls.cToS.at(std::make_pair(0,1))];
        D = cls.sites[cls.cToS.at(std::make_pair(1,1))];

        pIA = noprime(findtype(A.inds(), PHYS));
        pIB = noprime(findtype(B.inds(), PHYS));
        pIC = noprime(findtype(C.inds(), PHYS));
        pID = noprime(findtype(D.inds(), PHYS));
        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        
        // relabel original indices to take up desired bond dimensions
        auto taIA = noprime(findtype(A.inds(), AUXLINK));
        auto taIB = noprime(findtype(B.inds(), AUXLINK));
        auto taIC = noprime(findtype(C.inds(), AUXLINK));
        auto taID = noprime(findtype(D.inds(), AUXLINK));
        
        auto D_I = delta(taIA,aIA);
        A = A*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIB,aIB);
        B = B*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIC,aIC);
        C = C*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taID,aID);
        D = D*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};
	} else {
        Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
		ITensor A, B, C, D;

        // ----- DEFINE BLANK CLUSTER ----------------------------------
        cls = Cluster();
        cls.sizeN = 2;
        cls.sizeM = 2;
        cls.auxBondDim = auxBondDim;
        cls.physDim    = physDim;

        cls.siteIds = std::vector< std::string >(4);
        cls.siteIds = { "A", "B", "C", "D" };

        cls.cToS  = {
            {std::make_pair(0,0),"A"},
            {std::make_pair(1,0),"B"},
            {std::make_pair(0,1),"C"},
            {std::make_pair(1,1),"D"}
        };

        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        pIA = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIB = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

        A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
        B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);
        C = ITensor(aIC, prime(aIC,1), prime(aIC,2), prime(aIC,3), pIC);
        D = ITensor(aID, prime(aID,1), prime(aID,2), prime(aID,3), pID);

        if (initBy == "RND_AB") {
            std::cout <<"Initializing by RANDOM TENSORS A,B,C=B,D=A"<< std::endl;
            randomize(A);
            randomize(B);

            auto shift05 = [](double r){ return r-0.5; };
            A.apply(shift05);
            B.apply(shift05);

            C = B * delta(pIB, pIC);
            D = A * delta(pIA, pID);
            for (int i=0; i<=3; ++i) {
                C = C * delta(prime(aIB,i), prime(aIC,i));
                D = D * delta(prime(aIA,i), prime(aID,i));
            }
        } else if(initBy == "RANDOM") {
            std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;
            // Randomize

            randomize(A);
            randomize(B);
            randomize(C);
            randomize(D);

            // auto shift05 = [](Real r){ return r-0.5; };
            // A.apply(shift05);
            // B.apply(shift05);
            // C.apply(shift05);
            // D.apply(shift05);

        } else if (initBy == "AFM") {
            std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;
            // Spin DOWN on site A, spin   UP on site B
            // Spin UP   on site C, spin DOWN on site D
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(2), 1.0);
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0);
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(1), 1.0);
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(2), 1.0);
        }

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};
        // ----- END DEFINE CLUSTER ------------------------------------
    }

    // INITIALIZE ENVIRONMENT
    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, cls, 
        {"isoPseudoInvCutoff",arg_isoPseudoInvCutoff});
    switch (arg_initEnvType) {
        case CtmEnv::INIT_ENV_const1: {
            ctmEnv.initMockEnv();
            break;
        }
        case CtmEnv::INIT_ENV_ctmrg: {
            ctmEnv.initCtmrgEnv();
            break;
        }
        case CtmEnv::INIT_ENV_rnd: {
           	ctmEnv.initRndEnv(envIsComplex);
            break;
        } 
        // case CtmEnv::INIT_ENV_file: {
        //     io_env_fmt_type ioEnvFmt = toIO_ENV_FMT(std::string(argv[5]));
        //     std::string in_files_prefix = std::string(argv[6]);
            
        //     ctmEnv = CtmEnv("TEST_ENV_2x2", 
        //         readEnv_V2(ioEnvFmt, in_files_prefix, cluster),
        //         cluster);
        //     break;
        // }
    }
    
    // INITIALIZE EXPECTATION VALUE BUILDER
    EVBuilder ev(arg_ioEnvTag, cls, ctmEnv.getCtmData_DBG());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

    // SETUP OPTIMIZATION LOOP
    // Get Exp of 3-site operator u_123 - building block of Trotter Decomposition
    MPO_3site uJ1J2(getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2, arg_lambda));
//    MPO_3site uJ1J2(getMPO3s_Id_v2(physDim));

    // hold energies
    std::vector<double> e_curr(4,0.0), e_prev(4,0.0);
    std::vector<double> e_nnH;
    std::vector<double> e_nnH_AC;
    std::vector<double> e_nnH_BD;
    std::vector<double> e_nnH_CD;
    std::vector<double> evNN;

    std::vector<double> accT(8,0.0); // holds timings for CTM moves
    std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // DEFINE GATE SEQUENCE
    std::vector< std::vector<std::string> > gates;
    std::vector< std::vector<int> > gate_auxInds;
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

    // For symmetric Trotter decomposition
    int init_gate_size = gates.size();
    for (int i=0; i<init_gate_size; i++) {
        gates.push_back(gates[init_gate_size-1-i]);
        gate_auxInds.push_back(gate_auxInds[init_gate_size-1-i]);
    }

    // STORE ISOMETRIES
    std::vector< std::vector< ITensor > > iso_store(
        gates.size(), {ITensor(), ITensor(), ITensor(), ITensor()} );

    Args fuArgs = {
        "maxAltLstSqrIter",arg_maxAltLstSqrIter,
        "fuDbg",arg_fuDbg,
        "fuDbgLevel",arg_fuDbgLevel,
        "fuIsoInit",arg_fuIsoInit,
        "fuIsoInitNoiseLevel",arg_fuIsoInitNoiseLevel,
        "pseudoInvCutoff",pseudoInvCutoff,
        "isoEpsilon",isoEpsilon,
        "otNormType",arg_otNormType
    };
    // Diagnostic data
    std::vector<int> diag_ctmIter;
    std::vector< Args > diagData_fu;
    Args diag_fu;

    std::ofstream out_file_energy(outClusterFile+".energy.dat", std::ios::out);
    std::ofstream out_file_diag(outClusterFile+".diag.dat", std::ios::out);
    out_file_energy.precision( std::numeric_limits< double >::max_digits10 );
    out_file_diag.precision( std::numeric_limits< double >::max_digits10 );

    // ENTER OPTIMIZATION LOOP
    for (int fuI = 1; fuI <= arg_fuIter; fuI++) {
    	std::cout <<"Full Update - STEP "<< fuI << std::endl;

    	// SETUP ENVIRONMENT LOOP
    	t_begin_int = std::chrono::steady_clock::now();
		
    	// ENTER ENVIRONMENT LOOP
		for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {

	        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
	        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
	        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
	        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

	        if ( envI % 1 == 0 ) {
	            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
	        
	            t_end_int = std::chrono::steady_clock::now();

                e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
                e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
                e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
                e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

	            std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
	            	<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
	                <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                    << e_curr[3] << std::endl;

                // if the difference between energies along NN links is lower then arg_envEps
                // consider the environment converged
                if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                    (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                    (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                    (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                    diag_ctmIter.push_back(envI);
                    std::cout<< "ENV CONVERGED" << std::endl;
                    break;
                }

                if (envI==arg_maxEnvIter) {
                    diag_ctmIter.push_back(envI);
                    // diagnose spectra
                    
                }
                e_prev = e_curr;

                t_begin_int = std::chrono::steady_clock::now();
            }
	    }

	    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        	<<" "<< accT[3] << std::endl;
    	std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        	<<" "<< accT[7] << std::endl;

        // compute energies
        e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(0,0), std::make_pair(1,0)) );
        e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(0,0), std::make_pair(0,1)) );
        e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(1,0), std::make_pair(1,1)) );
        e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(0,1), std::make_pair(1,1)) );

        evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(1,0), std::make_pair(2,0))); //BA
        evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(0,1), std::make_pair(0,2))); //CA
        evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(1,1), std::make_pair(1,2))); //DB
        evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            std::make_pair(1,1), std::make_pair(2,1))); //DC

        // write energy
        double avgE_8links = 0.;
        out_file_energy << fuI-1 <<" "<< e_nnH.back() 
            <<" "<< e_nnH_AC.back()
            <<" "<< e_nnH_BD.back()
            <<" "<< e_nnH_CD.back();
            for ( unsigned int j=evNN.size()-4; j<evNN.size(); j++ ) {
                avgE_8links += evNN[j];
                out_file_energy<<" "<< evNN[j];
            }
            avgE_8links = avgE_8links + e_nnH.back() + e_nnH_AC.back() 
                + e_nnH_BD.back() + e_nnH_CD.back();
        out_file_energy <<" "<< avgE_8links/8.0;
        out_file_energy << std::endl;

        // PERFORM FULL UPDATE
        std::cout << "GATE: " << (fuI-1)%gates.size() << std::endl;
            
        diag_fu = fullUpdate(uJ1J2, cls, ctmEnv, gates[(fuI-1)%gates.size()], 
            gate_auxInds[(fuI-1)%gates.size()], iso_store[(fuI-1)%gates.size()], fuArgs);

        diagData_fu.push_back(diag_fu);

        out_file_diag << fuI <<" "<< diag_ctmIter.back() <<" "<< diag_fu.getInt("alsSweep",0)
            <<" "<< diag_fu.getString("siteMaxElem")
            <<" "<< diag_fu.getReal("finalDist0",0.0)
            <<" "<< diag_fu.getReal("finalDist1",0.0);
        if (arg_fuDbg && (arg_fuDbgLevel >=1))
            out_file_diag <<" "<< diag_fu.getReal("ratioNonSymLE",0.0)
            <<" "<< diag_fu.getReal("ratioNonSymFN",0.0);
        out_file_diag  <<std::endl;

        ctmEnv.updateCluster(cls);
        // reset environment
        if (arg_reinitEnv) 
            switch (arg_initEnvType) {
                case CtmEnv::INIT_ENV_const1: {
                    ctmEnv.initMockEnv();
                    break;
                }
                case CtmEnv::INIT_ENV_ctmrg: {
                    ctmEnv.initCtmrgEnv();
                    break;
                }
                case CtmEnv::INIT_ENV_rnd: {
                    ctmEnv.initRndEnv(envIsComplex);
                    break;
                } 
            }
        ev.setCluster(cls);
        writeCluster(outClusterFile, cls);
    }

    // // FULL UPDATE FINISHED - COMPUTING FINAL ENVIRONMENT
    std::cout <<"FULL UPDATE DONE - COMPUTING FINAL ENVIRONMENT "<< std::endl;
    // t_begin_int = std::chrono::steady_clock::now();
        
    // ENTER ENVIRONMENT LOOP
    for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

        if ( envI % 1 == 0 ) {
             ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                << e_curr[3] << std::endl;
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

            if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                std::cout<< "ENV CONVERGED" << std::endl;
                break;
            }

            e_prev = e_curr;
            t_begin_int = std::chrono::steady_clock::now();
        }
    }

    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;

    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

	// Compute final properties
    e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) );
    e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) );
    e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(1,1)) );
    e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(1,1)) );

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(2,1))); //DC
    

    // write energy
    double avgE_8links = 0.;
    out_file_energy << arg_fuIter <<" "<< e_nnH.back() 
        <<" "<< e_nnH_AC.back()
        <<" "<< e_nnH_BD.back()
        <<" "<< e_nnH_CD.back();
        for ( unsigned int j=evNN.size()-4; j<evNN.size(); j++ ) {
            avgE_8links += evNN[j];
            out_file_energy<<" "<< evNN[j];
        }
        avgE_8links = avgE_8links + e_nnH.back() + e_nnH_AC.back() 
            + e_nnH_BD.back() + e_nnH_CD.back();
    out_file_energy <<" "<< avgE_8links/8.0;
    out_file_energy << std::endl;

    std::cout <<"FU_ITER: "<<" E:"<< std::endl;
    for ( unsigned int i=0; i<e_nnH.size(); i++ ) {
        std::cout << i <<" "<< e_nnH[i] 
            <<" "<< e_nnH_AC[i]
            <<" "<< e_nnH_BD[i]
            <<" "<< e_nnH_CD[i];
            for ( unsigned int j=i*4; j<(i+1)*4; j++ ) {
                std::cout<<" "<< evNN[j];
            }
        std::cout<< std::endl;
    }

	// std::cout <<"BA: "<< evNN[0] <<" CA: "<< evNN[1] <<" DB: "<< evNN[2] 
    //     <<" DC: "<< evNN[3] << std::endl;

    std::cout <<"ID: "<< ev.eV_1sO_1sENV(EVBuilder::MPO_Id, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ2: "<< ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z2, std::make_pair(0,0)) << std::endl;

    std::vector<double> sA_zpm;
    std::vector<double> sB_zpm;
    std::vector<double> sC_zpm;
    std::vector<double> sD_zpm;

    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0)));
    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0)));
    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0)));
    std::cout<<"S_A: "<< sA_zpm[0] <<", "<< sA_zpm[1] <<", "<< sA_zpm[2] << std::endl;

    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0)));
    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0)));
    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0)));
    std::cout<<"S_B: "<< sB_zpm[0] <<", "<< sB_zpm[1] <<", "<< sB_zpm[2] << std::endl;

    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1)));
    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1)));
    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1)));
    std::cout<<"S_C: "<< sC_zpm[0] <<", "<< sC_zpm[1] <<", "<< sC_zpm[2] << std::endl;

    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1)));
    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1)));
    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1)));
    std::cout<<"S_D: "<< sD_zpm[0] <<", "<< sD_zpm[1] <<", "<< sD_zpm[2] << std::endl;

    // Store final new cluster
    writeCluster(outClusterFile, cls);

    // write out diagnostic data of full-update
    for (int i=0; i<arg_fuIter; i++) {
        std::cout<< i <<" "<< diag_ctmIter[i] <<" "<< diagData_fu[i].getInt("alsSweep",0)
            <<" "<< diagData_fu[i].getString("siteMaxElem")
            <<" "<< diagData_fu[i].getReal("finalDist0",0.0)
            <<" "<< diagData_fu[i].getReal("finalDist1",0.0);
        if (arg_fuDbg && (arg_fuDbgLevel >=1))
            std::cout<<" "<< diagData_fu[i].getReal("ratioNonSymLE",0.0)
            <<" "<< diagData_fu[i].getReal("ratioNonSymFN",0.0);
        std::cout<<std::endl;
    }
}